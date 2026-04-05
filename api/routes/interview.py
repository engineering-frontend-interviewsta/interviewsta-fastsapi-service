"""
Interview API endpoints

All blocking I/O (Redis, Celery AsyncResult) is run off the event loop via run_sync()
to keep the event loop free and avoid 502s / health-check timeouts under load.
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.responses import StreamingResponse
from typing import Any, Dict, Optional
import logging
import asyncio
import json
import time
from datetime import datetime

from tasks.interview_tasks import (
    process_interview_start,
    process_user_response_with_transcription
)

from schemas.interview import (
    InterviewStartRequest,
    InterviewStartResponse,
    UserResponseRequest,
    UserResponseSubmitResponse,
    RespondTaskStatusResponse,
    InterviewStatusResponse,
    VideoQualityData,
    InterviewStartStatusResponse,
    InterviewVideoTelemetrySample,
)
from api.dependencies import (
    get_current_user,
    get_redis,
    verify_token_from_query,
    get_interview_access_payload,
    get_interview_access_payload_from_token,
)
from schemas.interview import InterviewAccessTokenPayload
from services.interview_session import InterviewSessionManager
from celery.result import AsyncResult
from tasks.interview_tasks import process_interview_start
from tasks.audio_tasks import transcribe_audio, synthesize_speech
from redis import Redis
from tasks.celery_app import celery_app
from schemas.feedback import FeedbackStatusResponse
from services.interview_test_loader import (
    apply_interview_test_to_payload,
    fetch_interview_test_by_id,
    interview_test_row_is_active,
)


logger = logging.getLogger(__name__)

router = APIRouter()

# POST /start stores the Celery task id here so GET /{session_id}/stream can tail progress
# without passing task_id in the URL (single long-lived EventSource).
PENDING_START_TASK_TTL_SEC = 900

# Client sends telemetry ~every 20s; store time-ordered samples per session (newest at Redis list head).
VIDEO_TELEMETRY_SAMPLES_TTL_SEC = 3600
VIDEO_TELEMETRY_MAX_SAMPLES = 500


def _video_telemetry_samples_key(session_id: str) -> str:
    return f"session:{session_id}:video_telemetry_samples"


def _video_telemetry_environment_key(session_id: str) -> str:
    """One-shot attire / environment snapshot (first non-null payload wins)."""
    return f"session:{session_id}:video_telemetry_environment"


def _pending_start_task_key(session_id: str) -> str:
    return f"session:{session_id}:pending_start_task"


def _user_identifier(user_info: Dict[str, Any]) -> str:
    """Canonical user id for session ownership (from Bearer: uid/sub then email)."""
    return user_info.get("uid") or user_info.get("sub") or user_info.get("email") or ""


async def _run_sync(sync_fn, *args, **kwargs):
    """Run blocking sync work in thread pool to keep event loop free"""
    return await asyncio.to_thread(sync_fn, *args, **kwargs)


def _get_start_status_data_sync(
    task_id: str,
    redis_client: Redis,
    user_email: str
) -> Dict[str, Any]:
    """
    Get start interview task status (sync, run in thread)
    Optimized: No sleeps, single pass through Redis
    """
    task_result = AsyncResult(task_id, app=celery_app)
    
    # Map Celery states to our status
    state_mapping = {
        "PENDING": "queued",
        "STARTED": "processing",
        "RETRY": "processing",
        "PROGRESS": "processing",
        "SUCCESS": "completed",
        "FAILURE": "failed",
    }
    status_str = state_mapping.get(task_result.state, "processing")
    
    # Get progress from PROGRESS state
    progress = 0
    message = None
    
    if task_result.state == "PROGRESS":
        meta = task_result.info or {}
        progress = meta.get("progress", 0)
        message = meta.get("message")
    elif task_result.state == "SUCCESS":
        progress = 100
        message = "Ready"
    elif task_result.state == "FAILURE":
        message = "Failed"
    
    # Extract result and session_id
    session_id = None
    result = None
    error = None
    
    if task_result.state == "SUCCESS":
        task_data = task_result.result
        if isinstance(task_data, dict):
            session_id = task_data.get("session_id")
            result = {
                "status": task_data.get("status"),
                "message": task_data.get("message"),
                "last_node": task_data.get("last_node"),
            }
    elif task_result.state == "FAILURE":
        error = str(task_result.info) if task_result.info else "Task failed"
    
    # Get interview data from Redis (only if completed)
    interview_status = None
    interview_ai_response = None
    interview_transcript = None
    interview_is_complete = None
    interview_warning = None
    
    if session_id and task_result.state == "SUCCESS":
        try:
            session_manager = InterviewSessionManager(redis_client)
            session = session_manager.get_session(session_id)
            session_user = session.get("user_id") if session else None

            logger.warning(f"🔍 DEBUG: Comparing users (email-based)")
            logger.warning(f"   session.user_id = {session_user}")
            logger.warning(f"   user_email = {user_email}")
            logger.warning(f"   Match? {session_user == user_email}")
            
            if session and session_user == user_email:
                current_status = session_manager.get_status(session_id) or "waiting_for_response"
                
                # Check processing flag
                processing_key = f"session:{session_id}:processing"
                if redis_client.get(processing_key):
                    current_status = "processing"
                
                interview_status = current_status
                
                # Get response data
                response_data = session_manager.get_response(session_id)
                
                if response_data:
                    interview_ai_response = {
                        "message": response_data.get("message"),
                        "audio": response_data.get("audio"),
                        "audio_base64": response_data.get("audio"),
                        "last_node": response_data.get("last_node"),
                        "timestamp": response_data.get("timestamp"),
                        "question_number": response_data.get("question_number"),
                        "total_questions": response_data.get("total_questions"),
                        "question_raw_content": response_data.get("question_raw_content"),
                    }
                    if session.get("interview_type") in ("Company", "Subject", "Technical"):
                        interview_ai_response["interview_questions"] = (session.get("payload") or {}).get("Questions")
                    interview_status = "ai_responded"
                
                interview_transcript = session_manager.get_transcript(session_id)
                interview_is_complete = current_status == "completed"
                interview_warning = session_manager.get_warning(session_id)
        except Exception as e:
            logger.warning(f"Error getting session data for {session_id}: {e}")
    
    return {
        "task_id": task_id,
        "session_id": session_id,
        "status": status_str,
        "progress": progress,
        "message": message,
        "result": result,
        "error": error,
        "interview_status": interview_status,
        "interview_ai_response": interview_ai_response,
        "interview_transcript": interview_transcript,
        "interview_is_complete": interview_is_complete,
        "interview_warning": interview_warning,
    }


def _get_respond_status_data_sync(
    session_id: str,
    task_id: str,
    redis_client: Redis,
    user_email: str
) -> Dict[str, Any]:
    """
    Get respond task status (sync, run in thread)
    Optimized: No sleeps, single pass through Redis
    """
    try:
        session_manager = InterviewSessionManager(redis_client)
        session = session_manager.get_session(session_id)
        
        # Validate session exists and user has access
        if not session:
            return {
                "error": {
                    "code": 404,
                    "detail": "Session not found"
                }
            }
        
        if session.get("user_id") != user_email:
            return {
                "error": {
                    "code": 403,
                    "detail": "Not authorized to access this session"
                }
            }
        
        # Get task status
        task_result = AsyncResult(task_id, app=celery_app)
        
        state_mapping = {
            "PENDING": "queued",
            "STARTED": "processing",
            "RETRY": "processing",
            "PROGRESS": "processing",
            "SUCCESS": "completed",
            "FAILURE": "failed",
        }
        status_str = state_mapping.get(task_result.state, "processing")
        
        # Get progress
        progress = 0
        progress_message = None
        
        if task_result.state == "PROGRESS":
            meta = task_result.info or {}
            progress = meta.get("progress", 0)
            progress_message = meta.get("message")
        elif task_result.state == "SUCCESS":
            progress = 100
        
        # Extract result
        result = None
        error = None
        
        if task_result.state == "SUCCESS":
            task_data = task_result.result
            if isinstance(task_data, dict):
                result = {
                    "status": task_data.get("status"),
                    "message": task_data.get("message"),
                    "last_node": task_data.get("last_node"),
                }
        elif task_result.state == "FAILURE":
            error = str(task_result.info) if task_result.info else "Task failed"
        
        # Get interview data from Redis
        interview_status = None
        interview_ai_response = None
        interview_transcript = None
        interview_is_complete = None
        interview_warning = None
        
        try:
            current_status = session_manager.get_status(session_id) or "waiting_for_response"
            
            # Check processing flag
            processing_key = f"session:{session_id}:processing"
            if redis_client.get(processing_key):
                current_status = "processing"
            
            interview_status = current_status
            
            # Get response data (only if completed or status is ai_responded)
            if task_result.state == "SUCCESS" or current_status == "ai_responded":
                response_data = session_manager.get_response(session_id)
                
                if response_data:
                    interview_ai_response = {
                        "message": response_data.get("message"),
                        "audio": response_data.get("audio"),
                        "audio_base64": response_data.get("audio"),
                        "last_node": response_data.get("last_node"),
                        "timestamp": response_data.get("timestamp"),
                        # "interview_ai_response": response_data.get("interview_ai_response"),
                        "question_number": response_data.get("question_number"),
                        "total_questions": response_data.get("total_questions"),
                        "question_raw_content": response_data.get("question_raw_content"),
                    }

                    session_data = session_manager.get_session(session_id)
                    if session_data:
                        # Add phase data to response
                        interview_ai_response["currentspeaking"] = session_data.get("current_speaking")
                        interview_ai_response["speakingfeedback"] = session_data.get("speaking_feedback")
                        interview_ai_response["currentcomprehension"] = session_data.get("current_comprehension")
                        interview_ai_response["comprehensionfeedback"] = session_data.get("comprehension_feedback")
                        interview_ai_response["currentmcq"] = session_data.get("current_mcq")
                        interview_ai_response["mcqfeedback"] = session_data.get("mcq_feedback")
                    if session_data and session_data.get("interview_type") in ("Company", "Subject", "Technical"):
                        interview_ai_response["interview_questions"] = (session_data.get("payload") or {}).get("Questions")
                    interview_status = "ai_responded"
            
            interview_transcript = session_manager.get_transcript(session_id)
            interview_is_complete = current_status == "completed"
            interview_warning = session_manager.get_warning(session_id)
            
        except Exception as e:
            logger.warning(f"Error getting interview data for {session_id}: {e}")
        
        return {
            "task_id": task_id,
            "session_id": session_id,
            "status": status_str,
            "progress": progress,
            "progress_message": progress_message,
            "result": result,
            "error": error,
            "interview_status": interview_status,
            "interview_ai_response": interview_ai_response,
            "interview_transcript": interview_transcript,
            "interview_is_complete": interview_is_complete,
            "interview_warning": interview_warning,
        }
        
    except Exception as e:
        logger.error(f"Error in _get_respond_status_data_sync: {e}", exc_info=True)
        return {
            "error": {
                "code": 500,
                "detail": f"Internal error: {str(e)}"
            }
        }


# API Endpoints

@router.post("/start", response_model=InterviewStartResponse)
async def start_interview(
    request: InterviewStartRequest,
    user_info: Dict = Depends(get_current_user),
    interview_access: InterviewAccessTokenPayload = Depends(get_interview_access_payload),
    redis_client: Redis = Depends(get_redis),
):
    """
    Start a new interview session
    Optimized: Returns immediately after queuing, ~0.3s response time
    
    Flow:
    1. Queue Celery task (async, no blocking)
    2. Return task_id immediately
    3. Client polls GET /start-status/{task_id}
    """
    try:
        # interview_type and user_id from decoded tokens (not body)
        interview_type = (
            getattr(interview_access, "fastapi_interview_type", None) or request.interview_type
        )
        if not interview_type:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="interview_type is required: set fastapiInterviewType in X-Interview-Access-Token",
            )
        user_id = request.user_id or user_info.get("uid") or user_info.get("sub") or user_info.get("email")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="user_id could not be resolved from Bearer token (sub/uid/email)",
            )
        logger.info(f"Starting {interview_type} interview for user {user_info.get('email')}")
        # Merge feedback_item_id and interview_test_id from X-Interview-Access-Token into payload
        payload = dict(request.payload or {})
        if getattr(interview_access, "feedback_item_id", None):
            payload["feedback_item_id"] = interview_access.feedback_item_id
        if getattr(interview_access, "interview_test_id", None) is not None:
            payload["interview_test_id"] = interview_access.interview_test_id
            row = await fetch_interview_test_by_id(str(interview_access.interview_test_id))
            if row is not None and not interview_test_row_is_active(row):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="This interview test is not active",
                )
            if row is not None:
                db_type = row.get("fastapi_interview_type")
                if db_type and db_type != interview_type:
                    logger.warning(
                        "JWT fastapiInterviewType=%r differs from interview_tests row %s (%r)",
                        interview_type,
                        interview_access.interview_test_id,
                        db_type,
                    )
                apply_interview_test_to_payload(payload, interview_type, row)
            else:
                logger.warning(
                    "interview_test_id %r not found in interview_tests; using payload defaults",
                    interview_access.interview_test_id,
                )
        task = process_interview_start.apply_async(
            args=[request.session_id, interview_type, user_id, payload],
            queue="interview",
        )
        
        logger.info(f"Interview task queued: {task.id}")
        redis_client.setex(
            _pending_start_task_key(request.session_id),
            PENDING_START_TASK_TTL_SEC,
            task.id,
        )

        # Return immediately
        return InterviewStartResponse(
            task_id=task.id,
            session_id=request.session_id,
            status="queued",
            message="Interview initialization queued"
        )
        
    except Exception as e:
        logger.error(f"Error starting interview: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start interview: {str(e)}"
        )


@router.get("/start-status/{task_id}", response_model=InterviewStartStatusResponse)
async def get_start_interview_status(
    task_id: str,
    user_info: Dict = Depends(get_current_user),
    interview_access: InterviewAccessTokenPayload = Depends(get_interview_access_payload),
    redis_client: Redis = Depends(get_redis),
):
    """
    Poll status of start interview task
    Optimized: No sleeps, returns immediately, ~0.1s response time
    
    Returns:
    - Task state (queued, processing, completed, failed)
    - Progress (0-100)
    - When completed: full interview snapshot with audio
    """
    try:
        data = await _run_sync(
            _get_start_status_data_sync,
            task_id,
            redis_client,
            _user_identifier(user_info),
        )
        
        return InterviewStartStatusResponse(**data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting start status: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.post("/{session_id}/respond", response_model=UserResponseSubmitResponse)
async def submit_response(
    session_id: str,
    request: UserResponseRequest,
    user_info: Dict = Depends(get_current_user),
    interview_access: InterviewAccessTokenPayload = Depends(get_interview_access_payload),
    redis_client: Redis = Depends(get_redis),
):
    """
    Submit user response to interview question.
    Optimized: Returns immediately after queuing, ~0.3s response time.

    Accepts either audio (for speech) or text only:
    - **audio_data**: for speaking phases (e.g. Communication speaking); will be transcribed.
    - **text_response**: for text-only phases (e.g. Communication comprehension/writing).
      You can send only text_response with no audio — no transcription is run.

    Flow:
    1. Validate session exists (fast Redis check)
    2. Check processing flag
    3. Queue pipeline task (transcribe if audio, then process + audio)
    4. Return task_id immediately
    5. Client polls GET /{session_id}/respond-status/{task_id}
    """
    try:
        logger.info(f"Submitting response for session {session_id}")
        
        # Fast validation (run in thread to avoid blocking event loop)
        def validate_session():
            session_manager = InterviewSessionManager(redis_client)
            session = session_manager.get_session(session_id)
            
            if not session:
                return {"error": {"code": 404, "detail": "Session not found"}}
            if session.get("user_id") != _user_identifier(user_info):
                return {"error": {"code": 403, "detail": "Not authorized to access this session"}}
            
            # Check processing flag
            processing_key = f"session:{session_id}:processing"
            if redis_client.get(processing_key):
                return {
                    "error": {
                        "code": 429,
                        "detail": "Previous response is still being processed. Please wait."
                    }
                }
            
            # Set processing flag (TTL: 30 seconds)
            redis_client.setex(processing_key, 30, "true")
            
            return {"valid": True}
        
        validation_result = await _run_sync(validate_session)
        
        if "error" in validation_result:
            err = validation_result["error"]
            raise HTTPException(status_code=err["code"], detail=err["detail"])
        
        # Validate input: at least one of audio or text (text-only is valid e.g. for comprehension phase)
        has_audio = bool(request.audio_data and request.audio_data.strip())
        has_text = bool(request.text_response is not None and str(request.text_response).strip())
        if not has_audio and not has_text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Provide at least one of: audio_data (for speech) or text_response (for text/writing, e.g. Communication comprehension phase)"
            )
        
        # Queue complete pipeline task (non-blocking)
        task = process_user_response_with_transcription.apply_async(
            args=[
                session_id,
                request.audio_data,
                request.text_response,
                request.code_input,
                request.skip_audio
            ],
            queue="interview"
        )
        
        logger.info(f"Response task queued: {task.id}")
        
        # Return immediately
        return UserResponseSubmitResponse(
            task_id=task.id,
            session_id=session_id,
            status="processing"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting response: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process response: {str(e)}"
        )


@router.get("/{session_id}/respond-status/{task_id}", response_model=RespondTaskStatusResponse)
async def get_respond_task_status(
    session_id: str,
    task_id: str,
    user_info: Dict = Depends(get_current_user),
    interview_access: InterviewAccessTokenPayload = Depends(get_interview_access_payload),
    redis_client: Redis = Depends(get_redis),
):
    """
    Poll status of respond task
    Optimized: No sleeps, returns immediately, ~0.1s response time
    
    Returns:
    - Task state (queued, processing, completed, failed)
    - Progress (0-100) with stage messages
    - When completed: full AI response with audio
    """
    try:
        data = await _run_sync(
            _get_respond_status_data_sync,
            session_id,
            task_id,
            redis_client,
            _user_identifier(user_info),
        )
        
        # Validate data
        if not data:
            logger.error("_get_respond_status_data_sync returned None")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve status data"
            )
        
        # Check for error response
        if "error" in data and data["error"]:
            err = data["error"]
            if isinstance(err, dict):
                error_code = err.get("code", 500)
                error_detail = err.get("detail", str(e))
            else:
                error_code = 500
                error_detail = str(err)
            
            raise HTTPException(status_code=error_code, detail=error_detail)
        
        return RespondTaskStatusResponse(**data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting respond status: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get status: {str(e)}",
        )

@router.post("/{session_id}/video-quality")
async def submit_video_quality(
    session_id: str,
    data: VideoQualityData,
    user_info: Dict = Depends(get_current_user),
    interview_access: InterviewAccessTokenPayload = Depends(get_interview_access_payload),
    redis_client: Redis = Depends(get_redis),
):
    """
    Submit video quality and behavioral metrics
    
    Used for soft skills tracking (gaze, confidence, nervousness, etc.)
    Also triggers warnings for video quality issues (face detection, engagement, etc.)
    """
    try:
        session_manager = InterviewSessionManager(redis_client)
        
        # Verify session ownership
        session = session_manager.get_session(session_id)
        if not session or session["user_id"] != _user_identifier(user_info):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)
        
        # Store metrics for aggregation
        metrics_key = f"session:{session_id}:video_metrics"
        metrics_data = data.dict()
        redis_client.lpush(metrics_key, json.dumps(metrics_data))
        redis_client.expire(metrics_key, 3600)
        
        # Track video strikes for face detection (like old code)
        strikes_key = f"session:{session_id}:video_strikes"
        current_strikes = int(redis_client.get(strikes_key) or 0)
        
        # Check face status and increment strikes
        face_status = metrics_data.get("face", "ok")
        warning_message = None
        warning_type = None
        
        if face_status != "ok":
            current_strikes += 1
            redis_client.setex(strikes_key, 3600, current_strikes)
            
            if current_strikes == 1:
                warning_message = "I am unable to see you, can I ask you to kindly come back in frame and stay still."
                warning_type = "face_detection"
            elif current_strikes == 3:
                warning_message = "This is the final warning before I disconnect the call, I ask you again to kindly come back in frame and proceed with the interview"
                warning_type = "face_detection_final"
            elif current_strikes >= 5:
                warning_message = "Alright seems like you are unable to make into the frame, in that case - thanks for joining the interview and I will be ending our call now."
                warning_type = "face_detection_terminate"
                # Store termination flag for SSE stream to pick up
                session_manager.set_warning(session_id, warning_type, warning_message)
                return {
                    "status": "accepted",
                    "message": "Video quality data recorded",
                    "warning": {
                        "type": warning_type,
                        "message": warning_message
                    },
                    "terminate": True
                }
        else:
            # Reset strikes if face is detected
            if current_strikes > 0:
                redis_client.delete(strikes_key)
        
        # Check behavioral metrics for engagement/distraction warnings (every 10 samples)
        if face_status == "ok":
            # Get recent metrics for aggregation
            recent_metrics = redis_client.lrange(metrics_key, 0, 9)  # Last 10 samples
            if len(recent_metrics) >= 10:
                engagement_values = []
                distraction_values = []
                
                for metric_str in recent_metrics:
                    try:
                        metric = json.loads(metric_str)
                        if metric.get("engagement") is not None:
                            engagement_values.append(metric["engagement"])
                        if metric.get("distraction") is not None:
                            distraction_values.append(metric["distraction"])
                    except:
                        continue
                
                if engagement_values and distraction_values:
                    avg_engagement = sum(engagement_values) / len(engagement_values)
                    avg_distraction = sum(distraction_values) / len(distraction_values)
                    
                    # Send warning if engagement is low or distraction is high
                    if avg_engagement < 50 or avg_distraction > 70:
                        warning_message = "Please stay attentive and maintain eye contact with the camera."
                        warning_type = "engagement"
        
        # Store warning if any
        if warning_message and warning_type:
            session_manager.set_warning(session_id, warning_type, warning_message)
        
        return {
            "status": "accepted",
            "message": "Video quality data recorded",
            "warning": {
                "type": warning_type,
                "message": warning_message
            } if warning_message else None
        }
        
    except Exception as e:
        logger.error(f"Error submitting video quality: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/{session_id}/video-telemetry")
async def post_video_telemetry(
    session_id: str,
    payload: InterviewVideoTelemetrySample,
    user_info: Dict = Depends(get_current_user),
    interview_access: InterviewAccessTokenPayload = Depends(get_interview_access_payload),
    redis_client: Redis = Depends(get_redis),
):
    """
    Append one telemetry sample for the session (client typically POSTs every ~20s).

    Body: nested JSON with camelCase keys (``time``, ``duration``, ``environment``, ``audio``,
    ``background``, ``camera``, ``lighting``, ``presence``, ``speech``, ``overallScore``,
    ``suggestions``, ``criticalIssues``, …). Nested objects may be null or partial.

    ``environment`` is stored **once** (first non-null value wins, ``SET NX``); it is **not**
    repeated on each interval. Per-tick fields (lighting, camera, presence, speech, …) are
    appended to Redis list ``session:{session_id}:video_telemetry_samples`` (newest first),
    **without** the ``environment`` key, trimmed to ``VIDEO_TELEMETRY_MAX_SAMPLES``, TTL
    ``VIDEO_TELEMETRY_SAMPLES_TTL_SEC``.
    """
    try:
        session_manager = InterviewSessionManager(redis_client)
        session = session_manager.get_session(session_id)
        if not session or session["user_id"] != _user_identifier(user_info):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to access this session",
            )

        stored = payload.model_dump(mode="json", by_alias=True, exclude_none=False)
        env_val = stored.get("environment")
        if env_val is not None:
            env_key = _video_telemetry_environment_key(session_id)
            redis_client.set(
                env_key,
                json.dumps(env_val, default=str),
                ex=VIDEO_TELEMETRY_SAMPLES_TTL_SEC,
                nx=True,
            )

        series_payload = {k: v for k, v in stored.items() if k != "environment"}
        key = _video_telemetry_samples_key(session_id)
        redis_client.lpush(key, json.dumps(series_payload, default=str))
        redis_client.ltrim(key, 0, VIDEO_TELEMETRY_MAX_SAMPLES - 1)
        redis_client.expire(key, VIDEO_TELEMETRY_SAMPLES_TTL_SEC)

        count = redis_client.llen(key)
        return {
            "status": "accepted",
            "message": "Video telemetry sample recorded",
            "count": count,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error storing video telemetry for session {session_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.get("/{session_id}/video-telemetry")
async def get_video_telemetry(
    session_id: str,
    user_info: Dict = Depends(get_current_user),
    interview_access: InterviewAccessTokenPayload = Depends(get_interview_access_payload),
    redis_client: Redis = Depends(get_redis),
):
    """
    Return one-shot ``environment`` (if stored) plus time-series ``samples``, oldest first.
    """
    try:
        session_manager = InterviewSessionManager(redis_client)
        session = session_manager.get_session(session_id)
        if not session or session["user_id"] != _user_identifier(user_info):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to access this session",
            )

        env_key = _video_telemetry_environment_key(session_id)
        env_raw = redis_client.get(env_key)
        environment = None
        if env_raw:
            try:
                environment = json.loads(env_raw)
            except json.JSONDecodeError:
                logger.warning("Invalid video_telemetry environment JSON for session %s", session_id)

        key = _video_telemetry_samples_key(session_id)
        raw = redis_client.lrange(key, 0, -1)
        samples = []
        for item in reversed(raw):
            try:
                samples.append(json.loads(item))
            except json.JSONDecodeError:
                logger.warning("Skipping invalid video telemetry JSON for session %s", session_id)

        return {
            "status": "ok",
            "environment": environment,
            "count": len(samples),
            "samples": samples,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reading video telemetry for session {session_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.get("/{session_id}/stream")
async def stream_interview_status(
    session_id: str,
    token: str,  # Bearer token as query parameter (EventSource doesn't support headers)
    interview_access_token: str = Query(..., description="Same JWT as X-Interview-Access-Token (required for SSE)"),
    redis_client: Redis = Depends(get_redis),
):
    """
    Server-Sent Events stream for real-time interview updates

    Note: Token and interview_access_token must be passed as query parameters because
    EventSource doesn't support custom headers.

    Use one long-lived connection for the whole interview. After ``POST /start`` (same ``session_id``),
    the server stores the Celery task id in Redis; this stream picks it up and emits:

    - ``progress``: same shape as polling ``GET /start-status/{task_id}`` (0–100 + message).
    - ``ai_responded`` / ``ai_response``: when the greeting is ready (same as completed start-status).

    No task id in the query string — open the stream before or after ``POST /start``.

    Other events:
    - transcription, status, complete, quality_warning, error
    """
    # Verify Bearer token from query parameter
    try:
        user_info = await verify_token_from_query(token)
    except HTTPException as e:
        async def error_generator():
            yield f"event: error\ndata: {json.dumps({'error': e.detail})}\n\n"
        return StreamingResponse(
            error_generator(),
            media_type="text/event-stream"
        )
    # Verify interview access token (same JWT as X-Interview-Access-Token header)
    try:
        await get_interview_access_payload_from_token(interview_access_token)
    except HTTPException as e:
        async def error_generator():
            yield f"event: error\ndata: {json.dumps({'error': e.detail})}\n\n"
        return StreamingResponse(
            error_generator(),
            media_type="text/event-stream"
        )
    async def event_generator():
        """Generate SSE events"""
        try:
            logger.info(f"SSE stream connecting for session {session_id}")
            session_manager = InterviewSessionManager(redis_client)
            user_email = _user_identifier(user_info)

            last_status = None
            last_response_time = None

            def _read_pending_start_task_id_sync() -> Optional[str]:
                raw = redis_client.get(_pending_start_task_key(session_id))
                if raw is None:
                    return None
                return raw.decode() if isinstance(raw, bytes) else str(raw)

            def _clear_pending_start_task_sync() -> None:
                redis_client.delete(_pending_start_task_key(session_id))

            # Wait for POST /start to set pending task id, or detect existing session (reconnect / no new start)
            discover_deadline = time.monotonic() + 120.0
            start_task_id: Optional[str] = None
            while time.monotonic() < discover_deadline:
                start_task_id = await _run_sync(_read_pending_start_task_id_sync)
                if start_task_id:
                    break
                existing = await _run_sync(
                    lambda: session_manager.get_session(session_id),
                )
                if existing:
                    logger.info(
                        "SSE: session %s already exists; skip start-task tail",
                        session_id,
                    )
                    break
                await asyncio.sleep(0.25)

            # Tail start task on this same connection: progress → ai_responded (same as GET /start-status)
            if start_task_id:
                tail_deadline = time.monotonic() + 120.0
                last_progress_key: Optional[tuple] = None
                start_tail_ok = False
                while time.monotonic() < tail_deadline:
                    data = await _run_sync(
                        _get_start_status_data_sync,
                        start_task_id,
                        redis_client,
                        user_email,
                    )
                    sid = data.get("session_id")
                    if sid and sid != session_id:
                        logger.warning(
                            "SSE start task session mismatch: path=%s task=%s",
                            session_id,
                            sid,
                        )
                        await _run_sync(_clear_pending_start_task_sync)
                        yield f"event: error\ndata: {json.dumps({'error': 'Start task does not match this session_id'})}\n\n"
                        return

                    prog_key = (data.get("status"), data.get("progress"), data.get("message"))
                    if prog_key != last_progress_key:
                        last_progress_key = prog_key
                        progress_payload = {
                            "task_id": start_task_id,
                            "session_id": sid or session_id,
                            "status": data.get("status"),
                            "progress": data.get("progress", 0),
                            "message": data.get("message"),
                            "error": data.get("error"),
                        }
                        yield f"event: progress\ndata: {json.dumps(progress_payload)}\n\n"

                    if data.get("status") == "failed":
                        await _run_sync(_clear_pending_start_task_sync)
                        yield f"event: error\ndata: {json.dumps({'error': data.get('error') or 'Interview start failed', 'fatal': True})}\n\n"
                        return

                    if data.get("status") == "completed" and data.get("interview_ai_response"):
                        full = {
                            "task_id": start_task_id,
                            "session_id": session_id,
                            "status": "completed",
                            "progress": 100,
                            "message": data.get("message"),
                            "result": data.get("result"),
                            "error": data.get("error"),
                            "interview_status": data.get("interview_status"),
                            "interview_ai_response": data.get("interview_ai_response"),
                            "interview_transcript": data.get("interview_transcript"),
                            "interview_is_complete": data.get("interview_is_complete"),
                            "interview_warning": data.get("interview_warning"),
                        }
                        yield f"event: ai_responded\ndata: {json.dumps(full)}\n\n"

                        iar = data["interview_ai_response"]
                        formatted_response = {
                            "message": iar.get("message"),
                            "audio": iar.get("audio"),
                            "audio_base64": iar.get("audio_base64"),
                            "last_node": iar.get("last_node"),
                            "timestamp": iar.get("timestamp"),
                            "question_number": iar.get("question_number"),
                            "total_questions": iar.get("total_questions"),
                            "question_raw_content": iar.get("question_raw_content"),
                        }
                        if iar.get("interview_questions") is not None:
                            formatted_response["interview_questions"] = iar.get("interview_questions")
                        yield f"event: ai_response\ndata: {json.dumps(formatted_response)}\n\n"

                        last_response_time = iar.get("timestamp")
                        session_manager.set_status(session_id, "waiting_for_response")
                        await _run_sync(_clear_pending_start_task_sync)
                        start_tail_ok = True
                        break

                    await asyncio.sleep(0.35)

                if not start_tail_ok:
                    await _run_sync(_clear_pending_start_task_sync)
                    logger.warning(
                        "SSE start task timed out for session %s (task_id=%s)",
                        session_id,
                        start_task_id,
                    )
                    yield f"event: error\ndata: {json.dumps({'error': 'Start task timed out or greeting not ready', 'fatal': True})}\n\n"
                    return

            # Wait for session to be created (with timeout)
            session = None
            max_wait = 20 if not start_task_id else 5
            wait_interval = 0.5
            waited = 0.0

            while waited < max_wait:
                session = session_manager.get_session(session_id)
                if session:
                    logger.info(f"SSE: Session {session_id} found after {waited}s")
                    break
                await asyncio.sleep(wait_interval)
                waited += wait_interval

            if not session:
                logger.warning(f"SSE: Session {session_id} not found after {max_wait}s timeout")
                yield f"event: error\ndata: {json.dumps({'error': 'Session not found or timed out'})}\n\n"
                return

            if session["user_id"] != user_email:
                logger.warning(f"SSE: Unauthorized access attempt to session {session_id}")
                yield f"event: error\ndata: {json.dumps({'error': 'Unauthorized'})}\n\n"
                return

            logger.info(f"SSE: Stream established for session {session_id}")

            # Poll for updates every second
            while True:
                try:
                    current_status = session_manager.get_status(session_id)

                    if current_status != last_status:
                        last_status = current_status
                        yield f"event: status\ndata: {json.dumps({'status': current_status})}\n\n"

                        if current_status == "completed":
                            yield f"event: complete\ndata: {json.dumps({'status': 'completed'})}\n\n"
                            break

                    if current_status == "ai_responded":
                        response_data = session_manager.get_response(session_id)
                        if response_data and response_data.get("timestamp") != last_response_time:
                            last_response_time = response_data.get("timestamp")

                            formatted_response = {
                                "message": response_data.get("message"),
                                "audio": response_data.get("audio"),
                                "audio_base64": response_data.get("audio"),
                                "last_node": response_data.get("last_node"),
                                "timestamp": response_data.get("timestamp"),
                                "question_number": response_data.get("question_number"),
                                "total_questions": response_data.get("total_questions"),
                                "question_raw_content": response_data.get("question_raw_content"),
                            }
                            sess = session_manager.get_session(session_id)
                            if sess and sess.get("interview_type") in ("Company", "Subject", "Technical"):
                                formatted_response["interview_questions"] = (sess.get("payload") or {}).get("Questions")

                            responded = {
                                "task_id": None,
                                "session_id": session_id,
                                "status": "completed",
                                "progress": 100,
                                "message": None,
                                "interview_status": "ai_responded",
                                "interview_ai_response": dict(formatted_response),
                                "interview_transcript": session_manager.get_transcript(session_id),
                                "interview_is_complete": False,
                                "interview_warning": session_manager.get_warning(session_id),
                            }
                            yield f"event: ai_responded\ndata: {json.dumps(responded)}\n\n"
                            yield f"event: ai_response\ndata: {json.dumps(formatted_response)}\n\n"

                            session_manager.set_status(session_id, "waiting_for_response")
                    
                    # Check for new transcription
                    transcript = session_manager.get_transcript(session_id)
                    if transcript:
                        # Send transcript once, then clear it to prevent infinite loop
                        yield f"event: transcription\ndata: {json.dumps({'text': transcript})}\n\n"
                        # Clear transcript after sending (like old code behavior)
                        session_manager.set_transcript(session_id, "")
                    
                    # Check for video quality warnings
                    warning = session_manager.get_warning(session_id)
                    if warning:
                        yield f"event: quality_warning\ndata: {json.dumps(warning)}\n\n"
                        # If warning is termination, close stream
                        if warning.get("type") == "face_detection_terminate":
                            yield f"event: error\ndata: {json.dumps({'error': warning.get('message'), 'fatal': True})}\n\n"
                            break
                    
                    await asyncio.sleep(1)  # Poll every second
                    
                except Exception as e:
                    logger.error(f"Error in SSE stream: {e}", exc_info=True)
                    yield f"event: error\ndata: {json.dumps({'error': str(e), 'fatal': False})}\n\n"
                    # Don't break on non-fatal errors, continue polling
                    await asyncio.sleep(1)
                    continue
                    
        except Exception as e:
            logger.error(f"Fatal error in SSE generator: {e}", exc_info=True)
            yield f"event: error\ndata: {json.dumps({'error': str(e), 'fatal': True})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.post("/end")
async def end_interview(
    request: Dict[str, Any],
    user_info: Dict = Depends(get_current_user),
    interview_access: InterviewAccessTokenPayload = Depends(get_interview_access_payload),
    redis_client: Redis = Depends(get_redis),
):
    """
    End an interview session and trigger feedback generation
    
    This endpoint:
    1. Marks the session as ended
    2. Stores session metadata (duration, interview_type, etc.)
    3. Triggers feedback generation if session_finished is True
    4. Returns success response
    """
    try:
        session_id = request.get("session_id")
        interview_type = request.get("interview_type")
        interview_test_id = request.get("interview_test_id")
        duration = request.get("duration", 0)
        session_finished = request.get("session_finished", False)
        
        if not session_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="session_id is required"
            )
        
        session_manager = InterviewSessionManager(redis_client)
        
        # Verify ownership
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )
        
        if session["user_id"] != _user_identifier(user_info):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to access this session"
            )
        
        # Mark session as ended
        session_manager.set_status(session_id, "completed")
        
        # Build session metadata: only set interview_test_id if request sent a valid value,
        # otherwise keep the value stored at start (so Company/Subject don't get overwritten with null).
        # Persist feedback_item_id from X-Interview-Access-Token for feedback pipeline (feedback_items.json).
        updates = {
            "duration": int(duration) if isinstance(duration, (int, str)) else 0,
            "interview_type": interview_type,
            "session_finished": session_finished,
            "ended_at": datetime.utcnow().isoformat()
        }
        try:
            tid = int(interview_test_id) if interview_test_id is not None else getattr(interview_access, "interview_test_id", None)
            if tid is not None:
                updates["interview_test_id"] = tid
        except (TypeError, ValueError):
            pass
        if getattr(interview_access, "feedback_item_id", None):
            updates["feedback_item_id"] = interview_access.feedback_item_id
        session_manager.update_session(session_id, updates)

        # Telemetry scoring (stub technical rubric); logs full ScoredFeedback JSON
        try:
            from services.telemetry_scoring import log_telemetry_scoring_at_session_end

            dur_min = float(duration) / 60.0 if duration else 0.0
            log_telemetry_scoring_at_session_end(
                redis_client,
                session_id,
                dur_min if dur_min > 0 else 1.0,
            )
        except Exception as te:
            logger.warning(
                "Could not run telemetry scoring for session %s: %s",
                session_id,
                te,
            )
        
        # If session is finished and has conversation history, trigger feedback generation
        # #region agent log
        logger.info(f"interview.py:end_interview:feedback_check. \n Checking if feedback should be generated \n session_id: {session_id} \n session_finished: {session_finished} \n has_history: {'history' in session} \n history_length: {len(session.get('history', ''))}")
        # #endregion
        
        task = None
        # Queue feedback when there is history (unified pipeline via feedback_item_id, else legacy per-type).
        history = session.get("history", "")
        if history:
            try:
                from tasks.feedback_tasks import generate_feedback
                task = generate_feedback.apply_async(
                    args=[session_id, history, user_info["email"]],
                    queue="feedback"
                )
                if task and task.id:
                    redis_client.setex(f"feedback_task:{task.id}", 3600, session_id)
                logger.info(f"Feedback generation queued for session {session_id}")
            except Exception as e:
                logger.warning(f"Failed to queue feedback generation for session {session_id}: {e}")
        else:
            logger.info(f"No history for feedback session_id: {session_id}")
        
        logger.info(f"Interview session {session_id} ended successfully")
        
        return {
            "task_id": task.id if task else None,
            "status": "ended",
            "session_id": session_id,
            "message": "Interview ended successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error ending interview: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to end interview: {str(e)}"
        )

def _get_feedback_status_data_sync(task_id: str, redis_client: Redis) -> Dict[str, Any]:
    """Blocking: AsyncResult + Redis for feedback task. Run in thread."""
    task_result = AsyncResult(task_id, app=celery_app)
    state_mapping = {
        "PENDING": "queued",
        "STARTED": "processing",
        "RETRY": "processing",
        "SUCCESS": "completed",
        "FAILURE": "failed",
    }
    status_str = state_mapping.get(task_result.state, "processing")
    progress = 0
    if task_result.state == "PROGRESS":
        meta = task_result.info or {}
        progress = meta.get("progress", 0)
        status_str = "processing"
    elif task_result.state == "SUCCESS":
        progress = 100
    result = None
    error = None
    session_id = redis_client.get(f"feedback_task:{task_id}") or ""
    if task_result.state == "SUCCESS":
        task_data = task_result.result
        if isinstance(task_data, dict):
            if task_data.get("status") == "completed":
                result = task_data.get("feedback")
            elif task_data.get("status") == "error":
                status_str = "failed"
                error = task_data.get("error")
    elif task_result.state == "FAILURE":
        error = str(task_result.info)
    if result is not None and not isinstance(result, dict):
        result = None
    return {
        "task_id": task_id,
        "session_id": session_id or "",
        "status": status_str,
        "progress": progress,
        "result": result,
        "error": error,
    }


@router.get("/feedback-status/{task_id}", response_model=FeedbackStatusResponse)
async def get_interview_feedback_status(
    task_id: str,
    user_info: Dict = Depends(get_current_user),
    interview_access: InterviewAccessTokenPayload = Depends(get_interview_access_payload),
    redis_client: Redis = Depends(get_redis),
):
    """
    Get status of feedback generation task (queued after end_meeting).

    Poll this with the feedback_task_id returned from POST /end when feedback was queued.
    """
    try:
        data = await _run_sync(_get_feedback_status_data_sync, task_id, redis_client)
        return FeedbackStatusResponse(**data)
    except Exception as e:
        logger.error(f"Error getting interview feedback status: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get status: {str(e)}",
        )


@router.delete("/{session_id}")
async def delete_session(
    session_id: str,
    user_info: Dict = Depends(get_current_user),
    interview_access: InterviewAccessTokenPayload = Depends(get_interview_access_payload),
    redis_client: Redis = Depends(get_redis),
):
    """Delete an interview session"""
    try:
        session_manager = InterviewSessionManager(redis_client)
        
        # Verify ownership
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
        
        if session["user_id"] != _user_identifier(user_info):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)
        
        # Delete session
        session_manager.delete_session(session_id)
        
        return {"status": "deleted", "session_id": session_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
