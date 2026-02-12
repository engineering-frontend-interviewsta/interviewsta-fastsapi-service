"""
Interview API endpoints

All blocking I/O (Redis, Celery AsyncResult) is run off the event loop via run_sync()
to keep the event loop free and avoid 502s / health-check timeouts under load.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from typing import Any, Dict, Optional
import logging
import asyncio
import json
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
    VideoTelemetryData,
    # VideoTelemetryPayload
)
from api.dependencies import get_current_user, get_redis, verify_token_from_query
from services.interview_session import InterviewSessionManager
from celery.result import AsyncResult
from tasks.interview_tasks import process_interview_start
from tasks.audio_tasks import transcribe_audio, synthesize_speech
from redis import Redis
from tasks.celery_app import celery_app
from schemas.feedback import FeedbackStatusResponse


logger = logging.getLogger(__name__)

router = APIRouter()


async def _run_sync(sync_fn, *args, **kwargs):
    """Run blocking sync work in thread pool to keep event loop free"""
    return await asyncio.to_thread(sync_fn, *args, **kwargs)


def _get_start_status_data_sync(
    task_id: str,
    redis_client: Redis,
    user_uid: str
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
            
            if session and session.get("user_id") == user_uid:
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
                        "question_number": None,
                        "total_questions": None,
                    }
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
    user_uid: str
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
        
        if session.get("user_id") != user_uid:
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
                        "question_number": None,
                        "total_questions": None,
                    }
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
    redis_client: Redis = Depends(get_redis)
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
        logger.info(f"Starting {request.interview_type} interview for user {user_info['uid']}")
        
        # Optional: Validate user matches (if needed)
        # if request.user_id != user_info["uid"]:
        #     raise HTTPException(status_code=403, detail="User ID mismatch")
        
        # Queue Celery task (non-blocking)
        task = process_interview_start.apply_async(
            args=[
                request.session_id,
                request.interview_type,
                request.user_id,
                request.payload
            ],
            queue="interview"
        )
        
        logger.info(f"Interview task queued: {task.id}")
        
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
            user_info["uid"],
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
    redis_client: Redis = Depends(get_redis)
):
    """
    Submit user response to interview question
    Optimized: Returns immediately after queuing, ~0.3s response time
    
    Flow:
    1. Validate session exists (fast Redis check)
    2. Check processing flag
    3. Queue complete pipeline task (transcribe + process + audio)
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
        
        # Validate input
        if not request.audio_data and not request.text_response:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either audio_data or text_response must be provided"
            )
        
        # Queue complete pipeline task (non-blocking)
        task = process_user_response_with_transcription.apply_async(
            args=[
                session_id,
                request.audio_data,
                request.text_response,
                request.code_input
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
            user_info["uid"],
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
            error_code = err.get("code", 500)
            error_detail = err.get("detail", "Unknown error")
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
    redis_client: Redis = Depends(get_redis)
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
        if not session or session["user_id"] != user_info["uid"]:
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
async def set_video_telemetry(
    session_id: str,
    payload: VideoTelemetryData,
    user_info: Dict = Depends(get_current_user),
    redis_client: Redis = Depends(get_redis),
):
    """
    Set video telemetry for a session. Accepts payload:
    { "type": "video_quality", "data": { face, gaze, confidence, nervousness, engagement, distraction, big5_features } }.
    Stores soft skills in video_metrics (for aggregation) and big5_features in big5_profile:{session_id}.
    """
    try:
        session_manager = InterviewSessionManager(redis_client)
        session = session_manager.get_session(session_id)
        # if not session or session["user_id"] != user_info["uid"]:
        #     raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to access this session")

        data = payload
        avg_key = f"session:{session_id}:video_metrics"
        count_key = f"session:{session_id}:video_telemetry_count"

        # big5_key = f"big5_profile:{session_id}"
        ttl = 3600

        count = int(redis_client.get(count_key) or 0)
        new_count = count + 1

        # Store soft-skills sample for aggregation (same format as existing video-quality)
        if count == 0:
            avg_data = {
                "face": data.face or "ok",
                "gaze": data.gaze,
                "confidence": data.confidence,
                "nervousness": data.nervousness,
                "engagement": data.engagement,
                "distraction": data.distraction,
            }
            redis_client.setex(avg_key, ttl, json.dumps(avg_data))
            redis_client.setex(count_key, ttl, str(new_count))
        else:
            existing = json.loads(redis_client.get(avg_key) or "{}")
            def run_avg(old, new, c):
                old = old if old is not None else 0
                new = new if new is not None else 0
                return (old * c + new) / (c + 1)
            avg_data = {
                "face": data.face or existing.get("face", "ok"),
                "gaze": run_avg(existing.get("gaze"), data.gaze, count),
                "confidence": run_avg(existing.get("confidence"), data.confidence, count),
                "nervousness": run_avg(existing.get("nervousness"), data.nervousness, count),
                "engagement": run_avg(existing.get("engagement"), data.engagement, count),
                "distraction": run_avg(existing.get("distraction"), data.distraction, count),
            }
            redis_client.setex(avg_key, ttl, json.dumps(avg_data))
            redis_client.setex(count_key, ttl, str(new_count))

       

        return {"status": "accepted", "message": "Video telemetry data recorded"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting video telemetry for session {session_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )

@router.get("/{session_id}/video-telemetry")
async def get_video_telemetry(
    session_id: str,
    user_info: Dict = Depends(get_current_user),
    redis_client: Redis = Depends(get_redis),
):
    """Get stored video telemetry (running average) for the session. Same shape as set payload."""
    try:
        session_manager = InterviewSessionManager(redis_client)
        session = session_manager.get_session(session_id)
        if not session or session["user_id"] != user_info["uid"]:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to access this session")

        avg_key = f"session:{session_id}:video_telemetry_avg"
        # big5_key = f"big5_profile:{session_id}"
        count_key = f"session:{session_id}:video_telemetry_count"

        avg_json = redis_client.get(avg_key)
        if not avg_json:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No video telemetry for this session")

        data = json.loads(avg_json)
        count = int(redis_client.get(count_key) or 0)
        # big5_json = redis_client.get(big5_key)
        # big5_features = json.loads(big5_json) if big5_json else None
        # data["big5_features"] = big5_features
        return {"type": "video_quality", "data": data, "count": count}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting video telemetry for session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/{session_id}/stream")
async def stream_interview_status(
    session_id: str,
    token: str,  # Token as query parameter (EventSource doesn't support headers)
    redis_client: Redis = Depends(get_redis)
):
    """
    Server-Sent Events stream for real-time interview updates
    
    Note: Token must be passed as query parameter (?token=...) because
    EventSource doesn't support custom headers
    
    Events:
    - transcription: User's transcribed speech
    - ai_response: AI's response with audio
    - status: Status changes
    - complete: Interview completed
    """
    # Verify token from query parameter
    try:
        user_info = await verify_token_from_query(token)
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
            
            # Wait for session to be created (with timeout)
            session = None
            max_wait = 20  # Wait up to 20 seconds (tasks can take 8-15s to complete)
            wait_interval = 0.5  # Check every 500ms
            waited = 0
            
            while waited < max_wait:
                session = session_manager.get_session(session_id)
                if session:
                    logger.info(f"SSE: Session {session_id} found after {waited}s")
                    break
                await asyncio.sleep(wait_interval)
                waited += wait_interval
            
            # Verify session exists
            if not session:
                logger.warning(f"SSE: Session {session_id} not found after {max_wait}s timeout")
                yield f"event: error\ndata: {json.dumps({'error': 'Session not found or timed out'})}\n\n"
                return
            
            # Verify ownership
            if session["user_id"] != user_info["uid"]:
                logger.warning(f"SSE: Unauthorized access attempt to session {session_id}")
                yield f"event: error\ndata: {json.dumps({'error': 'Unauthorized'})}\n\n"
                return
            
            logger.info(f"SSE: Stream established for session {session_id}")
            
            last_status = None
            last_response_time = None
            
            # Poll for updates every second
            while True:
                try:
                    current_status = session_manager.get_status(session_id)
                    
                    # Status changed
                    if current_status != last_status:
                        last_status = current_status
                        yield f"event: status\ndata: {json.dumps({'status': current_status})}\n\n"
                        
                        # If completed, send final event and close
                        if current_status == "completed":
                            yield f"event: complete\ndata: {json.dumps({'status': 'completed'})}\n\n"
                            break
                    
                    # Check for new AI response
                    if current_status == "ai_responded":
                        response_data = session_manager.get_response(session_id)
                        if response_data and response_data.get("timestamp") != last_response_time:
                            last_response_time = response_data.get("timestamp")
                            
                            # Format response with both audio and audio_base64 for compatibility
                            formatted_response = {
                                "message": response_data.get("message"),
                                "audio": response_data.get("audio"),
                                "audio_base64": response_data.get("audio"),  # Add audio_base64 field
                                "last_node": response_data.get("last_node"),
                                "timestamp": response_data.get("timestamp"),
                                "question_number": None,
                                "total_questions": None
                            }
                            
                            yield f"event: ai_response\ndata: {json.dumps(formatted_response)}\n\n"
                            
                            # Reset status to waiting after sending response
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
    redis_client: Redis = Depends(get_redis)
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
        
        if session["user_id"] != user_info["uid"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to access this session"
            )
        
        # Mark session as ended
        session_manager.set_status(session_id, "completed")
        
        # Store session metadata
        session_manager.update_session(session_id, {
            "duration": int(duration) if isinstance(duration, (int, str)) else 0,
            "interview_type": interview_type,
            "interview_test_id": interview_test_id,
            "session_finished": session_finished,
            "ended_at": datetime.utcnow().isoformat()
        })
        
        # If session is finished and has conversation history, trigger feedback generation
        # #region agent log
        logger.info(f"interview.py:end_interview:feedback_check. \n Checking if feedback should be generated \n session_id: {session_id} \n session_finished: {session_finished} \n has_history: {'history' in session} \n history_length: {len(session.get('history', ''))}")
        # #endregion
        
        if session_finished:
            history = session.get("history", "")
            # interaction_log = session.get("messages", "")
            # #region agent log
            logger.info(f"interview.py:end_interview:session_finished. \n Session marked as finished \n session_id: {session_id} \n has_history: {bool(history)} \n history_length: {len(history)} \n history: {history[:20]}")
            # #endregion
            
            task = None
            if history:
                try:
                    from tasks.feedback_tasks import (
                        generate_technical_feedback,
                        generate_hr_feedback,
                        generate_case_study_feedback
                    )
                    
                    # Queue appropriate feedback task
                    # Map "Coding" to "Technical" for feedback generation
                    feedback_type = "Technical" if interview_type in ["Technical", "Coding"] else interview_type
                    
                    logger.info(f"interview.py:end_interview:feedback_type. \n Feedback type: {feedback_type} \n interview_type: {interview_type}")

                    if feedback_type == "Technical":
                        task = generate_technical_feedback.apply_async(
                            args=[session_id, history, user_info["email"]],
                            queue="feedback"
                        )
                    elif feedback_type == "HR Interview":
                        task = generate_hr_feedback.apply_async(
                            args=[session_id, history, user_info["email"]],
                            queue="feedback"
                        )
                    elif feedback_type == "Case Study Interview":
                        task = generate_case_study_feedback.apply_async(
                            args=[session_id, history, user_info["email"]],
                            queue="feedback"
                        )
                    
                    # #region agent log
                    logger.info(f"interview.py:end_interview:feedback_queued. \n Feedback task queued successfully \n session_id: {session_id} \n feedback_type: {feedback_type} \n interview_type: {interview_type}")
                    # #endregion
                    
                    logger.info(f"Feedback generation queued for session {session_id}")
                except Exception as e:
                    # #region agent log
                    logger.info(f"interview.py:end_interview:feedback_error. \n Failed to queue feedback \n session_id: {session_id} \n error: {str(e)}")
                    # #endregion
                    logger.warning(f"Failed to queue feedback generation for session {session_id}: {e}")
            else:
                # #region agent log
                logger.info(f"interview.py:end_interview:no_history. \n No history available for feedback generation \n session_id: {session_id}")
                # #endregion
        
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
    redis_client: Redis = Depends(get_redis)
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
    redis_client: Redis = Depends(get_redis)
):
    """Delete an interview session"""
    try:
        session_manager = InterviewSessionManager(redis_client)
        
        # Verify ownership
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
        
        if session["user_id"] != user_info["uid"]:
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
