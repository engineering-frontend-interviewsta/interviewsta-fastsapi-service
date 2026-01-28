"""
Interview API endpoints
"""
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from typing import Dict, Any
import logging
import asyncio
import json

from schemas.interview import (
    InterviewStartRequest,
    InterviewStartResponse,
    UserResponseRequest,
    UserResponseSubmitResponse,
    InterviewStatusResponse,
    VideoQualityData
)
from api.dependencies import get_current_user, get_redis, verify_token_from_query
from services.interview_session import InterviewSessionManager
from tasks.interview_tasks import process_interview_start, process_user_response
from tasks.audio_tasks import transcribe_audio, synthesize_speech
from redis import Redis

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/start", response_model=InterviewStartResponse)
async def start_interview(
    request: InterviewStartRequest,
    user_info: Dict = Depends(get_current_user),
    redis_client: Redis = Depends(get_redis)
):
    """
    Start a new interview session
    
    This endpoint:
    1. Creates a new session in Redis
    2. Queues a Celery task to initialize the workflow and generate greeting
    3. Returns task_id for status polling
    """
    try:
        logger.info(f"Starting {request.interview_type} interview for user {user_info['uid']}")
        
        # Validate user matches
        if request.user_id != user_info["uid"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User ID mismatch"
            )
        
        # Queue Celery task
        task = process_interview_start.apply_async(
            args=[request.session_id, request.interview_type, request.user_id, request.payload],
            queue="interview"
        )
        
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


@router.post("/{session_id}/respond", response_model=UserResponseSubmitResponse)
async def submit_response(
    session_id: str,
    request: UserResponseRequest,
    user_info: Dict = Depends(get_current_user),
    redis_client: Redis = Depends(get_redis)
):
    """
    Submit user's response to interview question
    
    This endpoint:
    1. Transcribes audio if provided (async task)
    2. Processes user response through workflow
    3. Generates AI response
    4. Returns task_id for status polling
    """
    try:
        logger.info(f"Submitting response for session {session_id}")
        
        # Get session and verify ownership
        session_manager = InterviewSessionManager(redis_client)
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
        
        # Process audio or use text response
        human_input = request.text_response
        
        if request.audio_data and not human_input:
            # Transcribe audio first
            transcribe_task = transcribe_audio.apply_async(
                args=[request.audio_data],
                queue="audio"
            )
            
            # Wait for transcription (or could make this async and return immediately)
            transcribe_result = transcribe_task.get(timeout=30)
            
            if transcribe_result["status"] == "error":
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Transcription failed: {transcribe_result['error']}"
                )
            
            human_input = transcribe_result["transcription"]
            session_manager.set_transcript(session_id, human_input)
        
        if not human_input:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either audio_data or text_response must be provided"
            )
        
        # Add code input if provided
        if request.code_input:
            human_input += f"\n\n[CODE INPUT]\n{request.code_input}"
        
        # Queue workflow processing task
        task = process_user_response.apply_async(
            args=[session_id, human_input],
            queue="interview"
        )
        
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


@router.get("/{session_id}/status", response_model=InterviewStatusResponse)
async def get_interview_status(
    session_id: str,
    user_info: Dict = Depends(get_current_user),
    redis_client: Redis = Depends(get_redis)
):
    """
    Get current status of interview session
    
    Returns:
    - Current status (waiting_for_response, processing, ai_responded, completed)
    - AI message and audio if ready
    - Last workflow node
    - Latest transcription
    """
    try:
        session_manager = InterviewSessionManager(redis_client)
        
        # Wait for session to be created (short timeout for status checks)
        session = None
        max_wait = 12  # Wait up to 12 seconds (increased from 5s to handle slower task completion)
        wait_interval = 0.3  # Check every 300ms
        waited = 0
        
        while waited < max_wait:
            session = session_manager.get_session(session_id)
            if session:
                break
            await asyncio.sleep(wait_interval)
            waited += wait_interval
        
        # Verify session exists
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found or not yet created"
            )
        
        # Verify ownership
        if session["user_id"] != user_info["uid"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to access this session"
            )
        
        # Get status
        current_status = session_manager.get_status(session_id) or "waiting_for_response"
        
        # Get response if AI has responded
        response_data = None
        if current_status == "ai_responded":
            response_data = session_manager.get_response(session_id)
        
        # Get latest transcript
        transcript = session_manager.get_transcript(session_id)
        
        return InterviewStatusResponse(
            session_id=session_id,
            status=current_status,
            message=response_data.get("message") if response_data else None,
            audio=response_data.get("audio") if response_data else None,
            last_node=response_data.get("last_node") if response_data else session.get("last_node"),
            transcript=transcript,
            created_at=session.get("created_at"),
            updated_at=session.get("updated_at")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting interview status: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get status: {str(e)}"
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
    """
    try:
        session_manager = InterviewSessionManager(redis_client)
        
        # Verify session ownership
        session = session_manager.get_session(session_id)
        if not session or session["user_id"] != user_info["uid"]:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)
        
        # Store metrics (simplified - in production might aggregate)
        metrics_key = f"session:{session_id}:video_metrics"
        redis_client.lpush(metrics_key, str(data.dict()))
        redis_client.expire(metrics_key, 3600)
        
        # In production, you might trigger warnings based on thresholds
        # For now, just acknowledge receipt
        
        return {
            "status": "accepted",
            "message": "Video quality data recorded"
        }
        
    except Exception as e:
        logger.error(f"Error submitting video quality: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


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
                            
                            yield f"event: ai_response\ndata: {json.dumps(response_data)}\n\n"
                            
                            # Reset status to waiting after sending response
                            session_manager.set_status(session_id, "waiting_for_response")
                    
                    # Check for new transcription
                    transcript = session_manager.get_transcript(session_id)
                    if transcript:
                        yield f"event: transcription\ndata: {json.dumps({'text': transcript})}\n\n"
                        # Clear transcript after sending
                        session_manager.set_transcript(session_id, "")
                    
                    await asyncio.sleep(1)  # Poll every second
                    
                except Exception as e:
                    logger.error(f"Error in SSE stream: {e}")
                    yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
                    break
                    
        except Exception as e:
            logger.error(f"Fatal error in SSE generator: {e}")
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
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
