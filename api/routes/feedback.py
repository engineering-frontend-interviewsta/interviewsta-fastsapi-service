"""
Feedback generation API endpoints
"""
from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, Any
import logging

from schemas.feedback import (
    FeedbackGenerationRequest,
    FeedbackGenerationResponse,
    FeedbackStatusResponse
)
from api.dependencies import get_current_user, get_redis
from services.interview_session import InterviewSessionManager
from tasks.feedback_tasks import (
    generate_technical_feedback,
    generate_hr_feedback,
    generate_case_study_feedback
)
from redis import Redis
from celery.result import AsyncResult

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/generate", response_model=FeedbackGenerationResponse)
async def request_feedback_generation(
    request: FeedbackGenerationRequest,
    user_info: Dict = Depends(get_current_user),
    redis_client: Redis = Depends(get_redis)
):
    """
    Request feedback generation for completed interview
    
    Args:
        request: Feedback generation request with session_id and interview_type
        
    Returns:
        Task ID for status polling
    """
    try:
        logger.info(f"Generating feedback for session {request.session_id}")
        
        # Verify user owns the session
        session_manager = InterviewSessionManager(redis_client)
        session = session_manager.get_session(request.session_id)
        
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
        
        # Get conversation history
        history = session.get("history", "")
        
        if not history:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No conversation history found for this session"
            )
        
        # Queue appropriate feedback task
        task = None
        
        if request.interview_type == "Technical":
            task = generate_technical_feedback.apply_async(
                args=[request.session_id, history, user_info["uid"]],
                queue="feedback"
            )
        elif request.interview_type == "HR":
            task = generate_hr_feedback.apply_async(
                args=[request.session_id, history, user_info["uid"]],
                queue="feedback"
            )
        elif request.interview_type == "CaseStudy":
            task = generate_case_study_feedback.apply_async(
                args=[request.session_id, history, user_info["uid"]],
                queue="feedback"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported interview type: {request.interview_type}"
            )
        
        logger.info(f"Feedback generation queued: {task.id}")
        
        return FeedbackGenerationResponse(
            task_id=task.id,
            session_id=request.session_id,
            status="queued"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error requesting feedback generation: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to request feedback: {str(e)}"
        )


@router.get("/{task_id}/status", response_model=FeedbackStatusResponse)
async def get_feedback_status(
    task_id: str,
    user_info: Dict = Depends(get_current_user)
):
    """
    Get status of feedback generation task
    
    Args:
        task_id: Celery task ID
        
    Returns:
        Task status and feedback if completed
    """
    try:
        # Get task result
        task_result = AsyncResult(task_id)
        
        # Map Celery states
        state_mapping = {
            "PENDING": "queued",
            "STARTED": "processing",
            "RETRY": "processing",
            "SUCCESS": "completed",
            "FAILURE": "failed"
        }
        
        status_str = state_mapping.get(task_result.state, "processing")
        
        # Get progress
        progress = 0
        if task_result.state == "PROGRESS":
            meta = task_result.info or {}
            progress = meta.get("progress", 0)
            status_str = "processing"
        elif task_result.state == "SUCCESS":
            progress = 100
        
        # Get result or error
        result = None
        error = None
        session_id = None
        
        if task_result.state == "SUCCESS":
            task_data = task_result.result
            if task_data and task_data.get("status") == "completed":
                result = task_data.get("feedback")
            elif task_data and task_data.get("status") == "error":
                status_str = "failed"
                error = task_data.get("error")
        elif task_result.state == "FAILURE":
            error = str(task_result.info)
        
        return FeedbackStatusResponse(
            task_id=task_id,
            session_id=session_id or "",
            status=status_str,
            progress=progress,
            result=result,
            error=error
        )
        
    except Exception as e:
        logger.error(f"Error getting feedback status: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get status: {str(e)}"
        )


@router.get("/session/{session_id}", response_model=Dict[str, Any])
async def get_session_feedback(
    session_id: str,
    user_info: Dict = Depends(get_current_user),
    redis_client: Redis = Depends(get_redis)
):
    """
    Get feedback for a session (if already generated)
    
    Args:
        session_id: Interview session ID
        
    Returns:
        Feedback data if available
    """
    try:
        # Verify session ownership
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
        
        # Get feedback from Redis
        feedback_key = f"feedback:{session_id}"
        feedback_data = redis_client.get(feedback_key)
        
        if not feedback_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Feedback not yet generated for this session"
            )
        
        return {
            "session_id": session_id,
            "feedback": eval(feedback_data),  # Convert string back to dict
            "status": "available"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session feedback: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get feedback: {str(e)}"
        )
