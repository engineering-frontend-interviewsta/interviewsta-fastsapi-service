"""
Resume analysis API endpoints
"""
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from typing import Optional
import asyncio
import json
import logging
import base64
import time

from schemas.resume import (
    ResumeAnalysisResponse,
    ResumeAnalysisStatusResponse
)
from api.dependencies import get_current_user, get_redis
from api.resume_task_status import resolve_resume_task_status
from tasks.resume_tasks import process_resume_upload
from redis import Redis

logger = logging.getLogger(__name__)

SSE_POLL_INTERVAL_S = 0.75
SSE_HEARTBEAT_INTERVAL_S = 20.0
SSE_MAX_DURATION_S = 125.0

router = APIRouter()


@router.post("/analyze", response_model=ResumeAnalysisResponse)
async def analyze_resume(
    resume: UploadFile = File(...),
    job_description: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    user_info: dict = Depends(get_current_user),
    redis_client: Redis = Depends(get_redis)
):
    """
    Submit resume and job description for analysis
    
    Args:
        resume: Resume file (PDF or image)
        job_description: Job description file (PDF or text)
        
    Returns:
        Task ID for status polling
    """
    try:
        logger.info(f"Analyzing resume for user {user_info['uid']}")
        
        # Validate file types
        allowed_extensions = [".pdf", ".png", ".jpg", ".jpeg", ".webp", ".txt"]
        
        resume_ext = resume.filename.split(".")[-1].lower()
        job_desc_ext = job_description.filename.split(".")[-1].lower()
        
        if f".{resume_ext}" not in allowed_extensions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Resume file type not supported: {resume_ext}"
            )
        
        if f".{job_desc_ext}" not in allowed_extensions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Job description file type not supported: {job_desc_ext}"
            )
        
        # Read file bytes
        resume_bytes = await resume.read()
        job_desc_bytes = await job_description.read()
        
        # Validate file sizes (max 10MB each)
        max_size = 10 * 1024 * 1024  # 10MB
        
        if len(resume_bytes) > max_size:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Resume file too large (max 10MB)"
            )
        
        if len(job_desc_bytes) > max_size:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Job description file too large (max 10MB)"
            )
        
        # Encode to base64 for Celery serialization
        resume_b64 = base64.b64encode(resume_bytes).decode("utf-8")
        job_desc_b64 = base64.b64encode(job_desc_bytes).decode("utf-8")
        
        # Use caller-provided session_id if present, otherwise generate one
        import uuid
        session_id = session_id or str(uuid.uuid4())
        
        # Queue analysis task
        task = process_resume_upload.apply_async(
            args=[
                None,  # task_id will be generated
                resume_b64,
                resume.filename,
                job_desc_b64,
                job_description.filename,
                user_info["email"],
                session_id  # Pass session_id
            ],
            queue="resume"
        )
        
        logger.info(f"Resume analysis queued: {task.id}")
        
        return ResumeAnalysisResponse(
            task_id=task.id,
            status="queued",
            message="Resume analysis queued"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting resume analysis: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit resume: {str(e)}"
        )


@router.get("/{task_id}/status", response_model=ResumeAnalysisStatusResponse)
async def get_analysis_status(
    task_id: str,
    user_info: dict = Depends(get_current_user)
):
    """
    Get status of resume analysis task
    
    Args:
        task_id: Celery task ID
        
    Returns:
        Task status and results if completed
    """
    try:
        status_str, progress, result, error = resolve_resume_task_status(task_id)

        logger.info(
            "Resume status: task_id=%s status_str=%s result_type=%s result_is_dict=%s",
            task_id,
            status_str,
            type(result).__name__ if result is not None else "None",
            isinstance(result, dict),
        )

        return ResumeAnalysisStatusResponse(
            task_id=task_id,
            status=status_str,
            progress=progress,
            result=result,
            error=error
        )

    except Exception as e:
        logger.error(f"Error getting analysis status: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get status: {str(e)}"
        )


async def _resume_analysis_sse_events(task_id: str):
    """Yield SSE lines until completed, failed, or timeout (single HTTP connection)."""
    deadline = time.monotonic() + SSE_MAX_DURATION_S
    last_heartbeat = time.monotonic()
    last_progress_sent = -1

    while time.monotonic() < deadline:
        status_str, progress, result, error = resolve_resume_task_status(task_id)

        if status_str == "completed" and result is not None:
            payload = json.dumps(result, ensure_ascii=False)
            yield f"event: complete\ndata: {payload}\n\n"
            return

        if status_str == "failed" or error:
            err_msg = error or "Resume analysis failed"
            yield f"event: error\ndata: {json.dumps({'error': err_msg})}\n\n"
            return

        if status_str == "completed" and result is None:
            yield (
                "event: error\ndata: "
                + json.dumps({"error": "Analysis completed but returned no result"})
                + "\n\n"
            )
            return

        if progress != last_progress_sent:
            last_progress_sent = progress
            yield (
                "event: progress\ndata: "
                + json.dumps({"progress": progress, "status": status_str})
                + "\n\n"
            )

        now = time.monotonic()
        if now - last_heartbeat >= SSE_HEARTBEAT_INTERVAL_S:
            yield ": ping\n\n"
            last_heartbeat = now

        await asyncio.sleep(SSE_POLL_INTERVAL_S)

    yield (
        "event: error\ndata: "
        + json.dumps({"error": "Resume analysis timed out"})
        + "\n\n"
    )


@router.get("/{task_id}/stream")
async def stream_resume_analysis(
    task_id: str,
    user_info: dict = Depends(get_current_user),
):
    """
    Server-Sent Events stream until the Celery task completes or fails.
    Nest (or clients with fetch streaming) should consume this instead of polling /status.
    """
    _ = user_info
    return StreamingResponse(
        _resume_analysis_sse_events(task_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
