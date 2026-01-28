"""
Resume analysis API endpoints
"""
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from typing import Optional
import logging
import base64

from schemas.resume import (
    ResumeAnalysisResponse,
    ResumeAnalysisStatusResponse
)
from api.dependencies import get_current_user, get_redis
from tasks.resume_tasks import process_resume_upload
from redis import Redis
from celery.result import AsyncResult

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/analyze", response_model=ResumeAnalysisResponse)
async def analyze_resume(
    resume: UploadFile = File(...),
    job_description: UploadFile = File(...),
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
        
        # Queue analysis task
        task = process_resume_upload.apply_async(
            args=[
                None,  # task_id will be generated
                resume_b64,
                resume.filename,
                job_desc_b64,
                job_description.filename,
                user_info["uid"]
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
        # Get task result
        task_result = AsyncResult(task_id)
        
        # Map Celery states to our states
        state_mapping = {
            "PENDING": "queued",
            "STARTED": "processing",
            "RETRY": "processing",
            "SUCCESS": "completed",
            "FAILURE": "failed"
        }
        
        status_str = state_mapping.get(task_result.state, "processing")
        
        # Get progress if available
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
        
        if task_result.state == "SUCCESS":
            task_data = task_result.result
            if task_data and task_data.get("status") == "completed":
                result = task_data.get("result")
            elif task_data and task_data.get("status") == "error":
                status_str = "failed"
                error = task_data.get("error")
        elif task_result.state == "FAILURE":
            error = str(task_result.info)
        
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
