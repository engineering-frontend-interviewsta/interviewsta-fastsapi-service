"""
Shared Celery AsyncResult → status payload for resume analysis (HTTP status + SSE).
"""
import logging
from typing import Any, Dict, Optional, Tuple

from celery.result import AsyncResult

from tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


def resolve_resume_task_status(task_id: str) -> Tuple[str, int, Optional[Dict[str, Any]], Optional[str]]:
    """
    Returns (status_str, progress, result, error) aligned with ResumeAnalysisStatusResponse.
    status_str: queued | processing | completed | failed
    """
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

    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    if task_result.state == "SUCCESS":
        task_data = task_result.result
        if task_data and task_data.get("status") == "completed":
            result = task_data.get("result")
        elif task_data and task_data.get("status") == "error":
            status_str = "failed"
            error = task_data.get("error")
    elif task_result.state == "FAILURE":
        error = str(task_result.info)

    if result is not None and not isinstance(result, dict):
        logger.warning(
            "Resume task %s: result is not a dict (value=%r), normalizing to None",
            task_id,
            result,
        )
        result = None

    return status_str, progress, result, error
