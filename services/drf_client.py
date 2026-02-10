"""
Client for persisting resume analysis and interview feedback via DRF internal API.
Replaces direct Django ORM usage (django_db.py).
"""
import logging
import os
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

# Base URL and auth - read at call time so env is available in Celery workers
def _drf_base_url() -> str:
    return os.getenv("DRF_BASE_URL", "http://localhost:8000").rstrip("/")

def _drf_headers() -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    api_key = os.getenv("DRF_INTERNAL_API_KEY", "")
    if api_key:
        headers["X-Internal-API-Key"] = api_key
    return headers

def _post(path: str, payload: Dict[str, Any], timeout: int = 30) -> Dict[str, Any]:
    url = f"{_drf_base_url()}{path}"
    try:
        r = requests.post(url, json=payload, headers=_drf_headers(), timeout=timeout)
        if r.ok:
            return r.json()
        logger.warning(f"DRF API error: {path} status={r.status_code} body={r.text[:500]}")
        return False
    except requests.RequestException as e:
        logger.error(f"DRF API request failed: {path} error={e}", exc_info=True)
        return False


def save_resume_analysis_to_db(
    user_email: str,
    session_id: str,
    analysis_result: Dict[str, Any],
) -> bool:
    """
    Save resume analysis via DRF internal API.
    Same contract as former django_db.save_resume_analysis_to_db.
    """
    payload = {
        "user_email": user_email,
        "session_id": session_id,
        "resume_name": analysis_result.get("resume_name", "Your_Resume.pdf"),
        "company": analysis_result.get("company", ""),
        "role": analysis_result.get("role", ""),
        "job_match_score": analysis_result.get("section_analysis", {}).job_match_score,
        "format_and_structure": analysis_result.get("section_analysis", {}).format_and_structure,
        "content_quality": analysis_result.get("section_analysis", {}).content_quality,
        "length_and_conciseness": analysis_result.get("section_analysis", {}).length_and_conciseness,
        "keywords_optimization": analysis_result.get("section_analysis", {}).keywords_optimization,
        "found_keywords": analysis_result.get("keyword_analysis", {}).found_keywords,
        "not_found_keywords": analysis_result.get("keyword_analysis", {}).not_found_keywords,
        "top_3_keywords": analysis_result.get("keyword_analysis", {}).top_3_keywords,
        "required_skills": analysis_result.get("job_alignment_analysis", {}).required_skills,
        "preferred_skills": analysis_result.get("job_alignment_analysis", {}).preferred_skills,
        "experience": analysis_result.get("job_alignment_analysis", {}).experience,
        "education": analysis_result.get("job_alignment_analysis", {}).education,
        "insights": analysis_result.get("job_alignment_analysis", {}).insights,
        "candidate_strengths": analysis_result.get("strengths_and_improvements", {}).candidate_strengths,
        "candidates_areas_of_improvements": analysis_result.get("strengths_and_improvements", {}).candidates_areas_of_improvements,
    }
    return _post("/api/internal/resume-analysis/", payload)


def save_feedback_to_db(
    user_email: str,
    session_id: str,
    interview_type: str,
    interview_test_id: Optional[int],
    duration_seconds: int,
    feedback_data: Dict[str, Any],
    interaction_log: List[Any],
    soft_skill_summary: Optional[Dict[str, Any]] = None,
    big5_profile: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Save interview feedback via DRF internal API.
    Same contract as former django_db.save_feedback_to_db.
    """
    # Map to DRF-friendly names (e.g. interaction_status_log)
    payload = {
        "user_email": user_email,
        "session_id": session_id,
        "interview_type": interview_type,
        "interview_test_id": interview_test_id,
        "duration_seconds": duration_seconds,
        "feedback_data": feedback_data,
        "interaction_log": interaction_log,
        # "interaction_log_feedback": feedback_data.get("interaction_log_feedback"),
        "soft_skill_summary": soft_skill_summary or {},
        "big5_profile": big5_profile or {},
    }
    return _post("/api/internal/feedback-analysis/", payload)