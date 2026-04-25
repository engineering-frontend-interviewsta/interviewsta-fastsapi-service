"""
Client for persisting resume analysis and interview feedback via DRF internal API.
Replaces direct Django ORM usage (django_db.py).
"""
import json
import logging
import os
from typing import Any, Dict, List, Optional, Union

import requests

YELLOW = "\033[33m"
RESET = "\033[0m"

logger = logging.getLogger(__name__)

# Base URL and auth - read at call time so env is available in Celery workers
def _drf_base_url() -> str:
    return os.getenv("DRF_BASE_URL", "http://localhost:3003").rstrip("/")

def _drf_headers() -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    api_key = os.getenv("DRF_INTERNAL_API_KEY", "")
    if api_key:
        headers["X-Internal-API-Key"] = api_key
    else:
        logger.warning(
            "DRF_INTERNAL_API_KEY is unset; Nest internal POSTs will get 401. "
            "Set it to the same value as INTERNAL_API_KEY in interviewsta-backend/.env"
        )
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


def _get(path: str, params: Optional[Dict[str, Any]] = None, timeout: int = 30) -> Any:
    """GET request to DRF internal API. Returns JSON on success, False on failure."""
    url = f"{_drf_base_url()}{path}"
    try:
        r = requests.get(url, params=params or {}, headers=_drf_headers(), timeout=timeout)
        if r.ok:
            return r.json()
        logger.warning(f"DRF API error: {path} status={r.status_code} body={r.text[:500]}")
        return None
    except requests.RequestException as e:
        logger.error(f"DRF API GET failed: {path} error={e}", exc_info=True)
        return None


def get_research_questions_for_subject(interview_test_id: Optional[Union[int, str]] = None) -> Optional[Any]:
    """
    Fetch relevant research questions for a Subject interview from DRF.
    Calls InternalQuestionResearchView: GET internal/question-research/?interview_type_id=...
    interview_test_id can be an integer or a UUID string (DRF may accept either).

    Returns:
        Response payload from DRF with keys interview_type_id, topic, research; or None on failure.
    """
    if interview_test_id is None:
        return None
    return _get("/internal/question/", params={"interview_type_id": interview_test_id})


def get_company_for_interview(interview_type_id: Optional[Union[int, str]] = None) -> Optional[Any]:
    """
    Fetch company name for a Company interview from DRF.
    interview_type_id can be an integer or a UUID string.

    Returns:
        Response payload from DRF with keys interview_type_id, company; or None on failure.
    """
    if interview_type_id is None:
        return None
    return _get("/api/internal/interview-company/", params={"interview_type_id": interview_type_id})


def get_interview_questions(interview_type_id: Optional[Union[int, str]] = None) -> Optional[Any]:
    """
    Fetch 5 interview questions (3 theoretical + 2 coding) for Company/Subject interviews from DRF.
    interview_type_id can be an integer or a UUID string.

    Returns:
        Response with interview_type_id, company, subjects, theoretical_questions, coding_questions; or None on failure.
    """
    if interview_type_id is None:
        return None
    return _get("/internal/interview-questions/", params={"interview_type_id": interview_type_id})


def build_resume_analysis_result_dict(
    session_id: str,
    resume_name: str,
    analysis_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build the full JSON-serializable result dict from LangGraph state (Pydantic nodes).
    Used for Redis and Nest polling even if the internal POST fails.
    """
    section = analysis_result.get("section_analysis")
    keyword = analysis_result.get("keyword_analysis")
    alignment = analysis_result.get("job_alignment_analysis")
    strengths = analysis_result.get("strengths_and_improvements")

    company = analysis_result.get("company", "") or ""
    role = analysis_result.get("role", "") or ""

    job_match_score = int(getattr(section, "job_match_score", 0) or 0)
    format_and_structure = int(getattr(section, "format_and_structure", 0) or 0)
    content_quality = int(getattr(section, "content_quality", 0) or 0)
    length_and_conciseness = int(getattr(section, "length_and_conciseness", 0) or 0)
    keywords_optimization = int(getattr(section, "keywords_optimization", 0) or 0)
    ats_score = int(getattr(section, "ats_score", 0) or 0)

    sections = [
        {"name": "job_match_score", "score": job_match_score},
        {"name": "format_and_structure", "score": format_and_structure},
        {"name": "content_quality", "score": content_quality},
        {"name": "length_and_conciseness", "score": length_and_conciseness},
        {"name": "keywords_optimization", "score": keywords_optimization},
        {"name": "ats_score", "score": ats_score},
    ]

    overall_score = round(
        sum(s["score"] for s in sections) / len(sections),
        2,
    ) if sections else 0.0

    found_kw = list(getattr(keyword, "found_keywords", []) or [])
    not_found_kw = list(getattr(keyword, "not_found_keywords", []) or [])
    top_3_kw = list(getattr(keyword, "top_3_keywords", []) or [])
    insights_list = list(getattr(alignment, "insights", []) or [])
    req_skills = int(getattr(alignment, "required_skills", 0) or 0)
    pref_skills = int(getattr(alignment, "preferred_skills", 0) or 0)
    experience = int(getattr(alignment, "experience", 0) or 0)
    education = int(getattr(alignment, "education", 0) or 0)
    cand_strengths = list(getattr(strengths, "candidate_strengths", []) or [])
    cand_improve = list(getattr(strengths, "candidates_areas_of_improvements", []) or [])

    return {
        "session_id": session_id,
        "resume_name": resume_name,
        "company": company,
        "role": role,
        "overall_score": overall_score,
        "job_match_score": job_match_score,
        "keywords_optimization": keywords_optimization,
        "sections": sections,
        "insights": insights_list,
        "candidate_strengths": cand_strengths,
        "candidates_areas_of_improvements": cand_improve,
        "job_alignment": {
            "required_skills": req_skills,
            "preferred_skills": pref_skills,
            "experience": experience,
            "education": education,
            "overall": job_match_score,
            "insights": insights_list,
        },
        "keywords": {
            "found": found_kw,
            "missing": not_found_kw,
            "jobSpecific": top_3_kw,
            "score": keywords_optimization,
        },
    }


def save_resume_analysis_to_db(
    user_email: str,
    session_id: str,
    analysis_result: Dict[str, Any],
    resume_filename: str = "Your_Resume.pdf",
) -> Dict[str, Any]:
    """
    Persist resume analysis via Nest internal API and return the full result dict for Redis.
    On POST failure, still returns the built dict so Celery can complete without crashing.
    analysis_result is the raw LangGraph State dict; nested values are Pydantic model instances.
    """
    full = build_resume_analysis_result_dict(session_id, resume_filename, analysis_result)

    payload = {
        "user_email": user_email,
        "session_id": full["session_id"],
        "resume_name": full["resume_name"],
        "company": full["company"],
        "role": full["role"],
        "overall_score": full["overall_score"],
        "job_match_score": full["job_match_score"],
        "format_and_structure": int(
            next((s["score"] for s in full["sections"] if s["name"] == "format_and_structure"), 0)
        ),
        "content_quality": int(
            next((s["score"] for s in full["sections"] if s["name"] == "content_quality"), 0)
        ),
        "length_and_conciseness": int(
            next((s["score"] for s in full["sections"] if s["name"] == "length_and_conciseness"), 0)
        ),
        "keywords_optimization": full["keywords_optimization"],
        "ats_score": int(next((s["score"] for s in full["sections"] if s["name"] == "ats_score"), 0)),
        "sections": full["sections"],
        "found_keywords": full["keywords"]["found"],
        "not_found_keywords": full["keywords"]["missing"],
        "top_3_keywords": full["keywords"]["jobSpecific"],
        "required_skills": full["job_alignment"]["required_skills"],
        "preferred_skills": full["job_alignment"]["preferred_skills"],
        "experience": full["job_alignment"]["experience"],
        "education": full["job_alignment"]["education"],
        "insights": full["insights"],
        "candidate_strengths": full["candidate_strengths"],
        "candidates_areas_of_improvements": full["candidates_areas_of_improvements"],
    }

    resp = _post("/api/internal/resume-analysis/", payload)
    if resp is False:
        logger.warning(
            "Nest internal POST /api/internal/resume-analysis/ failed; returning built result for Redis only"
        )
    return full


def _normalize_interview_type_for_drf(interview_type: str) -> str:
    """Normalize to what Django/DRF expects (e.g. TechnicalFeedback, get_session_history).
    Company and Subject coding interviews are stored as 'Coding Interview' so frontend shows the correct type."""
    if not interview_type:
        return "Technical Interview"
    t = interview_type.strip()
    if t in ("Company", "Subject"):
        return "Coding Interview"
    if t in ("Technical", "Coding", "Technical Interview", "Coding Interview", "Role-Based", "Role-Based Interview"):
        return "Technical Interview"
    if t in ("HR", "HR Interview"):
        return "HR Interview"
    if t in ("CaseStudy", "Case Study", "Case Study Interview"):
        return "Case Study Interview"
    if t in ("Communication", "Communication Interview"):
        return "Communication Interview"
    if t in ("Debate", "Debate Interview"):
        return "Debate Interview"
    return t


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
    interview_type_normalized = _normalize_interview_type_for_drf(interview_type)
    payload = {
        "user_email": user_email,
        "session_id": session_id,
        "interview_type": interview_type_normalized,
        "interview_test_id": interview_test_id,
        "duration_seconds": duration_seconds,
        "feedback_data": feedback_data,
        "interaction_log": interaction_log,
        "soft_skill_summary": soft_skill_summary or {},
        "big5_profile": big5_profile or {},
    }
    return _post("/interview-feedback/", payload)


def _seconds_to_duration_str(seconds: int) -> str:
    """Format seconds as HH:MM:SS for SaveFeedbackDto.duration."""
    if seconds is None or seconds < 0:
        return "00:00:00"
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def save_feedback_items_to_db(
    user_email: str,
    session_id: str,
    interview_test_id: str,
    items: Dict[str, Dict[str, int]],
    strengths: Optional[List[str]] = None,
    duration_seconds: Optional[int] = None,
    areas_of_improvements: Optional[List[str]] = None,
    interaction_logs: Optional[List[Any]] = None,
    interaction_status_logs: Optional[List[Any]] = None,
    user_id: Optional[str] = None,
    telemetry_data: Optional[Dict[str, Any]] = None,
    meeting_highlight: Optional[str] = None,
    communication_score: Optional[float] = None,
    grammar_score: Optional[float] = None,
    communication_metrics: Optional[Dict[str, Any]] = None,
    grammar_metrics: Optional[Dict[str, Any]] = None,
    is_free_interview: bool = False,
) -> bool:
    """
    Save feedback via NestJS internal API using SaveFeedbackDto schema only.
    Payload must contain DTO fields: interviewTestId, items, strengths, duration,
    areasOfImprovements, interactionLogs, interactionStatusLogs.
    ``telemetryData`` holds scored video telemetry (timeline, hire_probability, fillerWords, …) when present.
    user_email/session_id/user_id may optionally be forwarded if the Nest endpoint accepts them.
    """
    payload: Dict[str, Any] = {
        "interviewTestId": str(interview_test_id),
        "items": items,
        "strengths": strengths or [],
        "duration": _seconds_to_duration_str(duration_seconds or 0),
        "areasOfImprovements": areas_of_improvements or [],
        "interactionLogs": interaction_logs or [],
        "interactionStatusLogs": interaction_status_logs or [],
        "sessionId": session_id,
        "telemetryData": telemetry_data if telemetry_data is not None else {},
    }
    # Optionally include identity fields if the NestJS DTO supports them
    if user_email:
        payload["userEmail"] = user_email
    if session_id:
        payload["sessionId"] = session_id
    if user_id:
        payload["userId"] = user_id
    if isinstance(meeting_highlight, str) and meeting_highlight.strip():
        payload["meetingHighlight"] = meeting_highlight.strip()
    if communication_score is not None:
        payload["communicationScore"] = communication_score
    if grammar_score is not None:
        payload["grammarScore"] = grammar_score
    if communication_metrics is not None:
        payload["communicationMetrics"] = communication_metrics
    if grammar_metrics is not None:
        payload["grammarMetrics"] = grammar_metrics
    payload["isFreeInterview"] = is_free_interview
    path = "/interview-feedback/"
    full_url = f"{_drf_base_url()}{path}"
    logger.warning(
        "%s[SAVE_FEEDBACK] endpoint: %s%s",
        YELLOW,
        full_url,
        RESET,
    )
    logger.warning(
        "%s[SAVE_FEEDBACK] payload: %s%s",
        YELLOW,
        json.dumps(payload, indent=2, default=str),
        RESET,
    )
    return _post(path, payload)