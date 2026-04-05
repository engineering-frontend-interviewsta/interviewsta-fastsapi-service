"""Build LangGraph invoke state for the phase engine (workflows.interview.phase_engine)."""

from __future__ import annotations

from typing import Any, Dict, List


def _question_row_to_coding_phase_dict(q: Any) -> Dict[str, Any]:
    if isinstance(q, dict):
        return {
            "title": q.get("question_title") or q.get("title") or "",
            "description": q.get("question_description") or q.get("description") or "",
            "difficulty": q.get("question_difficulty") or q.get("difficulty") or "Medium",
            "raw_content": q.get("question_raw_content") or q.get("raw_content") or "",
            "question_type": q.get("question_type") or "coding",
        }
    return {
        "title": getattr(q, "question_title", "") or getattr(q, "title", "") or "",
        "description": getattr(q, "question_description", "") or getattr(q, "description", "") or "",
        "difficulty": getattr(q, "question_difficulty", None) or getattr(q, "difficulty", None) or "Medium",
        "raw_content": getattr(q, "question_raw_content", None) or getattr(q, "raw_content", None) or "",
        "question_type": getattr(q, "question_type", None) or "coding",
    }


def merged_coding_questions_for_api(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Flatten phase_questions Theoretical + Coding into one list (theoretical first),
    using the shape expected by _serialize_questions_for_payload / frontend.
    """
    pq = response.get("phase_questions") or {}
    out: List[Dict[str, Any]] = []
    for src, qtype in ((pq.get("Theoretical") or []), "theoretical"), ((pq.get("Coding") or []), "coding"):
        for x in src:
            d = x if isinstance(x, dict) else {}
            out.append(
                {
                    "question_title": d.get("title") or d.get("question_title") or "",
                    "question_description": d.get("description") or d.get("question_description") or "",
                    "question_difficulty": d.get("difficulty") or d.get("question_difficulty") or "Medium",
                    "question_type": d.get("question_type") or qtype,
                    "question_raw_content": d.get("raw_content") or d.get("question_raw_content") or "",
                }
            )
    return out


def question_progress_from_phase_response(
    response: Dict[str, Any],
    interview_type: str,
) -> tuple:
    """Return (question_number, total_questions, question_raw_content) for Company/Subject/Technical."""
    if interview_type not in ("Company", "Subject", "Technical"):
        return None, None, None
    merged = merged_coding_questions_for_api(response)
    total = len(merged)
    if total == 0:
        return None, None, None
    pq = response.get("phase_questions") or {}
    idx_c = (response.get("phase_question_idx") or {}).get("Coding") or 0
    idx_t = (response.get("phase_question_idx") or {}).get("Theoretical") or 0
    theo_n = len(pq.get("Theoretical") or [])
    if theo_n > 0 and idx_t < theo_n:
        flat_idx = idx_t
    else:
        flat_idx = theo_n + idx_c
    flat_idx = min(flat_idx, total - 1)
    q = merged[flat_idx] if flat_idx < len(merged) else merged[-1]
    raw = q.get("question_raw_content") or ""
    return flat_idx + 1, total, raw or None


def build_initial_invoke_state(bundle: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    """Initial state for phase_engine.BaseInterviewState."""
    meta = dict(bundle.get("interview_meta") or {})
    extra = dict(bundle.get("extra_background") or {})
    background = {**meta, **extra}
    phase_questions: Dict[str, List] = {}
    raw_q = payload.get("Questions") or []
    if raw_q:
        phase_questions["Coding"] = [_question_row_to_coding_phase_dict(q) for q in raw_q]

    return {
        "messages": [],
        "history": "",
        "LastNode": "",
        "background": background,
        "phase_questions": phase_questions,
        "phase_question_idx": {},
        "phase_state": dict(bundle.get("phase_state") or {}),
    }
