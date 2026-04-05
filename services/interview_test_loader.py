"""
Load interview_tests row by UUID (JWT interviewTestId) for greeting / phase-engine context.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.I,
)


def _is_uuid(s: str) -> bool:
    return bool(s and _UUID_RE.match(s.strip()))


def interview_test_row_is_active(row: Dict[str, Any]) -> bool:
    """Normalize DB boolean / string from query_raw."""
    v = row.get("is_active")
    if v is False:
        return False
    if v is True:
        return True
    if isinstance(v, str) and v.lower() in ("false", "f", "0", "no"):
        return False
    return True


async def fetch_interview_test_by_id(interview_test_id: str) -> Optional[Dict[str, Any]]:
    """
    Return one row from interview_tests (plus companies.name when company_id is set).
    """
    raw = (interview_test_id or "").strip()
    if not raw:
        return None
    if not _is_uuid(raw):
        logger.warning("interview_test_id is not a UUID, skipping DB lookup: %r", raw)
        return None

    try:
        from services.prisma_db import ensure_prisma_env, get_prisma

        ensure_prisma_env()
        prisma = await get_prisma()
        try:
            await prisma.connect()
        except Exception:
            # already connected
            pass

        rows = await prisma.query_raw(
            """
            SELECT it.id::text AS id,
                   it.title,
                   it.difficulty::text AS difficulty,
                   it.company AS company_text,
                   it.subject AS subject_text,
                   it.is_active AS is_active,
                   it.fastapi_interview_type::text AS fastapi_interview_type,
                   c.name AS company_name
            FROM interview_tests it
            LEFT JOIN companies c ON c.id = it.company_id
            WHERE it.id = CAST($1 AS uuid)
            LIMIT 1
            """,
            raw,
        )
    except Exception as e:
        logger.warning("Could not load interview_tests row for id=%s: %s", raw, e)
        return None

    if not rows:
        logger.info("Interview test not found in DB (interview_tests.id=%s)", raw)
        return None

    row = dict(rows[0])
    logger.info(
        "Retrieved interview_tests row id=%s title=%r fastapi_interview_type=%r "
        "difficulty=%r is_active=%s",
        row.get("id"),
        row.get("title"),
        row.get("fastapi_interview_type"),
        row.get("difficulty"),
        row.get("is_active"),
    )
    return row


def apply_interview_test_to_payload(
    payload: Dict[str, Any],
    api_interview_type: str,
    row: Optional[Dict[str, Any]],
) -> None:
    """
    Fill company / subject / difficulty / role from DB when the client did not override.

    Mutates ``payload`` in place.
    """
    if not row:
        return

    title = (row.get("title") or "").strip()
    company_resolved = (row.get("company_name") or row.get("company_text") or "").strip()
    subject_resolved = (row.get("subject_text") or "").strip()
    diff = row.get("difficulty")

    if diff is not None and str(diff).strip() and not (
        payload.get("Difficulty") or payload.get("difficulty")
    ):
        payload["Difficulty"] = str(diff).strip()

    if title:
        payload["interview_test_title"] = title

    it = (api_interview_type or "").strip()

    if it == "Company":
        if not (payload.get("company") or payload.get("Company")):
            payload["company"] = company_resolved or title
        if not (payload.get("subject") or payload.get("Subject")):
            payload["subject"] = subject_resolved or "General"
    elif it in ("Subject", "Technical"):
        if not (payload.get("subject") or payload.get("Subject")):
            payload["subject"] = subject_resolved or title
        if not (payload.get("company") or payload.get("Company")):
            payload["company"] = company_resolved or "InterviewSta"
    elif it == "Role-Based Interview":
        if not (str(payload.get("role") or "").strip()):
            payload["role"] = title or "Frontend Development"
