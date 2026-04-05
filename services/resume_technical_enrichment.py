"""
Build initial TechnicalResearch / CodingResearch seeds from resume text so the technical
workflow's *_before nodes can narrow to a tailored question bank.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Tuple

from workflows.utils import get_llm

logger = logging.getLogger(__name__)

ENRICHMENT_SYSTEM = """You are an expert technical interviewer preparing a single session plan.

You may receive TARGET_JOB_TITLE and/or JOB_DESCRIPTION in addition to the RESUME. When those are present, bias theory and coding topics toward skills, stack, and problem types relevant to that role—not only what appears on the resume.

Given the inputs below, produce two plain-text sections for internal use only (not shown to the candidate):

SECTION A — THEORY_BANK
- 18–22 bullet lines: interview questions spanning OS, DBMS, computer networks, AND technologies from the resume and (when provided) the target role/JD (e.g. Kubernetes, Kafka, Redis).
- Difficulty should match seniority implied by the resume and role (intern/junior → more fundamentals; senior → more depth and trade-offs).
- One bullet = one clear question or prompt the interviewer can ask.

SECTION B — CODING_BANK
- 10–14 bullet lines: coding problem *ideas* (topic + short scenario). Align with data structures / patterns the resume and JD suggest (e.g. graphs if they list graph work or the role requires it).
- Avoid naming exact LeetCode problem numbers. Vary topics (arrays, strings, trees, graphs, DP, design of a small API/cache, etc.).

Format your reply EXACTLY like this (including the markers):

<<<THEORY_BANK>>>
(bullets here)
<<<CODING_BANK>>>
(bullets here)
"""


def _split_banks(raw: str) -> Tuple[str, str]:
    t = raw or ""
    low = t.lower()
    marker_t = "<<<theory_bank>>>"
    marker_c = "<<<coding_bank>>>"
    if marker_t in low and marker_c in low:
        i_t = low.index(marker_t)
        i_c = low.index(marker_c)
        if i_t < i_c:
            theory = t[i_t + len(marker_t) : i_c].strip()
            coding = t[i_c + len(marker_c) :].strip()
            return theory, coding or theory
    return t.strip(), ""


def enrich_technical_payload_from_resume(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    When resume_tailored_technical is true and resume text is present, set TechnicalResearch
    and CodingResearch to LLM-generated banks. Existing non-empty values are left unchanged.
    """
    if not payload.get("resume_tailored_technical"):
        return payload
    resume = (payload.get("resume") or "").strip()
    if len(resume) < 80:
        logger.info("resume_tailored_technical set but resume too short; skipping enrichment")
        return payload

    if (payload.get("TechnicalResearch") or "").strip() and (payload.get("CodingResearch") or "").strip():
        return payload

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_GENAI_API_KEY") or ""
    if not api_key:
        logger.warning("resume technical enrichment: no GOOGLE_API_KEY; skipping")
        return payload

    try:
        llm = get_llm(google_api_key=api_key, temperature=0.35, model=os.getenv("GEMINI_RESUME_ENRICH_MODEL", "models/gemini-2.5-flash"))
        rmax = 14_000
        resume_clip = resume[:rmax] + ("\n[truncated]" if len(resume) > rmax else "")
        title = (payload.get("job_title") or payload.get("jobTitle") or "").strip()
        jd = (payload.get("job_description") or payload.get("jobDescription") or "").strip()
        jd_max = 6_000
        jd_clip = jd[:jd_max] + ("\n[truncated]" if len(jd) > jd_max else "")
        extra_parts = []
        if title:
            extra_parts.append("TARGET_JOB_TITLE:\n" + title)
        if jd_clip:
            extra_parts.append("JOB_DESCRIPTION:\n" + jd_clip)
        extra = ("\n\n" + "\n\n".join(extra_parts)) if extra_parts else ""
        msg = ENRICHMENT_SYSTEM + extra + "\n\nRESUME:\n" + resume_clip
        out = llm.invoke(msg)
        text = out.content if hasattr(out, "content") else str(out)
        theory, coding = _split_banks(text)
        if not theory:
            theory = text.strip()
        if not coding:
            coding = (
                "- Implement a function that merges k sorted lists.\n"
                "- Design an in-memory LRU cache with O(1) get/put.\n"
                "- Given a grid, count islands (connected components).\n"
            )
        merged = {**payload, "TechnicalResearch": theory, "CodingResearch": coding}
        logger.info(
            "resume technical enrichment: theory_chars=%s coding_chars=%s",
            len(theory),
            len(coding),
        )
        return merged
    except Exception as e:
        logger.warning("resume technical enrichment failed: %s", e, exc_info=True)
        return payload
