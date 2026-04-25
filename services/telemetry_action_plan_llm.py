"""
LLM-generated extensions to video-telemetry action plans (structured output).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

ActionUrgency = Literal["from_today", "this_week", "next_week", "next_few_weeks"]


class LlmActionPlanItem(BaseModel):
    title: str = Field(..., max_length=200)
    detail: str = Field(..., max_length=800)
    urgency: ActionUrgency
    category: Literal["technical", "presence", "speech", "environment"]


class LlmActionPlanBatch(BaseModel):
    items: List[LlmActionPlanItem] = Field(default_factory=list, max_length=4)


ACTION_PLAN_EXTEND_PROMPT = """You extend a post-interview coaching action plan for VIDEO / delivery telemetry.

Already planned (do NOT repeat or lightly rephrase these titles):
{existing_titles}

Context — gaps to address (if any):
{gaps_text}

Presence / speech / environment summaries:
{soft_summary}

Add 2 to 4 NEW concrete action items the candidate can follow. Each must have:
- title: short imperative (max ~12 words)
- detail: one or two sentences, specific and actionable
- urgency: exactly one of: from_today | this_week | next_week | next_few_weeks
- category: one of: technical | presence | speech | environment (pick best fit; use "speech" for vocal habits, "presence" for camera/body language, "environment" for desk/lighting/mic, "technical" only if coding practice fits)

Do not output duplicate ideas. Prefer items that complement the existing plan."""


def extend_video_telemetry_action_plan_llm(
    *,
    existing_plan: List[Dict[str, Any]],
    gaps_text: str,
    soft_summary: str,
    google_api_key: str,
    max_llm_items: int = 4,
) -> List[Dict[str, Any]]:
    """
    Returns **additional** action items (dicts without ``rank``) from the LLM, or [] on failure.
    """
    if not google_api_key:
        return []
    titles = [str(x.get("title", "")).strip() for x in existing_plan if x.get("title")]
    existing_titles = "\n".join(f"- {t}" for t in titles[:20]) or "(none)"
    gaps_block = (gaps_text or "").strip()[:4000] or "(none listed)"
    soft_block = (soft_summary or "").strip()[:4000] or "(none)"

    try:
        from workflows.utils import get_llm

        llm = get_llm(google_api_key=google_api_key, temperature=0.35)
        structured = llm.with_structured_output(LlmActionPlanBatch)
        prompt = ACTION_PLAN_EXTEND_PROMPT.format(
            existing_titles=existing_titles,
            gaps_text=gaps_block,
            soft_summary=soft_block,
        )
        batch: LlmActionPlanBatch = structured.invoke(prompt)
        raw = batch.items[:max_llm_items]
        out: List[Dict[str, Any]] = []
        for it in raw:
            out.append(
                {
                    "title": it.title.strip(),
                    "detail": it.detail.strip(),
                    "urgency": it.urgency,
                    "category": it.category,
                }
            )
        return out
    except Exception as e:
        logger.warning("[telemetry_action_plan_llm] extend failed: %s", e, exc_info=True)
        return []
