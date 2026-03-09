# service.py
import json
from pathlib import Path
from typing import Any, Dict, Optional

from .schemas import (
    build_interview_feedback_models,
    FeedbackGraphState,
    _sanitize_metric_name,
)
from .graph import build_feedback_graph

# Load JSON config once at module level
_FEEDBACK_ITEMS: Dict[str, dict] = {}


def _load_feedback_items() -> None:
    global _FEEDBACK_ITEMS
    path = Path(__file__).parent / "feedback_items.json"
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)
    _FEEDBACK_ITEMS = {item["id"]: item for item in items}


_load_feedback_items()


def get_feedback_item(feedback_item_id: str) -> Optional[Dict[str, Any]]:
    """Return the feedback item config for the given id, or None if not found."""
    return _FEEDBACK_ITEMS.get(feedback_item_id)


def run_feedback_pipeline(
    feedback_item_id: str,
    history_log: str,
    google_api_key: str,
) -> Dict[str, Any]:
    """
    Runs the full feedback pipeline for a given feedback item id (e.g. "fi-coding-i").

    Returns a structured dict with:
      - interview_type_id, interview_title
      - sleeve_scores: sleeve_name -> { metric_name: score } (original names; -1 for unscored)
      - strengths_and_improvements: { strength1..3, improvement1..3 }
      - interaction_feedback: [ { status, comment }, ... ]
    """
    from ..utils import get_llm

    if feedback_item_id not in _FEEDBACK_ITEMS:
        raise ValueError(f"Unknown feedback item id: {feedback_item_id}")

    feedback_item = _FEEDBACK_ITEMS[feedback_item_id]
    sleeve_models = build_interview_feedback_models(feedback_item)
    llm = get_llm(google_api_key)
    agent = build_feedback_graph(sleeve_models, llm)

    initial_state: FeedbackGraphState = {
        "history_log": history_log,
        "interview_type_id": feedback_item_id,
        "interview_title": feedback_item["title"],
        "sleeve_models": sleeve_models,
        "sleeve_scores": {},
        "strengths_and_improvements": None,
        "interaction_feedback": None,
    }

    final_state = agent.invoke(initial_state)

    # Serialize sleeve_scores to dicts with original metric names; -1 for unscored
    serialized_sleeves: Dict[str, Dict[str, int]] = {}
    for sleeve_name, score_obj in final_state["sleeve_scores"].items():
        raw = score_obj.model_dump()
        original_metrics = list(feedback_item["items"][sleeve_name].keys())
        serialized_sleeves[sleeve_name] = {
            orig: (raw.get(_sanitize_metric_name(orig)) if raw.get(_sanitize_metric_name(orig)) is not None else -1)
            for orig in original_metrics
        }

    return {
        "interview_type_id": feedback_item_id,
        "interview_title": feedback_item["title"],
        "sleeve_scores": serialized_sleeves,
        "strengths_and_improvements": (
            final_state["strengths_and_improvements"].model_dump()
            if final_state["strengths_and_improvements"] else None
        ),
        "interaction_feedback": [
            item.model_dump() for item in (final_state["interaction_feedback"] or [])
        ],
    }
