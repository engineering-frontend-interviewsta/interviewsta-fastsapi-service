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

# JWT / clients often send stable short ids (see docs/INTERVIEW_AND_FEEDBACK_API.md); JSON uses UUIDs.
_LEGACY_ID_ALIASES: Dict[str, str] = {
    "fi-coding-i": "f45201b4-c099-4221-a6e8-be9f0304e8d3",
    "fi-faang": "423b67d5-493f-4020-a027-bbefbde253c3",
    "fi-product-based": "5bcff863-406b-4f23-938d-04ed0d6f7dd",
    "fi-mass-hiring": "b22c0e6f-f9a6-440b-929a-6b49dff14a69",
    "fi-communication": "420a5484-b317-48e8-afa8-5ad001714809",
    "fi-debate": "f1b2abe6-06c5-4238-92f4-00bd3375d62",
    "fi-role-based": "79b985a8-4dd6-48b0-bc6b-2e32967366ef",
}


def _load_feedback_items() -> None:
    global _FEEDBACK_ITEMS
    path = Path(__file__).parent / "feedback_items.json"
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)
    _FEEDBACK_ITEMS = {item["id"]: item for item in items}
    for alias, canonical in _LEGACY_ID_ALIASES.items():
        if canonical in _FEEDBACK_ITEMS and alias not in _FEEDBACK_ITEMS:
            _FEEDBACK_ITEMS[alias] = _FEEDBACK_ITEMS[canonical]


_load_feedback_items()


def get_feedback_item(feedback_item_id: str) -> Optional[Dict[str, Any]]:
    """Return the feedback item config for the given id, or None if not found."""
    if feedback_item_id is None:
        return None
    k = str(feedback_item_id).strip()
    if not k:
        return None
    if k in _FEEDBACK_ITEMS:
        return _FEEDBACK_ITEMS[k]
    kl = k.lower()
    if kl in _FEEDBACK_ITEMS:
        return _FEEDBACK_ITEMS[kl]
    for existing_key, item in _FEEDBACK_ITEMS.items():
        if existing_key.lower() == kl:
            return item
    return None


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

    feedback_item = get_feedback_item(feedback_item_id)
    if feedback_item is None:
        raise ValueError(f"Unknown feedback item id: {feedback_item_id}")
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
