# schemas.py
from pydantic import BaseModel, Field, create_model
from typing import Optional, Literal, Dict, Any
from typing_extensions import TypedDict


# ── Dynamic skill model factory ─────────────────────────────────────────────

def _sanitize_metric_name(name: str) -> str:
    """Convert metric name to valid Python field name (e.g. 'Requirement Clarification' -> 'Requirement_Clarification')."""
    return name.replace(" ", "_").replace("&", "and")


def build_skill_model(sleeve_name: str, metrics: list[str]) -> type[BaseModel]:
    """
    Dynamically creates a Pydantic model for a sleeve (e.g. 'Problem Solving & Technical Logic').
    Each metric is Optional[int] — None means not tested. Uses sanitized field names for Pydantic.
    """
    fields = {
        _sanitize_metric_name(metric): (
            Optional[int],
            Field(
                default=None,
                description=(
                    f"Score 0-100 (granular, e.g. 67, 73) or null if not tested. "
                    f"Metric: {metric} under sleeve: {sleeve_name}."
                )
            )
        )
        for metric in metrics
    }
    return create_model(sleeve_name.replace(" ", "_").replace("&", "and"), **fields)


def build_interview_feedback_models(feedback_item: dict) -> Dict[str, type[BaseModel]]:
    """
    Given a feedback item's `items` dict, returns a map of sleeve_name -> dynamic Pydantic model.
    """
    return {
        sleeve_name: build_skill_model(sleeve_name, list(metrics.keys()))
        for sleeve_name, metrics in feedback_item["items"].items()
    }


# ── Fixed models (shared across all interview types) ─────────────────────────

class StrengthsAndImprovements(BaseModel):
    """
    Provide 3 specific strengths and 3 actionable areas for improvement.
    Address the interviewee in second person. Be concise and context-aware
    of the interview type and the actual questions asked.
    """
    strength1: str = Field(..., description="Crisp, specific strength in second person.")
    strength2: str = Field(..., description="Crisp, specific strength in second person.")
    strength3: str = Field(..., description="Crisp, specific strength in second person.")
    improvement1: str = Field(..., description="Crisp, actionable improvement in second person.")
    improvement2: str = Field(..., description="Crisp, actionable improvement in second person.")
    improvement3: str = Field(..., description="Crisp, actionable improvement in second person.")


class InteractionFeedbackItem(BaseModel):
    """
    For each question-answer pair in the chat log.
    """
    status: Literal[
        "correct", "incorrect", "partially-correct", "cross-question"
    ] = Field(..., description="Correctness label for this interaction.")
    comment: str = Field(..., description="Improvement comment. Empty string if correct.")


# ── Graph state ───────────────────────────────────────────────────────────────

class FeedbackGraphState(TypedDict):
    history_log: str                          # Serialized chat log
    interview_type_id: str                    # e.g. "fi-coding-i"
    interview_title: str                      # e.g. "Coding - I"
    sleeve_models: Dict[str, Any]             # sleeve_name -> Pydantic class (injected before run)
    sleeve_scores: Dict[str, Any]             # sleeve_name -> model instance (populated by nodes)
    strengths_and_improvements: Optional[StrengthsAndImprovements]
    interaction_feedback: Optional[list[InteractionFeedbackItem]]
