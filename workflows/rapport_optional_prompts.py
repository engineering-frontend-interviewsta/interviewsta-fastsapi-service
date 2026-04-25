"""
Shared optional rapport prompts and helpers for interview workflows.
"""

from __future__ import annotations

import secrets
from typing import List

RAPPORT_OPTIONAL_PROMPTS: List[str] = [
    "Ask what kind of work environment helps them do their best work.",
    "Ask about a small win from the past month they feel proud of.",
    "Ask how they usually learn a brand new topic quickly.",
    "Ask about a recent challenge they solved and what they learned.",
    "Ask what kind of teams they enjoy collaborating with most.",
    "Ask what keeps them motivated on difficult days.",
    "Ask how they typically prepare before an important interview or exam.",
    "Ask which skill they are currently trying to improve and why.",
    "Ask about a project they enjoyed because of the impact, not just the tech.",
    "Ask how they handle feedback when something does not go as planned.",
    "Ask what type of problems they naturally enjoy solving.",
    "Ask what success in their next role would look like to them.",
    "Ask about one habit that improved their productivity recently.",
    "Ask how they balance consistency and speed when deadlines are tight.",
    "Ask what they want interviewers to understand about them beyond the resume.",
]


def pick_random_rapport_optional_prompt() -> str:
    """
    Select one optional rapport prompt using a random index.
    """
    idx = secrets.randbelow(len(RAPPORT_OPTIONAL_PROMPTS))
    return RAPPORT_OPTIONAL_PROMPTS[idx]
