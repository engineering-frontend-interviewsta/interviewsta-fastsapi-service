"""
Heuristic filler-word counts from interviewee transcript (no extra NLP deps).
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Set

# Common English fillers / discourse markers (single tokens).
_FILLER_UNIGRAMS: Set[str] = frozenset(
    {
        "um",
        "umm",
        "uh",
        "uhh",
        "er",
        "erm",
        "ah",
        "oh",
        "hmm",
        "hm",
        "like",
        "basically",
        "literally",
        "actually",
        "right",
        "okay",
        "ok",
        "well",
        "so",
        "yeah",
        "yep",
        "yup",
        "nope",
        "kinda",
        "sorta",
    }
)

_FILLER_BIGRAMS: Set[str] = frozenset(
    {
        "you know",
        "i mean",
        "sort of",
        "kind of",
    }
)


def top_filler_words_from_transcript(transcript: str, *, max_items: int = 5) -> List[Dict[str, Any]]:
    """
    Return up to ``max_items`` filler tokens/phrases with counts (interviewee text only).
    Output shape: ``[{"word": str, "count": int}, ...]`` sorted by count desc.
    """
    if not (transcript or "").strip():
        return []
    text = transcript.lower()
    tokens = re.findall(r"[a-z0-9']+", text)
    if not tokens:
        return []

    ctr: Counter[str] = Counter()
    for w in tokens:
        if w in _FILLER_UNIGRAMS:
            ctr[w] += 1
    for a, b in zip(tokens, tokens[1:]):
        bg = f"{a} {b}"
        if bg in _FILLER_BIGRAMS:
            ctr[bg] += 1

    if not ctr:
        return []
    out: List[Dict[str, Any]] = []
    for w, n in ctr.most_common(max_items):
        out.append({"word": w, "count": int(n)})
    return out
