"""
Map unified feedback ``sleeve_scores`` (metric -> int, -1 = unscored) to a 0–100 overall %.
"""
from __future__ import annotations

from typing import Any, Dict, Optional


def sleeve_scores_to_overall_pct(sleeve_scores: Optional[Dict[str, Dict[str, Any]]]) -> Optional[float]:
    """
    Average all scored metrics (>= 0). Values are treated as 0–10 rubric scores when max <= 10,
    otherwise as already 0–100. Returns None if no valid scores.
    """
    if not sleeve_scores:
        return None
    vals: list[float] = []
    for _sleeve, metrics in sleeve_scores.items():
        if not isinstance(metrics, dict):
            continue
        for _k, v in metrics.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and float(v) >= 0.0:
                vals.append(float(v))
    if not vals:
        return None
    avg = sum(vals) / len(vals)
    hi = max(vals)
    if hi <= 10.5:
        pct = avg * 10.0
    else:
        pct = min(100.0, avg)
    return round(max(0.0, min(100.0, pct)), 1)
