"""
Interview telemetry scoring service.

Builds a ``FullAnalysisReport`` from Redis-stored per-interval samples (and optional
one-shot ``environment``), then runs ``ScoringEngine`` (technical rubric is stubbed
until wired to real rubric data).

Also hosts the Pydantic models and ``ScoringEngine`` moved from
``workflows/telemetery/main.py``.
"""

from __future__ import annotations
import json
import logging
import math
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ─── Input models ─────────────────────────────────────────────────────────────

class RubricCategory(BaseModel):
    """Maps 1:1 to the rubric JSON from the frontend."""
    scores: dict[str, float]  # metric_name → 0–10

class TechnicalScores(BaseModel):
    algorithmic_problem_solving: RubricCategory
    code_design_implementation:  RubricCategory
    system_thinking_tradeoffs:   RubricCategory
    communication_collaboration: RubricCategory

class GazeStats(BaseModel):
    camera_percent:    float
    screen_percent:    float
    down_percent:      float
    longest_down_gap_ms: float
    avg_confidence:    float

class PostureStats(BaseModel):
    upright_percent:       float
    forward_lean_percent:  float
    slouch_events:         int
    avg_spine_angle:       float

class StressStats(BaseModel):
    avg_brow_raise:           float
    avg_lip_compression:      float
    stress_events:            int
    stress_spike_timestamps:  list[float]

class FidgetStats(BaseModel):
    face_touch_count: int
    hair_touch_count: int

class PresenceScores(BaseModel):
    eye_contact: float
    posture:     float
    composure:   float
    gestures:    float
    overall:     float

class PresenceReport(BaseModel):
    session_duration_ms: float
    gaze:    GazeStats
    posture: PostureStats
    stress:  StressStats
    fidget:  FidgetStats
    scores:  PresenceScores

class SpeechStats(BaseModel):
    total_words:             int
    speaking_time_ms:        float
    avg_wpm:                 float
    fillers_per_minute:      float
    dead_pause_count:        int
    longest_silence_ms:      float
    avg_noise_floor_db:      float
    avg_snr_db:              float
    transcription_confidence: float

class TopFiller(BaseModel):
    word:  str
    count: int

class SpeechScores(BaseModel):
    pace:           float
    filler_density: float
    silence_control: float
    audio_clarity:  float
    overall:        float

class SpeechReport(BaseModel):
    stats:        SpeechStats
    top_fillers:  list[TopFiller]
    scores:       SpeechScores
    full_transcript: str

class EnvironmentItem(BaseModel):
    score:   float
    verdict: str
    issue:   Optional[str] = None

class LightingEnv(EnvironmentItem):
    backlight_detected: bool
    face_brightness:    float

class CameraEnv(EnvironmentItem):
    estimated_angle_deg: float
    is_above_eye_level:  bool

class BackgroundEnv(EnvironmentItem):
    edge_density:     float
    motion_detected:  bool

class AudioEnv(EnvironmentItem):
    noise_floor_db:       float
    has_background_noise: bool
    echo_detected:        bool
    external_event_count: int


class MicrophoneEnv(EnvironmentItem):
    """
    Input device judgment (built-in vs headset vs external), separate from ambient ``audio`` / noise.
    Client: top-level ``microphone`` on each sample or nested ``audio.microphone``; optional ``environment.microphone``.
    """

    device_kind: Optional[str] = None  # "built_in" | "headset" | "external_mic" | "other"
    suggestion: Optional[str] = None
    has_client_snapshot: bool = False


class AttireEnv(EnvironmentItem):
    """On-camera clothing / torso capture from client ``environment.attire``."""

    color_class: Optional[str] = None
    brightness_value: Optional[float] = None
    moire_risk: bool = False
    suggestion: Optional[str] = None
    has_client_snapshot: bool = False


class EnvironmentReport(BaseModel):
    lighting:     LightingEnv
    camera:       CameraEnv
    background:   BackgroundEnv
    audio:        AudioEnv
    microphone:   MicrophoneEnv
    attire:       AttireEnv
    overall_score: float
    critical_issues: list[str]
    suggestions:     list[str]

class FullAnalysisReport(BaseModel):
    presence:        PresenceReport
    speech:          SpeechReport
    environment:     EnvironmentReport
    composite_score: float

class ScoringRequest(BaseModel):
    session_id:   str
    technical:    TechnicalScores
    analysis:     FullAnalysisReport
    session_duration_minutes: float


# ─── Output models ────────────────────────────────────────────────────────────

class MetricScore(BaseModel):
    name:       str
    score:      float          # 0–10
    pct:        float          # 0–100 for display
    note:       str            # one-line specific feedback

class CategoryScore(BaseModel):
    name:    str
    metrics: list[MetricScore]
    avg:     float
    verdict: str               # "strong" | "developing" | "weak"
    narrative: str             # 2–3 sentence paragraph explaining the category

class PresenceDimension(BaseModel):
    name:      str
    score:     float           # 0–100
    stats:     dict            # raw measurements for display
    narrative: str

class SpeechDimension(BaseModel):
    avg_wpm:            float
    fillers_per_minute: float
    dead_pauses:        int
    transcription_conf: float
    top_fillers:        list[TopFiller]
    score:              float
    narrative:          str

class EnvironmentDimension(BaseModel):
    items: list[dict]          # [{label, score, verdict, note}]
    overall_score: float
    critical_issues: list[str]

class StrengthItem(BaseModel):
    title:     str
    detail:    str
    source:    str             # "technical" | "presence" | "speech"

class GapItem(BaseModel):
    title:     str
    detail:    str
    impact:    str             # "high" | "medium" | "low"
    source:    str

class HireImpactItem(BaseModel):
    label:      str
    probability_delta: float   # +/- percentage points
    action:     str

class HireProbability(BaseModel):
    probability:  float        # 0–100
    verdict:      str          # headline sentence
    narrative:    str
    breakdown:    dict[str, float]
    impact_items: list[HireImpactItem]

class ActionItem(BaseModel):
    rank:     int
    title:    str
    detail:   str
    urgency:  str              # "from_today" | "this_week" | "next_week" | "next_few_weeks" (+ legacy "today", "two_weeks")
    category: str              # "technical" | "presence" | "speech" | "environment"

class ScoredFeedback(BaseModel):
    session_id: str
    overall_score: float
    technical_categories: list[CategoryScore]
    presence_dimensions:  list[PresenceDimension]
    speech_dimension:     SpeechDimension
    environment_dimension: EnvironmentDimension
    strengths: list[StrengthItem]
    gaps:      list[GapItem]
    hire_probability: HireProbability
    action_plan: list[ActionItem]


class VideoTelemetryScoreResult(BaseModel):
    """Video-only scoring: no technical rubric; includes per-interval ``timeline``."""
    session_id: str
    overall_score: float
    timeline: list[dict[str, Any]]
    presence_dimensions: list[PresenceDimension]
    speech_dimension: SpeechDimension
    environment_dimension: EnvironmentDimension
    strengths: list[StrengthItem]
    gaps: list[GapItem]
    hire_probability: HireProbability
    action_plan: list[ActionItem]


def _extract_attire_dict(
    environment_once: Optional[Dict[str, Any]], samples: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Prefer one-shot ``environment.attire``; fall back to any sample that still embeds ``environment``."""
    if environment_once:
        a = environment_once.get("attire")
        if isinstance(a, dict) and a:
            return dict(a)
    for s in reversed(samples):
        env = s.get("environment")
        if isinstance(env, dict):
            a = env.get("attire")
            if isinstance(a, dict) and a:
                return dict(a)
    return None


def _attire_from_client_dict(a: Dict[str, Any]) -> AttireEnv:
    sc = a.get("score")
    score_f = float(sc) if isinstance(sc, (int, float)) else 75.0
    ver = a.get("verdict")
    verdict = str(ver).strip() if isinstance(ver, str) and str(ver).strip() else "good"
    iss = a.get("issue")
    issue = iss.strip() if isinstance(iss, str) and iss.strip() else None
    sug = a.get("suggestion")
    suggestion = sug.strip() if isinstance(sug, str) and sug.strip() else None
    cc = a.get("colorClass")
    color_class = cc.strip() if isinstance(cc, str) and cc.strip() else None
    bv = a.get("brightnessValue")
    brightness_value = float(bv) if isinstance(bv, (int, float)) else None
    moire_risk = a.get("moireRisk") is True
    return AttireEnv(
        score=round(score_f, 1),
        verdict=verdict,
        issue=issue,
        color_class=color_class,
        brightness_value=brightness_value,
        moire_risk=moire_risk,
        suggestion=suggestion,
        has_client_snapshot=True,
    )


def _default_attire_env() -> AttireEnv:
    return AttireEnv(
        score=75.0,
        verdict="good",
        issue=None,
        color_class=None,
        brightness_value=None,
        moire_risk=False,
        suggestion=None,
        has_client_snapshot=False,
    )


def _attire_dimension_note(a: AttireEnv) -> str:
    """Human-readable line for feedback when client ``issue`` / ``suggestion`` are empty."""
    if not a.has_client_snapshot:
        return "No attire snapshot was captured for this session."
    if a.issue:
        return a.issue
    if a.suggestion:
        return a.suggestion
    parts: List[str] = []
    if a.moire_risk:
        parts.append(
            "Moiré risk on clothing — solid colors or more distance from the camera usually help"
        )
    if a.color_class:
        parts.append(f"Palette reads as {a.color_class.replace('_', ' ')}")
    if a.brightness_value is not None:
        parts.append(f"torso brightness ~{a.brightness_value:.0f}")
    if parts:
        return "; ".join(parts) + "."
    vlow = (a.verdict or "").lower()
    if a.score >= 85 or vlow in ("good", "excellent", "strong"):
        return "Attire reads clearly on camera for this session."
    if a.verdict and vlow != "good":
        return a.verdict.capitalize()
    return "No attire snapshot was captured for this session."


def _mic_section_from_sample(s: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    m = s.get("microphone")
    if isinstance(m, dict) and m:
        return m
    aud = s.get("audio")
    if isinstance(aud, dict):
        inner = aud.get("microphone")
        if isinstance(inner, dict) and inner:
            return inner
    return None


def _infer_microphone_device_kind(m: Dict[str, Any]) -> Optional[str]:
    raw: Optional[str] = None
    for key in (
        "deviceKind",
        "microphoneType",
        "inputKind",
        "micDeviceClass",
        "sourceType",
        "type",
        "kind",
    ):
        v = m.get(key)
        if isinstance(v, str) and v.strip():
            raw = v.strip().lower()
            break
    if raw is None:
        return None
    if any(
        t in raw
        for t in (
            "built",
            "internal",
            "laptop",
            "integrated",
            "default_system",
            "system_default",
        )
    ):
        return "built_in"
    if any(
        t in raw
        for t in (
            "headset",
            "headphone",
            "headphones",
            "earbud",
            "earpods",
            "airpod",
            "bluetooth",
        )
    ):
        return "headset"
    if any(
        t in raw
        for t in (
            "usb",
            "external",
            "xm",
            "snowball",
            "yeti",
            "condenser",
            "dynamic",
            "interface",
        )
    ):
        return "external_mic"
    return "other"


def _default_microphone_env() -> MicrophoneEnv:
    return MicrophoneEnv(
        score=75.0,
        verdict="good",
        issue=None,
        device_kind=None,
        suggestion=None,
        has_client_snapshot=False,
    )


def _build_microphone_env(
    environment_once: Optional[Dict[str, Any]], samples: List[Dict[str, Any]]
) -> MicrophoneEnv:
    mic_scores: List[float] = []
    last_m: Optional[Dict[str, Any]] = None
    for s in samples:
        m = _mic_section_from_sample(s)
        if m:
            last_m = m
            sc = m.get("score")
            if isinstance(sc, (int, float)):
                mic_scores.append(float(sc))
    env_mic: Optional[Dict[str, Any]] = None
    if environment_once:
        x = environment_once.get("microphone")
        if isinstance(x, dict) and x:
            env_mic = x
    if last_m is None and env_mic is not None:
        last_m = env_mic
    if not mic_scores and env_mic is not None:
        sc0 = env_mic.get("score")
        if isinstance(sc0, (int, float)):
            mic_scores.append(float(sc0))

    if last_m is None:
        return _default_microphone_env()

    score_f = round(_mean(mic_scores), 1) if mic_scores else 75.0
    if not mic_scores and isinstance(last_m.get("score"), (int, float)):
        score_f = round(float(last_m["score"]), 1)
    ver = last_m.get("verdict")
    verdict = str(ver).strip() if isinstance(ver, str) and str(ver).strip() else "good"
    iss = last_m.get("issue")
    issue = iss.strip() if isinstance(iss, str) and iss.strip() else None
    sug = last_m.get("suggestion")
    suggestion = sug.strip() if isinstance(sug, str) and sug.strip() else None
    return MicrophoneEnv(
        score=float(score_f),
        verdict=verdict,
        issue=issue,
        device_kind=_infer_microphone_device_kind(last_m),
        suggestion=suggestion,
        has_client_snapshot=True,
    )


def _microphone_dimension_note(m: MicrophoneEnv) -> str:
    if not m.has_client_snapshot:
        return "No microphone profile was captured for this session."
    if m.issue:
        return m.issue
    if m.suggestion:
        return m.suggestion
    if m.device_kind == "built_in":
        return (
            "Built-in / laptop microphone — interviewers tend to hear more room tone and echo; "
            "a headset or USB mic is safer for high-stakes rounds."
        )
    if m.device_kind == "headset":
        return (
            "Headset or on-ear style input — typically clearer and more isolated than a built-in mic."
        )
    if m.device_kind == "external_mic":
        return (
            "External / USB-style microphone — strong choice for interview voice clarity."
        )
    vlow = (m.verdict or "").lower()
    if m.score >= 85 or vlow in ("good", "excellent", "strong"):
        return "Microphone capture looks solid for this session."
    if m.verdict and vlow != "good":
        return m.verdict.capitalize()
    return "Microphone input recorded; consider tagging device type (built-in vs headset) for sharper tips."


# ─── Scoring engine ───────────────────────────────────────────────────────────

class ScoringEngine:

    TECHNICAL_WEIGHT  = 0.45
    PRESENCE_WEIGHT   = 0.25
    SPEECH_WEIGHT     = 0.20
    ENVIRONMENT_WEIGHT = 0.10

    # Per-metric feedback templates
    # {score_range: note}
    METRIC_NOTES: dict[str, list[tuple[float, str]]] = {
        "Problem Decomposition": [
            (8, "Excellent — structured the problem into clear subproblems before coding."),
            (6, "Good — broke the problem down, though some steps needed prompting."),
            (4, "Partial — decomposition attempted but key subproblems were missed."),
            (0, "Jumped straight to coding without a clear decomposition strategy."),
        ],
        "Algorithm Selection": [
            (8, "Strong — selected an efficient algorithm and justified the choice."),
            (6, "Reasonable choice, though a more optimal algorithm was available."),
            (4, "Algorithm selected was suboptimal for the given constraints."),
            (0, "Algorithm selection was incorrect or not justified."),
        ],
        "Time Complexity Reasoning": [
            (8, "Accurate O() analysis derived from first principles."),
            (6, "Correct complexity identified with minor justification gaps."),
            (4, "Analysis had errors — complexity was off by at least one factor."),
            (0, "No complexity analysis provided or analysis was significantly incorrect."),
        ],
        "Space Complexity Reasoning": [
            (8, "Correctly accounted for all memory usage including call stack."),
            (6, "Mostly correct — auxiliary space identified but stack not mentioned."),
            (4, "Space analysis was incomplete or incorrect."),
            (0, "No space complexity analysis provided."),
        ],
        "Edge Case Identification": [
            (8, "Proactively identified null, empty, single-element, and overflow cases."),
            (6, "Caught main edge cases but missed one or two less obvious ones."),
            (4, "Only caught edge cases when directly prompted."),
            (0, "Edge cases not addressed."),
        ],
        "Data Structure Choice": [
            (8, "Optimal data structure chosen and trade-offs explained clearly."),
            (6, "Good choice, minor optimisation opportunity missed."),
            (4, "Functional but suboptimal data structure used."),
            (0, "Inappropriate data structure significantly hurt solution quality."),
        ],
        "Modular Code Design": [
            (8, "Code well-organised into focused, named functions."),
            (6, "Generally modular — some functions could be further decomposed."),
            (4, "Code functional but monolithic — single large function."),
            (0, "No modular structure evident."),
        ],
        "Correctness of Implementation": [
            (8, "Implementation correct and handles all test cases."),
            (6, "Mostly correct — minor bug that could be fixed quickly."),
            (4, "Implementation has a logical error affecting correctness."),
            (0, "Implementation does not produce correct output."),
        ],
        "Code Optimization": [
            (8, "Identified and removed redundant operations proactively."),
            (6, "Basic optimisations applied — further improvements possible."),
            (4, "Redundant operations present that affect performance."),
            (0, "No optimisation attempted."),
        ],
        "Scalability Awareness": [
            (8, "Discussed how solution behaves at 10x, 100x scale with specifics."),
            (6, "Mentioned scalability concerns but without quantification."),
            (4, "Scale mentioned only when prompted."),
            (0, "No scalability discussion."),
        ],
        "Tradeoff Discussion": [
            (8, "Proactively discussed time/space, read/write, consistency trade-offs."),
            (6, "Discussed one main trade-off when asked."),
            (4, "Trade-offs only discussed with significant prompting."),
            (0, "No trade-off discussion."),
        ],
        "Alternative Solutions": [
            (8, "Proposed 2+ alternatives and compared them analytically."),
            (6, "Proposed one alternative when asked."),
            (4, "Alternatives only mentioned after direct prompting."),
            (0, "No alternatives proposed."),
        ],
        "Constraint Awareness": [
            (8, "Asked clarifying questions about constraints before starting."),
            (6, "Verified most constraints — one key one missed."),
            (4, "Constraints checked only after partial implementation."),
            (0, "No constraint checking."),
        ],
        "Clarity of Thought": [
            (8, "Explanations were structured, logical, and easy to follow."),
            (6, "Generally clear — occasional jumps in reasoning."),
            (4, "Some explanations were confusing or required re-asking."),
            (0, "Explanations were unclear or incoherent."),
        ],
        "Think Aloud Process": [
            (8, "Consistent narration of reasoning throughout — no dead silences."),
            (6, "Good narration with 1–2 silent gaps."),
            (4, "Narrated when prompted but defaulted to silent thinking."),
            (0, "No think-aloud process observed."),
        ],
        "Responsiveness to Hints": [
            (8, "Immediately incorporated hints and redirected effectively."),
            (6, "Used hints after a brief lag — required at most 2 hints per problem."),
            (4, "Required 3+ hints before changing approach."),
            (0, "Did not respond to hints effectively."),
        ],
        "Technical Vocabulary": [
            (8, "Precise use of technical terms throughout — no misuse detected."),
            (6, "Good vocabulary with occasional imprecise usage."),
            (4, "Some technical terms used incorrectly."),
            (0, "Vocabulary was non-technical or frequently imprecise."),
        ],
    }

    def score(self, req: ScoringRequest) -> ScoredFeedback:
        technical_cats = self._score_technical(req.technical)
        technical_avg  = sum(c.avg for c in technical_cats) / len(technical_cats)

        presence_dims  = self._score_presence(req.analysis.presence)
        presence_avg   = sum(d.score for d in presence_dims) / max(len(presence_dims), 1)

        speech_dim     = self._score_speech(req.analysis.speech)
        env_dim        = self._score_environment(req.analysis.environment)

        # Composite: technical avg is 0–10, others are 0–100
        overall = (
            (technical_avg / 10 * 100) * self.TECHNICAL_WEIGHT +
            presence_avg               * self.PRESENCE_WEIGHT  +
            speech_dim.score           * self.SPEECH_WEIGHT    +
            env_dim.overall_score      * self.ENVIRONMENT_WEIGHT
        )

        strengths, gaps = self._extract_strengths_gaps(
            technical_cats, presence_dims, speech_dim, req.analysis.environment
        )

        hire = self._compute_hire_probability(
            technical_avg, presence_avg, speech_dim.score,
            req.analysis.environment.overall_score, gaps
        )

        action_plan = self._build_action_plan(technical_cats, presence_dims, speech_dim, req.analysis.environment)

        return ScoredFeedback(
            session_id=req.session_id,
            overall_score=round(overall, 1),
            technical_categories=technical_cats,
            presence_dimensions=presence_dims,
            speech_dimension=speech_dim,
            environment_dimension=env_dim,
            strengths=strengths,
            gaps=gaps,
            hire_probability=hire,
            action_plan=action_plan,
        )

    def score_video_telemetry_session(
        self,
        *,
        session_id: str,
        analysis: FullAnalysisReport,
        timeline: list[dict[str, Any]],
        session_duration_minutes: float,
        feedback_overall_score_pct: Optional[float] = None,
    ) -> VideoTelemetryScoreResult:
        """
        Presence + speech + environment only (technical rubric omitted).
        Overall score reweights the three pillars to sum to 100%.
        """
        _ = session_duration_minutes  # reserved for future calibration
        w_sum = self.PRESENCE_WEIGHT + self.SPEECH_WEIGHT + self.ENVIRONMENT_WEIGHT
        presence_dims = self._score_presence(analysis.presence)
        presence_avg = sum(d.score for d in presence_dims) / max(len(presence_dims), 1)
        speech_dim = self._score_speech(analysis.speech)
        env_dim = self._score_environment(analysis.environment)

        overall = (
            presence_avg * (self.PRESENCE_WEIGHT / w_sum)
            + speech_dim.score * (self.SPEECH_WEIGHT / w_sum)
            + env_dim.overall_score * (self.ENVIRONMENT_WEIGHT / w_sum)
        )

        strengths, gaps = self._extract_strengths_gaps(
            [], presence_dims, speech_dim, analysis.environment
        )
        hire = self._compute_hire_probability_soft_only(
            presence_avg,
            speech_dim.score,
            analysis.environment.overall_score,
            gaps,
            feedback_overall_score_pct=feedback_overall_score_pct,
        )
        action_plan = self._build_action_plan_video_telemetry(
            presence_dims, speech_dim, analysis.environment
        )

        return VideoTelemetryScoreResult(
            session_id=session_id,
            overall_score=round(overall, 1),
            timeline=timeline,
            presence_dimensions=presence_dims,
            speech_dimension=speech_dim,
            environment_dimension=env_dim,
            strengths=strengths,
            gaps=gaps,
            hire_probability=hire,
            action_plan=action_plan,
        )

    # ── Technical scoring ──────────────────────────────────────────────────

    def _score_technical(self, tech: TechnicalScores) -> list[CategoryScore]:
        cats_raw = {
            "Algorithmic Problem Solving": tech.algorithmic_problem_solving.scores,
            "Code Design & Implementation": tech.code_design_implementation.scores,
            "System Thinking & Tradeoffs": tech.system_thinking_tradeoffs.scores,
            "Communication & Collaboration": tech.communication_collaboration.scores,
        }

        results = []
        for cat_name, scores_dict in cats_raw.items():
            metrics = []
            for metric_name, raw_score in scores_dict.items():
                note = self._lookup_note(metric_name, raw_score)
                metrics.append(MetricScore(
                    name=metric_name,
                    score=raw_score,
                    pct=round(raw_score * 10),
                    note=note,
                ))

            avg = sum(m.score for m in metrics) / max(len(metrics), 1)
            verdict = "strong" if avg >= 7.5 else "developing" if avg >= 5.5 else "weak"
            narrative = self._category_narrative(cat_name, avg, metrics)

            results.append(CategoryScore(
                name=cat_name,
                metrics=metrics,
                avg=round(avg, 2),
                verdict=verdict,
                narrative=narrative,
            ))

        return results

    def _lookup_note(self, metric: str, score: float) -> str:
        levels = self.METRIC_NOTES.get(metric, [])
        for threshold, note in sorted(levels, reverse=True):
            if score >= threshold:
                return note
        return levels[-1][1] if levels else "Score recorded."

    def _category_narrative(self, name: str, avg: float, metrics: list[MetricScore]) -> str:
        # Identify best and worst metric in the category
        best  = max(metrics, key=lambda m: m.score)
        worst = min(metrics, key=lambda m: m.score)
        bar   = "at FAANG bar" if avg >= 7 else "approaching FAANG bar" if avg >= 5.5 else "below FAANG bar"

        narratives = {
            "Algorithmic Problem Solving": (
                f"Your algorithmic thinking averaged {avg:.1f}/10 — {bar}. "
                f"Your strongest area was {best.name.lower()} ({best.score}/10), showing you can "
                f"{'structure problems well' if best.score >= 7 else 'make progress under pressure'}. "
                f"The priority gap is {worst.name.lower()} ({worst.score}/10) — "
                f"this is one of the first things FAANG engineers look at when calibrating a hire decision."
            ),
            "Code Design & Implementation": (
                f"Implementation quality averaged {avg:.1f}/10 — {bar}. "
                f"{best.name} was a relative strength at {best.score}/10. "
                f"Focus on {worst.name.lower()} ({worst.score}/10) in your next sessions — "
                f"clean, correct code is table stakes at this level."
            ),
            "System Thinking & Tradeoffs": (
                f"Systems thinking averaged {avg:.1f}/10 — this category has the widest variance between "
                f"undergrad candidates and experienced engineers. "
                f"Your {worst.name.lower()} ({worst.score}/10) is the specific gap to close — "
                f"practise the '3 alternatives' rule: always name three approaches before implementing."
            ),
            "Communication & Collaboration": (
                f"Communication averaged {avg:.1f}/10 — {bar}. "
                f"This is the category most correlated with interview pass rate beyond technical ability. "
                f"Your {best.name.lower()} ({best.score}/10) is a genuine asset. "
                f"The area to improve is {worst.name.lower()} ({worst.score}/10)."
            ),
        }
        return narratives.get(name, f"Average score: {avg:.1f}/10.")

    # ── Presence scoring ───────────────────────────────────────────────────

    def _score_presence(self, presence: PresenceReport) -> list[PresenceDimension]:
        dims = []

        # Eye contact
        eye_score = presence.scores.eye_contact
        eye_narrative = self._eye_narrative(
            presence.gaze.camera_percent,
            presence.gaze.down_percent,
            presence.gaze.longest_down_gap_ms
        )
        dims.append(PresenceDimension(
            name="Eye Contact & Gaze",
            score=eye_score,
            stats={
                "camera_percent": presence.gaze.camera_percent,
                "screen_percent": presence.gaze.screen_percent,
                "down_percent":   presence.gaze.down_percent,
                "longest_gap_ms": presence.gaze.longest_down_gap_ms,
            },
            narrative=eye_narrative,
        ))

        # Posture
        posture_score = presence.scores.posture
        dims.append(PresenceDimension(
            name="Posture & Stability",
            score=posture_score,
            stats={
                "upright_percent":    presence.posture.upright_percent,
                "forward_lean_pct":   presence.posture.forward_lean_percent,
                "slouch_events":      presence.posture.slouch_events,
                "avg_spine_angle":    presence.posture.avg_spine_angle,
            },
            narrative=self._posture_narrative(presence.posture),
        ))

        # Composure
        composure_score = presence.scores.composure
        dims.append(PresenceDimension(
            name="Facial Composure",
            score=composure_score,
            stats={
                "stress_events": presence.stress.stress_events,
                "avg_brow_raise": presence.stress.avg_brow_raise,
                "avg_lip_compression": presence.stress.avg_lip_compression,
            },
            narrative=self._composure_narrative(presence.stress, presence.scores.composure),
        ))

        # Gestures / fidget
        dims.append(PresenceDimension(
            name="Gestures & Fidgeting",
            score=presence.scores.gestures,
            stats={
                "face_touch_count": presence.fidget.face_touch_count,
                "hair_touch_count": presence.fidget.hair_touch_count,
            },
            narrative=self._gesture_narrative(presence.fidget, presence.scores.gestures),
        ))

        return dims

    def _eye_narrative(self, camera_pct: float, down_pct: float, longest_ms: float) -> str:
        longest_s = round(longest_ms / 1000)
        if camera_pct >= 65:
            return (f"Good eye contact — you maintained camera gaze {camera_pct:.0f}% of the session. "
                    f"Down-gaze was {down_pct:.0f}%, which is within acceptable range. "
                    f"The longest continuous drop was {longest_s}s — aim to keep this under 20s.")
        elif camera_pct >= 45:
            return (f"Moderate eye contact at {camera_pct:.0f}% camera gaze. "
                    f"You spent {down_pct:.0f}% of the session looking down — likely at code or notes. "
                    f"Place a sticky dot next to your camera lens as a gaze anchor and hide your self-view.")
        else:
            return (f"Eye contact was low — only {camera_pct:.0f}% camera gaze. "
                    f"Your {longest_s}s longest drop is a clear signal you default to looking at the screen. "
                    f"This reduces perceived confidence even when your answer is strong.")

    def _posture_narrative(self, p: PostureStats) -> str:
        if p.slouch_events == 0:
            return (f"Excellent posture — {p.upright_percent:.0f}% of the session was upright. "
                    f"Your forward lean ({p.forward_lean_percent:.0f}%) signals engagement effectively.")
        elif p.slouch_events <= 3:
            return (f"Mostly good posture with {p.slouch_events} slouch event(s) detected. "
                    f"These occurred after the midpoint — fatigue during long sessions is normal. "
                    f"Build a physical cue: touch your back to your chair when you catch yourself drifting.")
        else:
            return (f"{p.slouch_events} slouch events detected — posture degraded significantly after the midpoint. "
                    f"FAANG interviews can run 45–60 min; posture stamina is a real factor. "
                    f"Consider standing for part of the session or using a lumbar cushion.")

    def _composure_narrative(self, s: StressStats, score: float) -> str:
        if score >= 75:
            return ("Strong composure throughout — minimal stress markers detected. "
                    "You maintained an open, engaged expression even during difficult questions.")
        elif score >= 50:
            return (f"{s.stress_events} stress events detected (brow raise, lip compression). "
                    f"These clustered around the harder problem variants. "
                    f"Interviewers often interpret visible stress as low confidence even when the answer is heading right. "
                    f"Practise a 'neutral face' drill: code in front of a mirror for 5 minutes daily.")
        else:
            return (f"High stress markers throughout — {s.stress_events} events with avg brow raise {s.avg_brow_raise:.2f}. "
                    f"This signals anxiety that undermines your otherwise solid technical work. "
                    f"Consider mock sessions specifically focused on composure, starting with easy problems to build baseline calm.")

    def _gesture_narrative(self, f: FidgetStats, score: float) -> str:
        total = f.face_touch_count + f.hair_touch_count
        if total == 0:
            return "No fidgeting detected — your hands were calm and purposeful throughout."
        elif total <= 5:
            return (f"Mild fidgeting — {f.face_touch_count} face touches and {f.hair_touch_count} hair touches detected. "
                    f"These are common self-soothing behaviours under pressure. Awareness alone usually reduces frequency.")
        else:
            return (f"{total} fidget events detected ({f.face_touch_count} face, {f.hair_touch_count} hair). "
                    f"This is noticeable to an interviewer and signals anxiety. "
                    f"Keep your hands on the desk or in your lap during thinking pauses.")

    # ── Speech scoring ─────────────────────────────────────────────────────

    def _score_speech(self, speech: SpeechReport) -> SpeechDimension:
        st = speech.stats
        narrative = self._speech_narrative(st, speech.top_fillers)
        return SpeechDimension(
            avg_wpm=st.avg_wpm,
            fillers_per_minute=st.fillers_per_minute,
            dead_pauses=st.dead_pause_count,
            transcription_conf=st.transcription_confidence,
            top_fillers=speech.top_fillers,
            score=round(speech.scores.overall, 1),
            narrative=narrative,
        )

    def _speech_narrative(self, s: SpeechStats, top_fillers: list[TopFiller]) -> str:
        pace_verdict = (
            "ideal" if 120 <= s.avg_wpm <= 160 else
            "slightly slow — aim for 120–160 wpm" if s.avg_wpm < 120 else
            "too fast — slow down for clarity"
        )
        filler_verdict = (
            "excellent" if s.fillers_per_minute < 2 else
            "acceptable" if s.fillers_per_minute < 4 else
            "high — target under 4/min"
        )
        top_word = top_fillers[0].word if top_fillers else None
        filler_note = f' Your most frequent filler is "{top_word}" — focus on eliminating it first.' if top_word else ""

        pause_note = (
            "" if s.dead_pause_count == 0 else
            f" {s.dead_pause_count} dead pause(s) over {round(s.longest_silence_ms/1000)}s detected — bridge with verbal narration."
        )

        return (
            f"Speech pace was {pace_verdict} at {s.avg_wpm} wpm. "
            f"Filler word rate is {filler_verdict} at {s.fillers_per_minute:.1f}/min.{filler_note}"
            f"{pause_note} "
            f"Audio transcription confidence was {s.transcription_confidence:.0f}% — "
            f"{'excellent clarity' if s.transcription_confidence > 88 else 'consider using earphones for clearer capture'}."
        )

    # ── Environment scoring ────────────────────────────────────────────────

    def _score_environment(self, env: EnvironmentReport) -> EnvironmentDimension:
        items = [
            {"label": "Lighting",         "score": env.lighting.score,    "verdict": env.lighting.verdict,    "note": env.lighting.issue   or "Good"},
            {"label": "Camera angle",     "score": env.camera.score,      "verdict": env.camera.verdict,      "note": env.camera.issue     or "Good"},
            {"label": "Background",       "score": env.background.score,  "verdict": env.background.verdict,  "note": env.background.issue or "Good"},
            {"label": "Background noise", "score": env.audio.score,       "verdict": env.audio.verdict,       "note": env.audio.issue      or "Good"},
            {
                "label": "Microphone",
                "score": env.microphone.score,
                "verdict": env.microphone.verdict,
                "note": _microphone_dimension_note(env.microphone),
                "deviceKind": env.microphone.device_kind
                if env.microphone.has_client_snapshot
                else None,
            },
            {
                "label": "Attire",
                "score": env.attire.score,
                "verdict": env.attire.verdict,
                "note": _attire_dimension_note(env.attire),
            },
        ]
        return EnvironmentDimension(
            items=items,
            overall_score=env.overall_score,
            critical_issues=env.critical_issues,
        )

    # ── Strengths & gaps ───────────────────────────────────────────────────

    def _extract_strengths_gaps(
        self,
        technical_cats: list[CategoryScore],
        presence_dims: list[PresenceDimension],
        speech: SpeechDimension,
        env: EnvironmentReport,
    ) -> tuple[list[StrengthItem], list[GapItem]]:
        strengths: list[StrengthItem] = []
        gaps: list[GapItem] = []

        # Technical
        for cat in technical_cats:
            for m in cat.metrics:
                if m.score >= 8:
                    strengths.append(StrengthItem(title=m.name, detail=m.note, source="technical"))
                elif m.score <= 4:
                    impact = "high" if cat.name in ("Algorithmic Problem Solving", "Code Design & Implementation") else "medium"
                    gaps.append(GapItem(title=m.name, detail=m.note, impact=impact, source="technical"))

        # Presence
        for dim in presence_dims:
            if dim.score >= 75:
                strengths.append(StrengthItem(title=dim.name, detail=dim.narrative[:120] + "…", source="presence"))
            elif dim.score < 50:
                gaps.append(GapItem(title=dim.name, detail=dim.narrative[:120] + "…", impact="medium", source="presence"))

        # Speech
        if speech.fillers_per_minute < 2:
            strengths.append(StrengthItem(title="Low filler word rate", detail=f"{speech.fillers_per_minute:.1f}/min — well below average.", source="speech"))
        elif speech.fillers_per_minute > 6:
            gaps.append(GapItem(title="High filler word rate", detail=f"{speech.fillers_per_minute:.1f}/min — target under 4.", impact="high", source="speech"))

        if speech.dead_pauses > 1:
            gaps.append(GapItem(
                title="Dead silence gaps",
                detail=f"{speech.dead_pauses} gaps over 5s — narrate your thinking continuously.",
                impact="high",
                source="speech",
            ))

        # Sort: high impact gaps first
        impact_order = {"high": 0, "medium": 1, "low": 2}
        gaps.sort(key=lambda g: impact_order.get(g.impact, 2))

        return strengths[:6], gaps[:6]

    # ── Hire probability ───────────────────────────────────────────────────

    def _compute_hire_probability(
        self,
        tech_avg: float,        # 0–10
        presence_avg: float,    # 0–100
        speech_score: float,    # 0–100
        env_score: float,       # 0–100
        gaps: list[GapItem],
    ) -> HireProbability:

        # Base probability from technical score (primary signal)
        # FAANG bar ≈ 7.5/10 for a pass — sigmoid centred there
        tech_pct = tech_avg / 10
        base_prob = 1 / (1 + math.exp(-8 * (tech_pct - 0.75))) * 100

        # Adjustments from soft signals
        presence_adj = (presence_avg - 60) * 0.15   # ±9 pts for presence
        speech_adj   = (speech_score   - 60) * 0.10  # ±6 pts for speech
        env_adj      = (env_score      - 60) * 0.04  # ±2.4 pts for env

        raw_prob = base_prob + presence_adj + speech_adj + env_adj
        probability = max(5, min(95, raw_prob))

        # Impact items: what specific fixes would move the needle
        impact_items = []
        high_gaps = [g for g in gaps if g.impact == "high"]
        for g in high_gaps[:3]:
            delta = 8 if g.source == "technical" else 4
            impact_items.append(HireImpactItem(
                label=f"Fix: {g.title}",
                probability_delta=delta,
                action=g.detail[:80],
            ))

        # Verdict sentence
        if probability >= 70:
            verdict = "Strong candidate — approaching FAANG bar."
        elif probability >= 50:
            verdict = "On the borderline — one or two improvements away from a hire."
        elif probability >= 30:
            verdict = "Solid foundation, but below FAANG bar at current performance."
        else:
            verdict = "Significant gaps remain — focused preparation needed before reapplying."

        narrative = self._hire_narrative(probability, tech_avg, high_gaps)

        return HireProbability(
            probability=round(probability, 1),
            verdict=verdict,
            narrative=narrative,
            breakdown={
                "technical":    round(tech_avg / 10 * 100, 1),
                "presence":     round(presence_avg, 1),
                "speech":       round(speech_score, 1),
                "environment":  round(env_score, 1),
            },
            impact_items=impact_items,
        )

    def _compute_hire_probability_soft_only(
        self,
        presence_avg: float,
        speech_score: float,
        env_score: float,
        gaps: list[GapItem],
        feedback_overall_score_pct: Optional[float] = None,
    ) -> HireProbability:
        """Hire-style score from presence, speech, and environment only (no technical rubric)."""
        blend = presence_avg * 0.45 + speech_score * 0.35 + env_score * 0.20
        raw_prob = 28.0 + (blend - 50.0) * 0.85
        p_video = max(5.0, min(94.0, raw_prob))

        if feedback_overall_score_pct is not None:
            fb = max(0.0, min(100.0, float(feedback_overall_score_pct)))
            if fb <= 0.0:
                probability = 0.0
            else:
                probability = round(0.45 * p_video + 0.55 * fb, 1)
                probability = max(0.0, min(94.0, probability))
        else:
            probability = p_video

        impact_items: list[HireImpactItem] = []
        high_gaps = [g for g in gaps if g.impact == "high"]
        for g in high_gaps[:3]:
            impact_items.append(
                HireImpactItem(
                    label=f"Fix: {g.title}",
                    probability_delta=4.0,
                    action=g.detail[:80],
                )
            )

        if feedback_overall_score_pct is not None and float(feedback_overall_score_pct) <= 0.0:
            verdict = "Overall interview score from the feedback model is 0% — hire probability is set to 0."
            narrative = (
                "The structured feedback rubric returned a 0% overall score, so hire probability is 0. "
                "Focus on the action plan and metric-level feedback before the next attempt."
            )
        elif probability >= 70:
            verdict = "Strong on-camera presence and delivery for this session."
            narrative = self._hire_narrative_soft(probability, high_gaps)
        elif probability >= 50:
            verdict = "Mixed signals — a few targeted improvements would sharpen the impression."
            narrative = self._hire_narrative_soft(probability, high_gaps)
        elif probability >= 35:
            verdict = "Several soft-skill gaps visible on the recording."
            narrative = self._hire_narrative_soft(probability, high_gaps)
        else:
            verdict = "Significant room to improve environment, presence, and speech clarity."
            narrative = self._hire_narrative_soft(probability, high_gaps)

        breakdown: dict[str, float] = {
            "presence": round(presence_avg, 1),
            "speech": round(speech_score, 1),
            "environment": round(env_score, 1),
        }
        if feedback_overall_score_pct is not None:
            breakdown["feedbackOverall"] = round(
                max(0.0, min(100.0, float(feedback_overall_score_pct))), 1
            )
        if feedback_overall_score_pct is not None:
            breakdown["videoTelemetryOnly"] = round(p_video, 1)

        return HireProbability(
            probability=round(probability, 1),
            verdict=verdict,
            narrative=narrative,
            breakdown=breakdown,
            impact_items=impact_items,
        )

    def _hire_narrative_soft(self, prob: float, high_gaps: list[GapItem]) -> str:
        gap_names = ", ".join(g.title.lower() for g in high_gaps[:2])
        if prob >= 65:
            return (
                f"At {prob:.0f}% (video-only model), delivery and setup look competitive. "
                f"Polish items: {gap_names or 'minor consistency'}."
            )
        if prob >= 45:
            return (
                f"At {prob:.0f}%, focus next on {gap_names or 'eye contact, audio clarity, and background stability'}."
            )
        return (
            f"At {prob:.0f}%, prioritize environment and on-camera habits before the next session "
            f"({gap_names or 'see action plan'})."
        )

    def _hire_narrative(self, prob: float, tech_avg: float, high_gaps: list[GapItem]) -> str:
        gap_names = ", ".join(g.title.lower() for g in high_gaps[:2])
        if prob >= 65:
            return (f"At {prob:.0f}% hire probability, you're genuinely in the conversation for an offer. "
                    f"Your technical average of {tech_avg:.1f}/10 shows real algorithmic ability. "
                    f"The remaining gaps ({gap_names or 'minor polish items'}) are the difference between a strong and borderline evaluation.")
        elif prob >= 40:
            return (f"At {prob:.0f}% hire probability, you have the foundation but the current performance level would "
                    f"likely result in a 'No Hire' from a senior engineer. "
                    f"The primary blockers are {gap_names or 'complexity analysis and system thinking'}. "
                    f"These are learnable — targeted prep over 3–4 weeks could realistically push you to 65%+.")
        else:
            return (f"At {prob:.0f}%, there are fundamental gaps in {gap_names or 'core technical skills'} "
                    f"that need to be addressed before a real FAANG application. "
                    f"Focus the next 4–6 weeks on the action plan below rather than applying. "
                    f"A second mock in 3 weeks would give a clearer picture of progress.")

    # ── Action plan ────────────────────────────────────────────────────────

    def _build_action_plan(
        self,
        technical_cats: list[CategoryScore],
        presence_dims: list[PresenceDimension],
        speech: SpeechDimension,
        env: EnvironmentReport,
    ) -> list[ActionItem]:
        items: list[ActionItem] = []
        rank = 1

        # Critical environment issues first (easy wins)
        for issue in env.critical_issues[:2]:
            items.append(ActionItem(
                rank=rank, title="Fix your environment setup", detail=issue,
                urgency="from_today", category="environment",
            ))
            rank += 1

        # Worst technical category
        worst_cat = min(technical_cats, key=lambda c: c.avg)
        worst_metric = min(worst_cat.metrics, key=lambda m: m.score)
        items.append(ActionItem(
            rank=rank,
            title=f"Drill {worst_metric.name.lower()} — 30 min/day for 2 weeks",
            detail=worst_metric.note,
            urgency="from_today",
            category="technical",
        ))
        rank += 1

        # Speech — filler words
        if speech.fillers_per_minute > 4:
            top = speech.top_fillers[0].word if speech.top_fillers else "um"
            items.append(ActionItem(
                rank=rank,
                title=f'Cut filler words — start with "{top}"',
                detail=f"Record yourself for 2 minutes answering a question. Count fillers. Target under 4/min. The habit builds in 10 days.",
                urgency="this_week",
                category="speech",
            ))
            rank += 1

        # Worst presence dimension
        worst_presence = min(presence_dims, key=lambda d: d.score)
        if worst_presence.score < 70:
            items.append(ActionItem(
                rank=rank,
                title=f"Improve {worst_presence.name.lower()}",
                detail=worst_presence.narrative[:200],
                urgency="this_week",
                category="presence",
            ))
            rank += 1

        # System thinking (common weakness)
        sys_cat = next((c for c in technical_cats if "System" in c.name), None)
        if sys_cat and sys_cat.avg < 6:
            items.append(ActionItem(
                rank=rank,
                title="Practise the '3-alternatives' rule for every problem",
                detail="Before implementing, name three approaches (brute force → better → optimal) and compare trade-offs. Do this for your next 20 practice sessions.",
                urgency="this_week",
                category="technical",
            ))
            rank += 1

        # Long-term: DP / Graph (most common FAANG gap)
        items.append(ActionItem(
            rank=rank,
            title="Complete Blind 75 DP + Graph subset",
            detail="Dynamic programming and graph traversal are the two highest-variance areas in FAANG coding rounds. Dedicate 2–3 weeks to this subset specifically.",
            urgency="next_few_weeks",
            category="technical",
        ))

        return items[:7]  # cap at 7 action items

    def _build_action_plan_video_telemetry(
        self,
        presence_dims: list[PresenceDimension],
        speech: SpeechDimension,
        env: EnvironmentReport,
    ) -> list[ActionItem]:
        """Action items without technical-drill steps (video telemetry only)."""
        items: list[ActionItem] = []
        rank = 1

        for issue in env.critical_issues[:2]:
            items.append(
                ActionItem(
                    rank=rank,
                    title="Fix your environment setup",
                    detail=issue,
                    urgency="from_today",
                    category="environment",
                )
            )
            rank += 1

        if speech.fillers_per_minute > 4:
            top = speech.top_fillers[0].word if speech.top_fillers else "um"
            items.append(
                ActionItem(
                    rank=rank,
                    title=f'Cut filler words — start with "{top}"',
                    detail="Record a 2-minute answer, count fillers, target under 4/min.",
                    urgency="this_week",
                    category="speech",
                )
            )
            rank += 1

        if speech.dead_pauses > 1:
            items.append(
                ActionItem(
                    rank=rank,
                    title="Reduce long silences",
                    detail=f"{speech.dead_pauses} dead pause(s) — narrate thinking out loud between steps.",
                    urgency="this_week",
                    category="speech",
                )
            )
            rank += 1

        worst_presence = min(presence_dims, key=lambda d: d.score)
        if worst_presence.score < 70:
            items.append(
                ActionItem(
                    rank=rank,
                    title=f"Improve {worst_presence.name.lower()}",
                    detail=worst_presence.narrative[:200],
                    urgency="this_week",
                    category="presence",
                )
            )
            rank += 1

        items.append(
            ActionItem(
                rank=rank,
                title="Re-run a 10-minute dry run with telemetry on",
                detail="Use the same desk, mic, and lighting as interview day; review timeline deltas for touches and stress.",
                urgency="next_week",
                category="presence",
            )
        )

        return items[:7]


# ─── Redis load + aggregate client telemetry → FullAnalysisReport ───────────

def _telemetry_environment_key(session_id: str) -> str:
    return f"session:{session_id}:video_telemetry_environment"


def _telemetry_samples_key(session_id: str) -> str:
    return f"session:{session_id}:video_telemetry_samples"


def fetch_interview_telemetry_from_redis(
    redis_client: Any, session_id: str
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load one-shot environment blob (if any) and time-series samples (oldest first)."""
    env_raw = redis_client.get(_telemetry_environment_key(session_id))
    env_once: Optional[Dict[str, Any]] = None
    if env_raw:
        try:
            env_once = json.loads(env_raw)
        except json.JSONDecodeError:
            logger.warning("Invalid video_telemetry_environment JSON for session %s", session_id)

    raw_list = redis_client.lrange(_telemetry_samples_key(session_id), 0, -1)
    samples: List[Dict[str, Any]] = []
    for item in reversed(raw_list):
        try:
            samples.append(json.loads(item))
        except json.JSONDecodeError:
            logger.warning("Skipping invalid video telemetry sample for session %s", session_id)
    return env_once, samples


def _sec(s: Dict[str, Any], name: str) -> Dict[str, Any]:
    x = s.get(name)
    return x if isinstance(x, dict) else {}


def _mean(nums: List[float]) -> float:
    return sum(nums) / len(nums) if nums else 0.0


def _duration_seconds(sample: Dict[str, Any]) -> float:
    d = sample.get("duration")
    if d is None:
        return 0.0
    if isinstance(d, (int, float)):
        return float(d)
    s = str(d).strip().lower().rstrip("s")
    try:
        return float(s)
    except ValueError:
        return 0.0


def _safe_int(v: Any) -> int:
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(v)
    return 0


def _speech_curr_interval_segment_fields(
    sp: Dict[str, Any],
) -> Tuple[int, List[Any], List[Any]]:
    """
    Per-interval segment aggregates from the client ``speech`` object (camelCase or snake_case).
    """
    raw_count = sp.get("currIntSegmentCounts")
    if raw_count is None:
        raw_count = sp.get("curr_int_segment_counts")
    if raw_count is None:
        raw_count = sp.get("currIntSegmentCount")
    if raw_count is None:
        raw_count = sp.get("curr_int_segment_count")
    counts = _safe_int(raw_count) if raw_count is not None else 0

    words = sp.get("currIntSegmentWords")
    if words is None:
        words = sp.get("curr_int_segment_words")
    if not isinstance(words, list):
        words = []

    wpm_raw = sp.get("currIntSegmentWpm")
    if wpm_raw is None:
        wpm_raw = sp.get("curr_int_segment_wpm")
    if not isinstance(wpm_raw, list):
        wpm_raw = []

    return counts, words, wpm_raw


def _numeric_wpm_values(wpm_list: List[Any]) -> List[float]:
    out: List[float] = []
    for x in wpm_list:
        if isinstance(x, bool):
            continue
        if isinstance(x, (int, float)):
            out.append(float(x))
        else:
            try:
                out.append(float(x))
            except (TypeError, ValueError):
                continue
    return out


def _coerce_curr_segment_wpm_json(wpm_raw: List[Any]) -> List[Any]:
    """Normalize segment WPM list for JSON output (floats for numeric entries)."""
    out: List[Any] = []
    for x in wpm_raw:
        if isinstance(x, bool):
            out.append(x)
        elif isinstance(x, (int, float)):
            out.append(float(x))
        else:
            out.append(x)
    return out


def _infer_timeline_interval_issues(
    *,
    dead_delta: int,
    stress_delta: int,
    face_delta: int,
    hair_delta: int,
    brow_val: Optional[float],
    lip_val: Optional[float],
    current_wpm: Any,
    wpm_segments: List[float],
) -> List[Dict[str, Any]]:
    """
    Interval-level issues aligned with session speech/presence heuristics (pace band, pauses, stress, fidgets).
    """
    issues: List[Dict[str, Any]] = []

    wpm_candidates = [w for w in wpm_segments if w > 1e-6]
    if not wpm_candidates and isinstance(current_wpm, (int, float)) and not isinstance(
        current_wpm, bool
    ):
        wpm_candidates = [float(current_wpm)]

    if wpm_candidates:
        lo, hi = min(wpm_candidates), max(wpm_candidates)
        if hi > 160:
            issues.append(
                {
                    "category": "speech",
                    "title": "Speaking pace — fast",
                    "detail": (
                        f"Peak pace around {hi:.0f} wpm in this window "
                        f"(target ~120–160 wpm for clarity)."
                    ),
                    "impact": "medium",
                }
            )
        if lo < 120:
            issues.append(
                {
                    "category": "speech",
                    "title": "Speaking pace — slow",
                    "detail": (
                        f"Pace near {lo:.0f} wpm — aim closer to 120–160 wpm "
                        f"so answers stay engaging."
                    ),
                    "impact": "low",
                }
            )
        if len(wpm_candidates) >= 2 and (hi - lo) >= 50:
            issues.append(
                {
                    "category": "speech",
                    "title": "Uneven pace across segments",
                    "detail": (
                        "Large swing in words-per-minute between segments; "
                        "steadier delivery reads as more confident."
                    ),
                    "impact": "low",
                }
            )

    if dead_delta >= 1:
        issues.append(
            {
                "category": "speech",
                "title": "Dead silence / long pauses",
                "detail": (
                    f"{dead_delta} long pause(s) in this window — "
                    "briefly narrate your thinking to avoid extended silence."
                ),
                "impact": "high" if dead_delta >= 2 else "medium",
            }
        )

    if stress_delta >= 1:
        issues.append(
            {
                "category": "stress",
                "title": "Stress indicators",
                "detail": (
                    "Elevated stress signals in this interval — short breathing "
                    "resets often help composure."
                ),
                "impact": "medium",
            }
        )

    touch_total = face_delta + hair_delta
    if touch_total >= 1:
        issues.append(
            {
                "category": "presence",
                "title": "Face / hair touching",
                "detail": (
                    "Visible self-touching detected — resting hands on desk or lap "
                    "reduces nervous cues."
                ),
                "impact": "low" if touch_total < 3 else "medium",
            }
        )

    if brow_val is not None and brow_val >= 0.55:
        issues.append(
            {
                "category": "stress",
                "title": "Raised brow tension",
                "detail": (
                    "Raised brow can read as tension — relax forehead between phrases."
                ),
                "impact": "low",
            }
        )
    if lip_val is not None and lip_val >= 0.55:
        issues.append(
            {
                "category": "stress",
                "title": "Lip compression",
                "detail": (
                    "Lip compression often signals stress — soften jaw between sentences."
                ),
                "impact": "low",
            }
        )

    return issues


def build_telemetry_timeline(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    One entry per stored sample (e.g. each ~20s POST). Cumulative counters are differenced
    from the previous sample to get per-segment increments.
    """
    timeline: List[Dict[str, Any]] = []
    prev_face = 0
    prev_hair = 0
    prev_dead_pause = 0
    prev_stress_events = 0

    for sample in samples:
        p = _sec(sample, "presence")
        sp = _sec(sample, "speech")

        face_c = _safe_int(p.get("faceTouchCount"))
        hair_c = _safe_int(p.get("hairTouchCount"))

        face_delta = max(0, face_c - prev_face)
        hair_delta = max(0, hair_c - prev_hair)
        if face_c < prev_face:
            face_delta = face_c
        if hair_c < prev_hair:
            hair_delta = hair_c
        prev_face, prev_hair = face_c, hair_c

        dead_src = sp.get("deadPauseCount")
        if dead_src is None:
            dead_src = sp.get("deadPauses")
        dead_c = _safe_int(dead_src)
        dead_delta = max(0, dead_c - prev_dead_pause)
        if dead_c < prev_dead_pause:
            dead_delta = dead_c
        prev_dead_pause = dead_c

        sec_raw = p.get("stressEventCount")
        if isinstance(sec_raw, int):
            stress_delta = max(0, sec_raw - prev_stress_events)
            if sec_raw < prev_stress_events:
                stress_delta = sec_raw
            prev_stress_events = sec_raw
        else:
            os = p.get("overallStress")
            stress_delta = (
                1
                if isinstance(os, (int, float)) and float(os) >= 0.65
                else 0
            )

        br = p.get("browRaise")
        lc = p.get("lipCompression")
        brow_val = float(br) if isinstance(br, (int, float)) else None
        lip_val = float(lc) if isinstance(lc, (int, float)) else None

        seg_count = sp.get("segmentCount")
        tw = sp.get("totalWords")
        wpm = sp.get("currentWpm")
        if wpm is None:
            wpm = sp.get("currentWPM")
        filler_raw = sp.get("fillerCount")
        if filler_raw is None:
            filler_raw = sp.get("filler_count")
        if isinstance(filler_raw, (int, float)) and not isinstance(filler_raw, bool):
            filler_count_val: Any = _safe_int(filler_raw)
        else:
            filler_count_val = filler_raw
        if isinstance(wpm, (int, float)) and not isinstance(wpm, bool):
            wpm_val: Any = float(wpm)
        else:
            wpm_val = wpm

        curr_seg_counts, curr_seg_words, curr_seg_wpm_raw = _speech_curr_interval_segment_fields(
            sp
        )
        curr_seg_wpm_json = _coerce_curr_segment_wpm_json(curr_seg_wpm_raw)
        wpm_seg_numeric = _numeric_wpm_values(curr_seg_wpm_raw)

        interval_issues = _infer_timeline_interval_issues(
            dead_delta=dead_delta,
            stress_delta=stress_delta,
            face_delta=face_delta,
            hair_delta=hair_delta,
            brow_val=brow_val,
            lip_val=lip_val,
            current_wpm=wpm_val,
            wpm_segments=wpm_seg_numeric,
        )

        speech_out: Dict[str, Any] = dict(sp)
        speech_out.update(
            {
                "segmentCount": seg_count,
                "totalWords": tw,
                "fillerCount": filler_count_val,
                "currentWpm": wpm_val,
                "currentWPM": wpm_val,
                "deadPauses": dead_delta,
                "currIntSegmentCounts": curr_seg_counts,
                "currIntSegmentWords": curr_seg_words,
                "currIntSegmentWpm": curr_seg_wpm_json,
            }
        )

        timeline.append(
            {
                "time": sample.get("time"),
                "duration": _duration_seconds(sample),
                "speech": speech_out,
                "faceTouchDelta": face_delta,
                "hairTouchDelta": hair_delta,
                "stressEventsDelta": stress_delta,
                "browRaiseLipCompression": {
                    "browRaise": brow_val,
                    "lipCompression": lip_val,
                },
                "verdicts": {"issues": interval_issues},
            }
        )

    return timeline


def stub_technical_scores() -> TechnicalScores:
    """Neutral placeholder until real rubric scores are passed from the interview."""
    mid = RubricCategory(scores={"Session average": 5.0})
    return TechnicalScores(
        algorithmic_problem_solving=mid,
        code_design_implementation=mid,
        system_thinking_tradeoffs=mid,
        communication_collaboration=mid,
    )


def build_full_analysis_from_telemetry(
    *,
    environment_once: Optional[Dict[str, Any]],
    samples: List[Dict[str, Any]],
    session_duration_minutes: float,
) -> FullAnalysisReport:
    """
    Map Redis telemetry (time-series slices + optional one-shot environment dict)
    into the structured report expected by ScoringEngine.
    """
    n = max(len(samples), 1)
    dur_ms = max(session_duration_minutes, 0.01) * 60_000.0

    # ── Presence ─────────────────────────────────────────────────────
    cam_hits = 0
    gaze_conf: List[float] = []
    slouch = 0
    spine_angles: List[float] = []
    brow: List[float] = []
    lip: List[float] = []
    stress_spikes: List[float] = []
    face_touches = 0
    hair_touches = 0

    for s in samples:
        p = _sec(s, "presence")
        if (p.get("gazeDirection") or "").lower() == "camera":
            cam_hits += 1
        gc = p.get("gazeConfidence")
        if isinstance(gc, (int, float)):
            gaze_conf.append(float(gc) * 100.0 if gc <= 1.0 else float(gc))
        if p.get("isSlouching") is True:
            slouch += 1
        sa = p.get("spineAngleDeg")
        if isinstance(sa, (int, float)):
            spine_angles.append(float(abs(sa)))
        br = p.get("browRaise")
        if isinstance(br, (int, float)):
            brow.append(float(br))
        lc = p.get("lipCompression")
        if isinstance(lc, (int, float)):
            lip.append(float(lc))
        os = p.get("overallStress")
        if isinstance(os, (int, float)) and float(os) >= 0.65:
            stress_spikes.append(float(os))
        ft = p.get("faceTouchCount")
        if isinstance(ft, int):
            face_touches += ft
        ht = p.get("hairTouchCount")
        if isinstance(ht, int):
            hair_touches += ht

    camera_pct = 100.0 * cam_hits / n
    down_pct = max(0.0, 100.0 - camera_pct)
    screen_pct = down_pct * 0.6
    avg_conf = _mean(gaze_conf) if gaze_conf else 70.0
    longest_gap_ms = max(0.0, (100.0 - camera_pct) / 100.0 * (dur_ms / max(n, 1)))

    upright_pct = max(0.0, 100.0 - min(100.0, slouch * (100.0 / n)))
    forward_lean = 25.0
    avg_spine = _mean(spine_angles) if spine_angles else 5.0

    gaze_stats = GazeStats(
        camera_percent=camera_pct,
        screen_percent=screen_pct,
        down_percent=down_pct,
        longest_down_gap_ms=longest_gap_ms,
        avg_confidence=min(1.0, max(0.0, avg_conf / 100.0)),
    )
    posture_stats = PostureStats(
        upright_percent=upright_pct,
        forward_lean_percent=forward_lean,
        slouch_events=slouch,
        avg_spine_angle=avg_spine,
    )
    stress_stats = StressStats(
        avg_brow_raise=_mean(brow) if brow else 0.15,
        avg_lip_compression=_mean(lip) if lip else 0.2,
        stress_events=len(stress_spikes),
        stress_spike_timestamps=[],
    )
    fidget_stats = FidgetStats(
        face_touch_count=face_touches,
        hair_touch_count=hair_touches,
    )

    eye = min(100.0, camera_pct * 0.9 + avg_conf * 0.25)
    posture_score = max(0.0, 100.0 - slouch * (80.0 / n))
    composure_score = max(0.0, 100.0 - len(stress_spikes) * (40.0 / n))
    gesture_score = max(0.0, 100.0 - min(80.0, (face_touches + hair_touches) * 3.0))
    presence_overall = _mean([eye, posture_score, composure_score, gesture_score])

    presence_scores = PresenceScores(
        eye_contact=round(eye, 1),
        posture=round(posture_score, 1),
        composure=round(composure_score, 1),
        gestures=round(gesture_score, 1),
        overall=round(presence_overall, 1),
    )
    presence_report = PresenceReport(
        session_duration_ms=dur_ms,
        gaze=gaze_stats,
        posture=posture_stats,
        stress=stress_stats,
        fidget=fidget_stats,
        scores=presence_scores,
    )

    # ── Speech ───────────────────────────────────────────────────────
    total_words = 0
    speak_ms_vals: List[float] = []
    wpm_vals: List[float] = []
    filler_counts: List[int] = []
    snr_vals: List[float] = []
    nf_vals: List[float] = []

    for s in samples:
        sp = _sec(s, "speech")
        tw = sp.get("totalWords")
        if isinstance(tw, int):
            total_words = max(total_words, tw)
        stm = sp.get("speakingTimeMs")
        if isinstance(stm, (int, float)):
            speak_ms_vals.append(float(stm))
        wpm = sp.get("currentWpm")
        if isinstance(wpm, (int, float)) and float(wpm) > 0:
            wpm_vals.append(float(wpm))
        fc = sp.get("fillerCount")
        if isinstance(fc, int):
            filler_counts.append(fc)
        snr = sp.get("snrDb")
        if isinstance(snr, (int, float)):
            snr_vals.append(float(snr))
        nf = sp.get("noiseFloorDb")
        if isinstance(nf, (int, float)):
            nf_vals.append(float(nf))

    speaking_time_ms = max(speak_ms_vals) if speak_ms_vals else 0.0
    avg_wpm = _mean(wpm_vals) if wpm_vals else 0.0
    total_fillers = sum(filler_counts)
    minutes = max(session_duration_minutes, 0.01)
    fillers_per_min = total_fillers / minutes

    speech_stats = SpeechStats(
        total_words=total_words,
        speaking_time_ms=speaking_time_ms,
        avg_wpm=avg_wpm,
        fillers_per_minute=fillers_per_min,
        dead_pause_count=0,
        longest_silence_ms=0.0,
        avg_noise_floor_db=_mean(nf_vals) if nf_vals else -60.0,
        avg_snr_db=_mean(snr_vals) if snr_vals else 5.0,
        transcription_confidence=88.0,
    )
    clarity = min(100.0, max(40.0, 60.0 + _mean(snr_vals) * 2)) if snr_vals else 65.0
    pace_score = 75.0 if 100 <= avg_wpm <= 170 else 55.0 if avg_wpm > 0 else 50.0
    filler_score = max(0.0, 100.0 - fillers_per_min * 8.0)
    speech_scores = SpeechScores(
        pace=round(pace_score, 1),
        filler_density=round(filler_score, 1),
        silence_control=70.0,
        audio_clarity=round(clarity, 1),
        overall=round(_mean([pace_score, filler_score, 70.0, clarity]), 1),
    )
    speech_report = SpeechReport(
        stats=speech_stats,
        top_fillers=[],
        scores=speech_scores,
        full_transcript="",
    )

    # ── Environment (rolling slices: lighting / camera / background / audio) ─
    def _scores(section: str) -> List[float]:
        out: List[float] = []
        for s in samples:
            sc = _sec(s, section).get("score")
            if isinstance(sc, (int, float)):
                out.append(float(sc))
        return out

    def _last_issue(section: str) -> Optional[str]:
        for s in reversed(samples):
            iss = _sec(s, section).get("issue")
            if isinstance(iss, str) and iss.strip():
                return iss
        return None

    def _last_verdict(section: str, default: str = "good") -> str:
        for s in reversed(samples):
            v = _sec(s, section).get("verdict")
            if isinstance(v, str) and v.strip():
                return v
        return default

    lt_scores = _scores("lighting")
    cam_scores = _scores("camera")
    bg_scores = _scores("background")
    aud_scores = _scores("audio")

    lt = _mean(lt_scores) if lt_scores else 75.0
    cs = _mean(cam_scores) if cam_scores else 80.0
    bs = _mean(bg_scores) if bg_scores else 75.0
    aus = _mean(aud_scores) if aud_scores else 75.0

    face_b = []
    for s in samples:
        fb = _sec(s, "lighting").get("faceBrightness")
        if isinstance(fb, (int, float)):
            face_b.append(float(fb))
    backlight_any = any(_sec(s, "lighting").get("backlightDetected") is True for s in samples)

    cam_angles: List[float] = []
    above_eye_any = False
    for s in samples:
        c = _sec(s, "camera")
        ad = c.get("estimatedAngleDeg")
        if isinstance(ad, (int, float)):
            cam_angles.append(float(ad))
        if c.get("isAboveEyeLevel") is True:
            above_eye_any = True

    edge_d: List[float] = []
    motion_high = False
    for s in samples:
        b = _sec(s, "background")
        ed = b.get("edgeDensity")
        if isinstance(ed, (int, float)):
            edge_d.append(float(ed))
        ms = b.get("motionScore")
        if isinstance(ms, (int, float)) and float(ms) > 0.35:
            motion_high = True

    nf_audio: List[float] = []
    noise_any = False
    echo_any = False
    ext_events = 0
    for s in samples:
        a = _sec(s, "audio")
        nd = a.get("noiseFloorDb")
        if isinstance(nd, (int, float)):
            nf_audio.append(float(nd))
        if a.get("hasBackgroundNoise") is True:
            noise_any = True
        if a.get("echoDetected") is True:
            echo_any = True
        ec = a.get("externalEventCount")
        if isinstance(ec, int):
            ext_events += ec

    microphone_env = _build_microphone_env(environment_once, samples)

    attire_raw = _extract_attire_dict(environment_once, samples)
    attire_env = (
        _attire_from_client_dict(attire_raw)
        if attire_raw is not None
        else _default_attire_env()
    )
    env_dim_scores = [lt, cs, bs, aus]
    if microphone_env.has_client_snapshot:
        env_dim_scores.append(microphone_env.score)
    if attire_raw is not None:
        env_dim_scores.append(attire_env.score)

    env_report = EnvironmentReport(
        lighting=LightingEnv(
            score=round(lt, 1),
            verdict=_last_verdict("lighting"),
            issue=_last_issue("lighting"),
            backlight_detected=backlight_any,
            face_brightness=_mean(face_b) if face_b else 100.0,
        ),
        camera=CameraEnv(
            score=round(cs, 1),
            verdict=_last_verdict("camera"),
            issue=_last_issue("camera"),
            estimated_angle_deg=_mean(cam_angles) if cam_angles else 0.0,
            is_above_eye_level=above_eye_any,
        ),
        background=BackgroundEnv(
            score=round(bs, 1),
            verdict=_last_verdict("background"),
            issue=_last_issue("background"),
            edge_density=_mean(edge_d) if edge_d else 0.2,
            motion_detected=motion_high,
        ),
        audio=AudioEnv(
            score=round(aus, 1),
            verdict=_last_verdict("audio"),
            issue=_last_issue("audio"),
            noise_floor_db=_mean(nf_audio) if nf_audio else -65.0,
            has_background_noise=noise_any,
            echo_detected=echo_any,
            external_event_count=ext_events,
        ),
        microphone=microphone_env,
        attire=attire_env,
        overall_score=round(_mean(env_dim_scores), 1),
        critical_issues=[],
        suggestions=[],
    )

    crit: List[str] = []
    for s in samples:
        ci = s.get("criticalIssues")
        if isinstance(ci, list):
            for x in ci:
                if isinstance(x, str) and x.strip() and x not in crit:
                    crit.append(x)
    if environment_once:
        ei = environment_once.get("issue")
        if isinstance(ei, str) and ei.strip() and ei not in crit:
            crit.append(ei)
        sug = environment_once.get("suggestion")
        if isinstance(sug, str) and sug.strip() and sug not in env_report.suggestions:
            env_report.suggestions.append(sug)
        att = environment_once.get("attire")
        if isinstance(att, dict):
            ai = att.get("issue")
            if isinstance(ai, str) and ai.strip() and ai not in crit:
                crit.append(ai.strip())
            asug = att.get("suggestion")
            if (
                isinstance(asug, str)
                and asug.strip()
                and asug.strip() not in env_report.suggestions
            ):
                env_report.suggestions.append(asug.strip())
        mic_e = environment_once.get("microphone")
        if isinstance(mic_e, dict):
            mi = mic_e.get("issue")
            if isinstance(mi, str) and mi.strip() and mi.strip() not in crit:
                crit.append(mi.strip())
            msug = mic_e.get("suggestion")
            if (
                isinstance(msug, str)
                and msug.strip()
                and msug.strip() not in env_report.suggestions
            ):
                env_report.suggestions.append(msug.strip())
    for s in samples:
        su = s.get("suggestions")
        if isinstance(su, list):
            for x in su:
                if isinstance(x, str) and x.strip() and x not in env_report.suggestions:
                    env_report.suggestions.append(x)
    env_report.critical_issues = crit[:20]

    overall_samples = [
        float(s["overallScore"])
        for s in samples
        if isinstance(s.get("overallScore"), (int, float))
    ]
    composite = round(_mean(overall_samples), 1) if overall_samples else env_report.overall_score

    return FullAnalysisReport(
        presence=presence_report,
        speech=speech_report,
        environment=env_report,
        composite_score=composite,
    )


def _merge_action_plan_with_llm_extensions(
    base_plan: list[dict[str, Any]],
    llm_items: list[dict[str, Any]],
    *,
    max_total: int = 12,
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    rank = 1
    seen = {str(x.get("title", "")).lower().strip() for x in base_plan if x.get("title")}
    for item in base_plan:
        row = dict(item)
        row["rank"] = rank
        merged.append(row)
        rank += 1
    for it in llm_items:
        t = str(it.get("title", "")).lower().strip()
        if not t or t in seen:
            continue
        seen.add(t)
        merged.append(
            {
                "rank": rank,
                "title": it.get("title", ""),
                "detail": it.get("detail", ""),
                "urgency": it.get("urgency", "this_week"),
                "category": it.get("category", "presence"),
            }
        )
        rank += 1
        if len(merged) >= max_total:
            break
    return merged


def compute_video_telemetry_score_payload(
    redis_client: Any,
    session_id: str,
    session_duration_minutes: float,
    *,
    feedback_overall_score_pct: Optional[float] = None,
    candidate_transcript: Optional[str] = None,
    google_api_key: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Build the same JSON as logged at interview end (``VideoTelemetryScoreResult``), plus:

    - ``fillerWords``: top filler tokens from ``candidate_transcript`` (interviewee only).
    - ``action_plan``: rule-based items plus LLM extensions (when ``google_api_key`` set).
    - Hire probability blends in ``feedback_overall_score_pct`` (0–100); 0% feedback forces 0% hire.
    """
    try:
        from services.filler_words import top_filler_words_from_transcript
        from services.telemetry_action_plan_llm import extend_video_telemetry_action_plan_llm

        env_once, samples = fetch_interview_telemetry_from_redis(redis_client, session_id)
        if not samples and env_once is None:
            return None
        dur = max(float(session_duration_minutes or 0), 0.01)
        analysis = build_full_analysis_from_telemetry(
            environment_once=env_once,
            samples=samples,
            session_duration_minutes=dur,
        )
        timeline = build_telemetry_timeline(samples)
        result = ScoringEngine().score_video_telemetry_session(
            session_id=session_id,
            analysis=analysis,
            timeline=timeline,
            session_duration_minutes=dur,
            feedback_overall_score_pct=feedback_overall_score_pct,
        )
        out = result.model_dump(mode="json")
        out["fillerWords"] = top_filler_words_from_transcript(
            candidate_transcript or "", max_items=5
        )

        base_plan = list(out.get("action_plan") or [])
        gaps_list = out.get("gaps") or []
        gaps_txt = "\n".join(
            f"- {g.get('title', '')}: {g.get('detail', '')}" for g in gaps_list[:8]
        )
        soft_parts: list[str] = []
        for pd in out.get("presence_dimensions") or []:
            if isinstance(pd, dict):
                soft_parts.append(
                    f"{pd.get('name', '')}: {(pd.get('narrative') or '')[:240]}"
                )
        sd = out.get("speech_dimension")
        if isinstance(sd, dict):
            soft_parts.append(f"Speech: {(sd.get('narrative') or '')[:360]}")
        env_dim = out.get("environment_dimension")
        if isinstance(env_dim, dict):
            for it in env_dim.get("items") or []:
                if isinstance(it, dict):
                    soft_parts.append(
                        f"{it.get('label', '')}: {it.get('note', '')}"
                    )
        soft_summary = "\n".join(soft_parts)[:4500]

        llm_extra: list[dict[str, Any]] = []
        if google_api_key:
            llm_extra = extend_video_telemetry_action_plan_llm(
                existing_plan=base_plan,
                gaps_text=gaps_txt,
                soft_summary=soft_summary,
                google_api_key=google_api_key,
            )
        out["action_plan"] = _merge_action_plan_with_llm_extensions(
            base_plan, llm_extra, max_total=12
        )
        return out
    except Exception:
        logger.exception(
            "compute_video_telemetry_score_payload failed for session=%s", session_id
        )
        return None


def log_telemetry_scoring_at_session_end(
    redis_client: Any,
    session_id: str,
    session_duration_minutes: float,
) -> None:
    """
    After interview end: load Redis telemetry, run ScoringEngine, log full result.
    Never raises (failures are logged).
    """
    payload = compute_video_telemetry_score_payload(
        redis_client, session_id, session_duration_minutes
    )
    if not payload:
        logger.info(
            "telemetry_scoring: no stored telemetry for session=%s; skipping engine log",
            session_id,
        )
        return
    logger.info(
        "telemetry_scoring: session=%s overall_score=%s video_telemetry_result=%s",
        session_id,
        payload.get("overall_score"),
        json.dumps(payload, default=str),
    )