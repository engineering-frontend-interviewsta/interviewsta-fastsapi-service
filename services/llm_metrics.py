"""
LLM-based Big Five and speech summary from interview transcript.
Uses the same Google LLM as the rest of the app (get_llm) with structured output.
"""
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)


class BigFiveScores(BaseModel):
    """Big Five personality scores from transcript (0-100 per trait)."""
    openness: int = Field(..., ge=0, le=100, description="Openness to experience")
    conscientiousness: int = Field(..., ge=0, le=100, description="Conscientiousness")
    extraversion: int = Field(..., ge=0, le=100, description="Extraversion")
    agreeableness: int = Field(..., ge=0, le=100, description="Agreeableness")
    neuroticism: int = Field(..., ge=0, le=100, description="Neuroticism")


class SpeechSummaryScores(BaseModel):
    """Speech quality scores from transcript (0-100)."""
    grammar: int = Field(..., ge=0, le=100, description="Grammar and sentence correctness")
    fluency: int = Field(..., ge=0, le=100, description="Fluency and flow of speech")
    fillers: int = Field(..., ge=0, le=100, description="Low filler usage = high score")
    clarity: int = Field(..., ge=0, le=100, description="Clarity and coherence")


BIG5_PROMPT = """You are an expert at inferring personality from spoken or written text.
Based ONLY on the following interview transcript (the INTERVIEWEE's answers), estimate the Big Five personality traits.
Output a score 0-100 for each trait. Important: differentiate between traits based on the content — do not return the same value for all five. Use the full range (e.g. 20-80) where supported by the transcript. Only use mid-range (40-60) when the transcript is too short to infer.

Transcript of the candidate's responses:
---
{transcript}
---

Return the five scores: openness, conscientiousness, extraversion, agreeableness, neuroticism (each 0-100)."""


SPEECH_SUMMARY_PROMPT = """You are an expert at assessing spoken communication from written transcripts.
Based on the following interview transcript (INTERVIEWEE's answers only), score their speech quality on four dimensions (0-100 each):
- grammar: correctness of grammar and sentence structure
- fluency: flow and ease of expression
- fillers: low use of fillers (um, uh, like) = high score; frequent fillers = low score
- clarity: clarity and coherence of ideas

Important: Differentiate the four scores based on evidence in the transcript. Do not return the same value for all. Use the full range where appropriate (e.g. strong grammar but more fillers → grammar high, fillers lower).

Transcript:
---
{transcript}
---

Return the four scores as integers 0-100."""


def get_big5_from_transcript_llm(transcript: str, google_api_key: str) -> Optional[Dict[str, Any]]:
    """
    Infer Big Five personality scores from candidate transcript using LLM.
    Returns dict with keys O, C, E, A, N (0-100) or None on failure.
    """
    if not (transcript or "").strip() or not google_api_key:
        return None
    try:
        from workflows.utils import get_llm
        llm = get_llm(google_api_key=google_api_key, temperature=0.2)
        structured_llm = llm.with_structured_output(BigFiveScores)
        result = structured_llm.invoke(BIG5_PROMPT.format(transcript=transcript[:12000]))
        out = {
            "O": result.openness,
            "C": result.conscientiousness,
            "E": result.extraversion,
            "A": result.agreeableness,
            "N": result.neuroticism,
        }
        logger.info(f"[llm_metrics] Big5 from transcript: O={out['O']} C={out['C']} E={out['E']} A={out['A']} N={out['N']}")
        return out
    except Exception as e:
        logger.warning(f"[llm_metrics] Big5 from transcript failed: {e}", exc_info=True)
        return None


def get_speech_summary_from_transcript_llm(transcript: str, google_api_key: str) -> Optional[Dict[str, Any]]:
    """
    Infer speech summary (grammar, fluency, fillers, clarity) from candidate transcript using LLM.
    Returns dict with keys grammar, fluency, fillers, clarity (0-100) or None on failure.
    """
    if not (transcript or "").strip() or not google_api_key:
        return None
    try:
        from workflows.utils import get_llm
        llm = get_llm(google_api_key=google_api_key, temperature=0.2)
        structured_llm = llm.with_structured_output(SpeechSummaryScores)
        result = structured_llm.invoke(SPEECH_SUMMARY_PROMPT.format(transcript=transcript[:12000]))
        out = {
            "grammar": result.grammar,
            "fluency": result.fluency,
            "fillers": result.fillers,
            "clarity": result.clarity,
        }
        logger.info(f"[llm_metrics] Speech summary from transcript: grammar={out['grammar']} fluency={out['fluency']} fillers={out['fillers']} clarity={out['clarity']}")
        return out
    except Exception as e:
        logger.warning(f"[llm_metrics] Speech summary from transcript failed: {e}", exc_info=True)
        return None


def get_candidate_transcript_from_messages(messages) -> str:
    """
    Build a single transcript string from LangChain messages (human = candidate answers only).
    """
    if not messages:
        return ""
    parts = []
    for msg in messages:
        if getattr(msg, "type", None) == "human":
            content = getattr(msg, "content", "") or ""
            if isinstance(content, str) and content.strip():
                parts.append(content.strip())
    return "\n\n".join(parts) if parts else ""