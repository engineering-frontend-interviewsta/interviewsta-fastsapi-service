"""
Sarvam AI STT (streaming WebSocket) and TTS (REST stream MP3).
Used when session payload has use_sarvam_audio=True (dev-gated at API).
"""
from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
from typing import Any, List

import requests

logger = logging.getLogger(__name__)

SARVAM_TTS_STREAM_URL = "https://api.sarvam.ai/text-to-speech/stream"
_MAX_TTS_CHARS = 8000


def _api_key() -> str:
    key = (os.getenv("SARVAM_API_SUBSCRIPTION_KEY") or "").strip()
    if not key:
        raise ValueError("SARVAM_API_SUBSCRIPTION_KEY is not set")
    return key


def _raise_if_stt_error(message: Any) -> None:
    if getattr(message, "type", None) != "error":
        return
    data = getattr(message, "data", None)
    err = getattr(data, "error", None) if data is not None else None
    code = getattr(data, "code", "") if data is not None else ""
    raise RuntimeError(f"Sarvam STT error ({code}): {err or data}")


async def _transcribe_wav_base64_async(audio_base64: str) -> str:
    from sarvamai import AsyncSarvamAI
    from sarvamai.types.speech_to_text_transcription_data import SpeechToTextTranscriptionData

    client = AsyncSarvamAI(api_subscription_key=_api_key())
    model = os.getenv("SARVAM_STT_MODEL", "saaras:v2")
    language_code = os.getenv("SARVAM_STT_LANGUAGE_CODE", "en-IN")
    sample_rate = int(os.getenv("SARVAM_STT_SAMPLE_RATE", "16000"))

    transcripts: List[str] = []
    async with client.speech_to_text_streaming.connect(
        model=model,
        mode="transcribe",
        language_code=language_code,
        high_vad_sensitivity=True,
        sample_rate=str(sample_rate),
    ) as ws:
        await ws.transcribe(audio=audio_base64, encoding="audio/wav", sample_rate=sample_rate)
        await ws.flush()
        for _ in range(32):
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=90.0)
            except asyncio.TimeoutError:
                logger.warning("Sarvam STT recv timeout")
                break
            _raise_if_stt_error(msg)
            data = getattr(msg, "data", None)
            if isinstance(data, SpeechToTextTranscriptionData) and data.transcript:
                transcripts.append(data.transcript.strip())
    text = " ".join(transcripts).strip()
    if not text:
        raise ValueError("Empty transcription from Sarvam")
    return text


def transcribe_wav_base64(audio_base64: str) -> str:
    """Sync wrapper for Celery: WAV as base64 string → transcript text."""
    return asyncio.run(_transcribe_wav_base64_async(audio_base64))


def synthesize_speech_mp3_bytes(text: str) -> bytes:
    """TTS via Sarvam REST stream; returns MP3 bytes (same family as AWS Polly path)."""
    raw = (text or "").strip()
    if not raw:
        raise ValueError("Empty text for Sarvam TTS")
    if len(raw) > _MAX_TTS_CHARS:
        raw = raw[: _MAX_TTS_CHARS - 3] + "..."
        logger.warning("Sarvam TTS text truncated to %s chars", _MAX_TTS_CHARS)

    pace = float(os.getenv("SARVAM_TTS_PACE", "1.1"))
    sample_rate = int(os.getenv("SARVAM_TTS_SAMPLE_RATE", "22050"))

    payload = {
        "text": raw,
        "target_language_code": os.getenv("SARVAM_TTS_TARGET_LANGUAGE_CODE", "en-IN"),
        "speaker": os.getenv("SARVAM_TTS_SPEAKER", "Priya"),
        "model": os.getenv("SARVAM_TTS_MODEL", "bulbul:v2"),
        "pace": pace,
        "speech_sample_rate": sample_rate,
        "output_audio_codec": "mp3",
        "enable_preprocessing": True,
    }
    headers = {
        "api-subscription-key": _api_key(),
        "Content-Type": "application/json",
    }
    buf = io.BytesIO()
    with requests.post(
        SARVAM_TTS_STREAM_URL,
        headers=headers,
        json=payload,
        stream=True,
        timeout=120,
    ) as resp:
        resp.raise_for_status()
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                buf.write(chunk)
    data = buf.getvalue()
    if not data:
        raise RuntimeError("Sarvam TTS returned empty audio")
    return data


def synthesize_speech_mp3_base64(text: str) -> str:
    return base64.b64encode(synthesize_speech_mp3_bytes(text)).decode("utf-8")
