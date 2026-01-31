"""
Celery tasks for audio processing (STT/TTS)
"""
from celery import Task
from tasks.celery_app import celery_app
from services.audio_processor import AudioProcessor
import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class AudioTask(Task):
    """Base task with shared AudioProcessor (Cartesia STT + AWS Polly TTS)"""
    _audio_processor = None
    
    @property
    def audio_processor(self):
        if self._audio_processor is None:
            # Cartesia for STT
            cartesia_api_key = os.getenv("CARTESIA_API_KEY", "")
            cartesia_model = os.getenv("CARTESIA_MODEL", "ink-whisper")
            
            # AWS Polly for TTS
            aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID", "")
            aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY", "")
            aws_region = os.getenv("AWS_REGION", "us-east-1")
            polly_voice_id = os.getenv("AWS_POLLY_VOICE_ID", "Joanna")
            polly_engine = os.getenv("AWS_POLLY_ENGINE", "neural")
            polly_speech_rate = os.getenv("AWS_POLLY_SPEECH_RATE", "75%")
            
            self._audio_processor = AudioProcessor(
                cartesia_api_key=cartesia_api_key,
                aws_access_key_id=aws_access_key_id or None,
                aws_secret_access_key=aws_secret_access_key or None,
                aws_region=aws_region,
                polly_voice_id=polly_voice_id,
                polly_engine=polly_engine,
                polly_speech_rate=polly_speech_rate,
                cartesia_model=cartesia_model
            )
        return self._audio_processor


@celery_app.task(bind=True, base=AudioTask, name="tasks.audio_tasks.transcribe_audio")
def transcribe_audio(self, audio_base64: str) -> Dict[str, Any]:
    """
    Transcribe audio to text using Cartesia ink-whisper STT
    
    Args:
        audio_base64: Base64 encoded audio (WAV format)
        
    Returns:
        dict: Result with transcription
    """
    try:
        logger.info("Starting audio transcription")
        
        transcription = self.audio_processor.transcribe_audio(audio_base64)
        
        if not transcription:
            return {
                "status": "error",
                "error": "No transcription detected",
                "transcription": ""
            }
        
        return {
            "status": "success",
            "transcription": transcription
        }
        
    except Exception as e:
        logger.error(f"Error in transcribe_audio task: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "transcription": ""
        }


@celery_app.task(bind=True, base=AudioTask, name="tasks.audio_tasks.synthesize_speech")
def synthesize_speech(self, text: str, voice_id: Optional[str] = None, speed: Optional[str] = None) -> Dict[str, Any]:
    """
    Synthesize speech from text using AWS Polly TTS
    
    Args:
        text: Text to convert to speech
        voice_id: Polly voice ID (optional, e.g., 'Joanna', 'Matthew', 'Amy')
        speed: Speech rate (optional, e.g., '75%', 'slow', 'medium', 'fast', 'x-slow', 'x-fast')
        
    Returns:
        dict: Result with base64 encoded audio
    """
    try:
        logger.info(f"Starting AWS Polly speech synthesis for text (length: {len(text)})")
        
        audio_base64 = self.audio_processor.synthesize_speech_base64(text, voice_id, speed)
        
        return {
            "status": "success",
            "audio_base64": audio_base64,
            "audio": audio_base64  # Keep both for compatibility
        }
        
    except Exception as e:
        logger.error(f"Error in synthesize_speech task: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "audio": None
        }


@celery_app.task(bind=True, base=AudioTask, name="tasks.audio_tasks.process_interview_audio")
def process_interview_audio(self, session_id: str, audio_base64: str) -> Dict[str, Any]:
    """
    Process interview audio: transcribe and prepare for workflow
    
    Args:
        session_id: Interview session ID
        audio_base64: Base64 encoded audio
        
    Returns:
        dict: Transcription result
    """
    try:
        logger.info(f"Processing audio for session {session_id}")
        
        # Transcribe
        result = transcribe_audio(audio_base64)
        
        if result["status"] == "error":
            return result
        
        # Store transcription in Redis (could be done by interview task)
        from redis import Redis
        from services.interview_session import InterviewSessionManager
        
        redis_client = Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"), decode_responses=True)
        session_manager = InterviewSessionManager(redis_client)
        session_manager.set_transcript(session_id, result["transcription"])
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing interview audio: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "transcription": ""
        }
