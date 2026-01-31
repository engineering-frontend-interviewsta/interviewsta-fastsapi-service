"""
Audio processing service using Cartesia ink-whisper (STT) and AWS Polly (TTS)
"""
from openai import OpenAI
import boto3
from botocore.exceptions import BotoCoreError, ClientError
import base64
import tempfile
import os
import logging
from typing import Optional
import html

logger = logging.getLogger(__name__)


class AudioProcessor:
    """Handle STT with Cartesia ink-whisper and TTS with AWS Polly"""
    
    def __init__(
        self, 
        cartesia_api_key: str,
        aws_access_key_id: str = None,
        aws_secret_access_key: str = None,
        aws_region: str = "ap-south-1",
        polly_voice_id: str = "Joanna",
        polly_engine: str = "neural",
        polly_speech_rate: str = "75%",
        cartesia_model: str = "ink-whisper"
    ):
        """
        Initialize AudioProcessor with Cartesia ink-whisper (STT) and AWS Polly (TTS)
        
        Args:
            cartesia_api_key: Cartesia API key for STT
            aws_access_key_id: AWS access key (optional, uses env/IAM role if not provided)
            aws_secret_access_key: AWS secret key (optional)
            aws_region: AWS region for Polly
            polly_voice_id: AWS Polly voice ID
            polly_engine: Polly engine ('neural' or 'standard')
            polly_speech_rate: Speech rate in SSML format (e.g., '75%', 'slow', 'medium')
            cartesia_model: Cartesia STT model (default: 'ink-whisper')
        """
        # Initialize Cartesia for STT using OpenAI SDK
        self.cartesia_client = OpenAI(
            api_key=cartesia_api_key,
            base_url="https://api.cartesia.ai"
        )
        self.cartesia_model = cartesia_model
        
        # Initialize AWS Polly for TTS
        if aws_access_key_id and aws_secret_access_key:
            self.polly_client = boto3.client(
                'polly',
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                region_name=aws_region
            )
        else:
            # Use default credentials (IAM role, env vars, or credentials file)
            self.polly_client = boto3.client('polly', region_name=aws_region)
        
        self.polly_voice_id = polly_voice_id
        self.polly_engine = polly_engine
        self.polly_speech_rate = polly_speech_rate
        
        logger.info(f"🔧 AudioProcessor initialized: STT=Cartesia/{self.cartesia_model}, TTS=AWS Polly/{polly_voice_id} (rate={polly_speech_rate})")
    
    def transcribe_audio(self, audio_base64: str) -> str:
        """
        Transcribe audio using Cartesia ink-whisper STT
        
        Args:
            audio_base64: Base64 encoded audio (WAV format)
            
        Returns:
            str: Transcribed text
        """
        temp_path = None
        try:
            # Decode base64 audio
            audio_bytes = base64.b64decode(audio_base64)
            
            # Write to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_bytes)
                temp_path = tmp.name
            
            # Check file size
            file_size = os.path.getsize(temp_path)
            logger.info(f"🎤 STT MODEL: Using Cartesia '{self.cartesia_model}' for transcription ({file_size} bytes)")
            
            if file_size == 0:
                raise ValueError("Audio file is empty")
            
            # Transcribe using Cartesia (via OpenAI SDK)
            with open(temp_path, "rb") as audio_file:
                result = self.cartesia_client.audio.transcriptions.create(
                    file=audio_file,
                    model=self.cartesia_model,
                    language="en"
                )
            
            transcription = result.text.strip()
            logger.info(f"✅ STT SUCCESS: Cartesia '{self.cartesia_model}' transcription completed: '{transcription[:50]}...'")
            
            return transcription
            
        except Exception as e:
            logger.error(f"❌ STT ERROR: Cartesia '{self.cartesia_model}' transcription failed: {e}")
            raise
        finally:
            # Cleanup temp file
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
    
    def synthesize_speech(self, text: str, voice_id: Optional[str] = None, speed: Optional[str] = None) -> bytes:
        """
        Synthesize speech from text using AWS Polly TTS
        
        Args:
            text: Text to convert to speech
            voice_id: AWS Polly voice ID (optional, uses default if not provided)
            speed: Speech rate (e.g., '75%', 'slow', 'medium', 'fast', 'x-slow', 'x-fast')
            
        Returns:
            bytes: MP3 audio data
        """
        try:
            logger.info(f"Synthesizing speech with AWS Polly (length: {len(text)})")
            
            # Truncate if too long (Polly has 3000 character limit for standard, 6000 for neural)
            max_chars = 6000 if self.polly_engine == "neural" else 3000
            if len(text) > max_chars:
                text = text[:max_chars - 3] + "..."
                logger.warning(f"Text truncated to {max_chars} characters for Polly")
            
            # Escape special characters for SSML
            text_escaped = html.escape(text)
            
            # Wrap text in SSML with speech rate control
            speech_rate = speed or self.polly_speech_rate
            ssml_text = f'<speak><prosody rate="{speech_rate}">{text_escaped}</prosody></speak>'
            
            # Determine voice
            voice = voice_id or self.polly_voice_id
            
            logger.info(f"Using Polly voice: {voice}, engine: {self.polly_engine}, rate: {speech_rate}")
            
            # Synthesize speech with AWS Polly
            response = self.polly_client.synthesize_speech(
                Text=ssml_text,
                TextType='ssml',
                OutputFormat='mp3',
                VoiceId=voice,
                Engine=self.polly_engine
            )
            
            # Read audio stream
            if "AudioStream" in response:
                audio_bytes = response["AudioStream"].read()
                logger.info(f"AWS Polly synthesis completed ({len(audio_bytes)} bytes)")
                return audio_bytes
            else:
                raise Exception("No audio stream in Polly response")
            
        except (BotoCoreError, ClientError) as error:
            logger.error(f"AWS Polly error: {error}")
            raise
        except Exception as e:
            logger.error(f"Error synthesizing speech with AWS Polly: {e}")
            raise
    
    def synthesize_speech_base64(self, text: str, voice_id: Optional[str] = None, speed: Optional[str] = None) -> str:
        """
        Synthesize speech and return as base64 string
        
        Args:
            text: Text to convert
            voice_id: Polly voice ID (optional)
            speed: Speech rate (e.g., '75%', 'slow', 'medium', 'fast')
            
        Returns:
            str: Base64 encoded MP3 audio
        """
        audio_bytes = self.synthesize_speech(text, voice_id, speed)
        return base64.b64encode(audio_bytes).decode("utf-8")
