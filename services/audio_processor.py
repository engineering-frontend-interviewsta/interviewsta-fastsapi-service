"""
Audio processing service using Cartesia ink-whisper (STT) and AWS Polly (TTS)
"""
import requests
import boto3
from botocore.exceptions import BotoCoreError, ClientError
import base64
import tempfile
import os
import logging
from typing import Optional
import html
import io 

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
        cartesia_model: str = "ink-whisper",
        cartesia_api_version: str = "2025-04-16"
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
            cartesia_model: Cartesia model name (default: 'ink-whisper')
            cartesia_api_version: Cartesia API version (default: '2025-04-16')
        """
        # Initialize Cartesia for STT
        self.cartesia_api_key = cartesia_api_key
        self.cartesia_api_url = "https://api.cartesia.ai/stt"
        self.cartesia_model = cartesia_model
        self.cartesia_api_version = cartesia_api_version
        
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
        
        logger.info(f"AudioProcessor initialized: Cartesia ink-whisper (STT, model={cartesia_model}), AWS Polly (TTS, voice={polly_voice_id}, rate={polly_speech_rate})")
    
    def transcribe_audio(self, audio_base64: str) -> str:
        """
        Transcribe audio using Cartesia ink-whisper STT
        
        Args:
            audio_base64: Base64 encoded audio (WAV format)
            
        Returns:
            str: Transcribed text
        """
        try:
            # Decode base64 audio
            audio_bytes = base64.b64decode(audio_base64)
            
            # Validate WAV header
            if not audio_bytes.startswith(b'RIFF'):
                raise ValueError("Invalid WAV file - missing RIFF header")
            
            file_size = len(audio_bytes)
            logger.info(f"Transcribing audio ({file_size} bytes)")
            
            if file_size < 44:  # Minimum WAV header size
                raise ValueError(f"Audio file too small: {file_size} bytes")
            
            # ✅ FIX: Use correct Cartesia headers (Bearer auth, not custom header)
            headers = {
                "Authorization": f"Bearer {self.cartesia_api_key}",
                "Cartesia-Version": self.cartesia_api_version
                # DO NOT set Content-Type - let requests set it automatically for multipart
            }
            
            # ✅ FIX: Use io.BytesIO for proper file-like object
            # This ensures proper multipart/form-data encoding
            audio_file = io.BytesIO(audio_bytes)
            audio_file.name = "audio.wav"  # Set filename attribute
            
            files = {
                "file": ("audio.wav", audio_file, "audio/wav")
            }
            
            data = {
                "model": self.cartesia_model,  # "ink-whisper"
                "language": "en"
            }
            
            logger.info(f"Sending to Cartesia: {self.cartesia_api_url}")
            
            # Make request to Cartesia API
            response = requests.post(
                self.cartesia_api_url,
                headers=headers,
                files=files,
                data=data,
                timeout=30
            )
            
            # Enhanced error logging
            if response.status_code != 200:
                logger.error(f"Cartesia API error (status {response.status_code})")
                logger.error(f"Response headers: {dict(response.headers)}")
                logger.error(f"Response body: {response.text[:1000]}")
                raise Exception(f"Cartesia API error ({response.status_code}): {response.text}")
            
            # Parse response
            result = response.json()
            transcription = result.get("text", "").strip()
            
            if not transcription:
                logger.warning("Empty transcription received from Cartesia")
                logger.warning(f"Full response: {result}")
            
            logger.info(f"Transcription success: '{transcription[:100]}...'")
            
            return transcription
            
        except base64.binascii.Error as e:
            logger.error(f"Invalid base64 encoding: {e}")
            raise ValueError(f"Invalid base64 audio data: {e}")
        except requests.exceptions.Timeout:
            logger.error("Cartesia API timeout after 30s")
            raise Exception("Transcription service timeout")
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error calling Cartesia: {e}")
            raise Exception(f"Network error: {str(e)}")
        except Exception as e:
            logger.error(f"Transcription error: {e}", exc_info=True)
            raise
        
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
