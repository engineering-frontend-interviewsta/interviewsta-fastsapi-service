"""
Configuration settings for FastAPI Interview Service
"""
from pydantic_settings import BaseSettings
from functools import lru_cache
import os


class Settings(BaseSettings):
    """Application settings"""
    
    # Service info
    SERVICE_NAME: str = "InterviewSta Interview Service"
    VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8001
    
    # Redis
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://localhost/interviewsta")
    
    # API Keys
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    CARTESIA_API_KEY: str = os.getenv("CARTESIA_API_KEY", "")
    CARTESIA_MODEL: str = os.getenv("CARTESIA_MODEL", "ink-whisper")
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
    
    # Firebase
    FIREBASE_CREDENTIALS_JSON: str = os.getenv("FIREBASE_CREDENTIALS_JSON", "")
    
    # Celery - Use REDIS_URL if CELERY URLs not explicitly set
    CELERY_BROKER_URL: str = os.getenv("CELERY_BROKER_URL") or os.getenv("REDIS_URL", "redis://localhost:6379")
    CELERY_RESULT_BACKEND: str = os.getenv("CELERY_RESULT_BACKEND") or os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # CORS
    CORS_ORIGINS: list = [
        "http://localhost:3000",
        "http://localhost:5173",
        "https://interviewsta.com",
        "https://*.interviewsta.com",
        "https://interviewsta-app-frontend.vercel.app"
    ]
    
    # Session settings
    SESSION_EXPIRE_SECONDS: int = 3600  # 1 hour
    MAX_AUDIO_SIZE_MB: int = 10
    
    # AWS Polly (TTS)
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    AWS_REGION: str = os.getenv("AWS_REGION", "ap-south-1")
    AWS_POLLY_VOICE_ID: str = os.getenv("AWS_POLLY_VOICE_ID", "Joanna")
    AWS_POLLY_ENGINE: str = os.getenv("AWS_POLLY_ENGINE", "neural")  # neural or standard
    AWS_POLLY_SPEECH_RATE: str = os.getenv("AWS_POLLY_SPEECH_RATE", "85%")  # 20% to 200%, default is slower
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # Ignore extra environment variables


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()
