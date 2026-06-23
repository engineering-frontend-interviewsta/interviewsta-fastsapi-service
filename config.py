"""
Configuration settings for FastAPI CommByAI Service
"""
from pydantic_settings import BaseSettings
from functools import lru_cache
import os


class Settings(BaseSettings):
    """Application settings"""

    SERVICE_NAME: str = "CommByAI Service"
    VERSION: str = "1.0.0"
    DEBUG: bool = True

    HOST: str = "0.0.0.0"
    PORT: int = 8000

    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")

    JWT_SECRET: str = os.getenv("JWT_SECRET", "change-me")

    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    CARTESIA_API_KEY: str = os.getenv("CARTESIA_API_KEY", "")
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")

    CORS_ORIGINS: list = [
        "http://localhost:3000",
        "http://localhost:5174",
        os.getenv("CORS_ORIGINS", "")
    ]

    SESSION_EXPIRE_SECONDS: int = 3600
    MAX_AUDIO_SIZE_MB: int = 10

    CARTESIA_MODEL: str = os.getenv("CARTESIA_MODEL", "ink-whisper")
    CARTESIA_API_VERSION: str = os.getenv("CARTESIA_API_VERSION", "2025-04-16")

    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    AWS_REGION: str = os.getenv("AWS_REGION", "ap-south-1")
    AWS_POLLY_VOICE_ID: str = os.getenv("AWS_POLLY_VOICE_ID", "Joanna")
    AWS_POLLY_ENGINE: str = os.getenv("AWS_POLLY_ENGINE", "neural")
    AWS_POLLY_SPEECH_RATE: str = os.getenv("AWS_POLLY_SPEECH_RATE", "85%")

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
