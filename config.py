"""
Configuration settings for FastAPI Interview Service
"""
from pydantic_settings import BaseSettings
from functools import lru_cache
from urllib.parse import quote_plus
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
    
    # Database (SQLAlchemy / existing code)
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://localhost/interviewsta")

    # Prisma Client Python — use PRISMA_DATABASE_URL, or build from DB_* when DB_NAME is set
    PRISMA_DATABASE_URL: str = ""
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_USERNAME: str = "arryuannkhanna"
    DB_PASSWORD: str = ""
    DB_NAME: str = "my_new_db"

    def get_prisma_database_url(self) -> str:
        """Connection string for Prisma (postgresql)."""
        if self.PRISMA_DATABASE_URL and self.PRISMA_DATABASE_URL.strip():
            return self.PRISMA_DATABASE_URL.strip()
        if self.DB_NAME and self.DB_NAME.strip():
            user = self.DB_USERNAME or ""
            pw = self.DB_PASSWORD or ""
            if pw:
                auth = f"{quote_plus(user)}:{quote_plus(pw)}@"
            elif user:
                auth = f"{quote_plus(user)}@"
            else:
                auth = ""
            return f"postgresql://{auth}{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME.strip()}"
        return self.DATABASE_URL
    
    # API Keys
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    CARTESIA_API_KEY: str = os.getenv("CARTESIA_API_KEY", "")
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
    
    # Cartesia (STT only)
    CARTESIA_MODEL: str = os.getenv("CARTESIA_MODEL", "ink-whisper")
    CARTESIA_API_VERSION: str = os.getenv("CARTESIA_API_VERSION", "2025-04-16")
    
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
