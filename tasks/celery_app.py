"""
Celery application configuration
"""
from celery import Celery
from kombu import Exchange, Queue
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

# Get Redis URL from environment
# If CELERY URLs not set, fall back to REDIS_URL
REDIS_URL = os.getenv("REDIS_URL", "redis://default:XDyW9potKonGk0X6swKV7hzUFbP93LMc@redis-12217.crce263.ap-south-1-1.ec2.cloud.redislabs.com:12217")
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL") or REDIS_URL
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND") or REDIS_URL

# Create Celery app
celery_app = Celery(
    "interview_service",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=[
        "tasks.interview_tasks",
        "tasks.audio_tasks",
        "tasks.resume_tasks",
        "tasks.feedback_tasks",
    ]
)

# Celery configuration
celery_app.conf.update(
    # Task settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    
    # Task execution
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_time_limit=600,  # 10 minutes hard limit
    task_soft_time_limit=540,  # 9 minutes soft limit
    
    # Result backend
    result_expires=3600,  # 1 hour
    result_extended=True,
    
    # Worker settings
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
    
    # Queue routing
    task_routes={
        "tasks.interview_tasks.*": {"queue": "interview"},
        "tasks.audio_tasks.*": {"queue": "audio"},
        "tasks.resume_tasks.*": {"queue": "resume"},
        "tasks.feedback_tasks.*": {"queue": "feedback"},
    },
    
    # Task priority
    task_default_priority=5,
    
    # Retry settings
    task_autoretry_for=(Exception,),
    task_retry_kwargs={"max_retries": 3},
    task_retry_backoff=True,
    task_retry_backoff_max=600,
    task_retry_jitter=True,
)

# Define queues with priorities
default_exchange = Exchange("default", type="direct")

celery_app.conf.task_queues = (
    Queue("interview", exchange=default_exchange, routing_key="interview", priority=10),
    Queue("audio", exchange=default_exchange, routing_key="audio", priority=8),
    Queue("resume", exchange=default_exchange, routing_key="resume", priority=5),
    Queue("feedback", exchange=default_exchange, routing_key="feedback", priority=5),
    Queue("default", exchange=default_exchange, routing_key="default", priority=1),
)


# Celery beat schedule (for periodic tasks if needed)
celery_app.conf.beat_schedule = {
    "cleanup-expired-sessions": {
        "task": "tasks.interview_tasks.cleanup_expired_sessions",
        "schedule": 3600.0,  # Every hour
    },
}


if __name__ == "__main__":
    celery_app.start()
