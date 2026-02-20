"""
Pydantic schemas for interview operations
"""
from pydantic import BaseModel, Field
from typing import Optional, Literal, Dict, Any
from datetime import datetime


class InterviewStartRequest(BaseModel):
    """Request to start an interview session"""
    interview_type: Literal["Technical", "HR", "Company", "Subject", "CaseStudy", "Communication", "Role-Based Interview"]
    session_id: str
    user_id: str
    payload: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_schema_extra = {
            "example": {
                "interview_type": "Technical",
                "session_id": "uuid-123",
                "user_id": "firebase_uid",
                "payload": {
                    "resume": "Experienced developer...",
                    "TechnicalResearch": "..."
                }
            }
        }


class InterviewStartResponse(BaseModel):
    """Response after starting an interview"""
    task_id: str
    session_id: str
    status: str = "queued"
    message: Optional[str] = None


class UserResponseRequest(BaseModel):
    """
    User's response to interview question.
    At least one of audio_data or text_response is required.
    - audio_data: for speaking phases (transcribed).
    - text_response: for text-only phases (e.g. Communication comprehension/writing); can be sent alone.
    """
    audio_data: Optional[str] = None  # base64 encoded; omit for text-only (e.g. comprehension)
    text_response: Optional[str] = None  # plain text; can be the only field for writing phases
    code_input: Optional[str] = None
    video_quality_data: Optional[Dict[str, Any]] = None
    
    class Config:
        json_schema_extra = {
            "examples": [
                {"audio_data": "base64_encoded_wav_data...", "code_input": "def solution(): pass"},
                {"text_response": "The customer should contact support with order ID and request a replacement."},
            ]
        }


class UserResponseSubmitResponse(BaseModel):
    """Response after submitting user response"""
    task_id: str
    session_id: str
    status: str = "processing"


class InterviewStatusResponse(BaseModel):
    """Current status of interview session"""
    session_id: str
    status: Literal["waiting_for_response", "processing", "ai_responded", "completed", "error"]
    message: Optional[str] = None
    audio: Optional[str] = None  # base64 encoded MP3 (deprecated, use ai_response)
    last_node: Optional[str] = None
    transcript: Optional[str] = None
    error: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    is_complete: Optional[bool] = None
    ai_response: Optional[Dict[str, Any]] = None  # Structured AI response with audio_base64


class VideoQualityData(BaseModel):
    """Video quality and behavioral metrics"""
    face: str = "ok"
    gaze: Optional[float] = None
    confidence: Optional[float] = None
    nervousness: Optional[float] = None
    engagement: Optional[float] = None
    distraction: Optional[float] = None

class RespondTaskStatusResponse(BaseModel):
    """Status of the respond task (process_user_response) for polling."""
    task_id: str
    session_id: str
    status: Literal["queued", "processing", "completed", "failed"]
    result: Optional[Dict[str, Any]] = None  # When completed: message, last_node, status
    error: Optional[str] = None
    # Expanded snapshot so clients can avoid SSE and separate status calls:
    # Mirrors what /{session_id}/status and the SSE stream expose.
    interview_status: Optional[
        Literal["waiting_for_response", "processing", "ai_responded", "completed", "error"]
    ] = None
    interview_ai_response: Optional[Dict[str, Any]] = None  # Same shape as InterviewStatusResponse.ai_response
    interview_transcript: Optional[str] = None
    interview_is_complete: Optional[bool] = None
    interview_warning: Optional[Dict[str, Any]] = None  # Video quality / termination warnings if any

class InterviewStartStatusResponse(BaseModel):
    """Status of the start interview task (process_interview_start) for polling with progress."""
    task_id: str
    session_id: Optional[str] = None  # Set once task completes or session is created
    status: Literal["queued", "processing", "completed", "failed"]
    progress: int = 0  # 0-100
    message: Optional[str] = None  # e.g. "Workflow compiled", "Getting first response"
    result: Optional[Dict[str, Any]] = None  # When completed: message, last_node, status
    error: Optional[str] = None
    # Expanded snapshot (same as respond-status) when session_id is available
    interview_status: Optional[
        Literal["waiting_for_response", "processing", "ai_responded", "completed", "error"]
    ] = None
    interview_ai_response: Optional[Dict[str, Any]] = None
    interview_transcript: Optional[str] = None
    interview_is_complete: Optional[bool] = None
    interview_warning: Optional[Dict[str, Any]] = None



# Optional: if you want to type big5 keys (O,C,E,A,N or long names)
# big5_features can be Dict[str, float] or Dict[str, Any]

class VideoTelemetryData(BaseModel):
    """Video telemetry data (soft skills + optional Big5)."""
    face: str = "ok"
    gaze: Optional[float] = None
    confidence: Optional[float] = None
    nervousness: Optional[float] = None
    engagement: Optional[float] = None
    distraction: Optional[float] = None
    big5_features: Optional[Dict[str, Any]] = None

