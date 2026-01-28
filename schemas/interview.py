"""
Pydantic schemas for interview operations
"""
from pydantic import BaseModel, Field
from typing import Optional, Literal, Dict, Any
from datetime import datetime


class InterviewStartRequest(BaseModel):
    """Request to start an interview session"""
    interview_type: Literal["Technical", "HR", "Company", "Subject", "CaseStudy"]
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
    """User's response to interview question"""
    audio_data: Optional[str] = None  # base64 encoded
    text_response: Optional[str] = None
    code_input: Optional[str] = None
    video_quality_data: Optional[Dict[str, Any]] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "audio_data": "base64_encoded_wav_data...",
                "code_input": "def solution(): pass"
            }
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
    audio: Optional[str] = None  # base64 encoded MP3
    last_node: Optional[str] = None
    transcript: Optional[str] = None
    error: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class VideoQualityData(BaseModel):
    """Video quality and behavioral metrics"""
    face: str = "ok"
    gaze: Optional[float] = None
    confidence: Optional[float] = None
    nervousness: Optional[float] = None
    engagement: Optional[float] = None
    distraction: Optional[float] = None
