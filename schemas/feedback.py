"""
Pydantic schemas for feedback generation
"""
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Literal


class FeedbackGenerationRequest(BaseModel):
    """Request to generate feedback for an interview"""
    session_id: str
    interview_type: Literal["Technical", "HR", "CaseStudy"]
    user_id: str


class FeedbackGenerationResponse(BaseModel):
    """Response after submitting feedback generation request"""
    task_id: str
    session_id: str
    status: str = "queued"


class FeedbackStatusResponse(BaseModel):
    """Status of feedback generation"""
    task_id: str
    session_id: str
    status: str  # queued, processing, completed, failed
    progress: int = 0
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class TechnicalFeedbackResult(BaseModel):
    """Technical interview feedback"""
    language_score: int
    framework_score: int
    algorithms_score: int
    data_structures_score: int
    approach_score: int
    optimization_score: int
    debugging_score: int
    syntax_score: int
    strengths: List[str]
    areas_of_improvements: List[str]
    interaction_log_feedback: Dict[str, Any]


class HRFeedbackResult(BaseModel):
    """HR interview feedback"""
    clarity_score: int
    confidence_score: int
    structure_score: int
    engagement_score: int
    values_score: int
    teamwork_score: int
    growth_score: int
    initiative_score: int
    strengths: List[str]
    areas_of_improvements: List[str]


class CaseStudyFeedbackResult(BaseModel):
    """Case study interview feedback"""
    problem_understanding_score: int
    hypothesis_score: int
    analysis_score: int
    synthesis_score: int
    business_judgment_score: int
    creativity_score: int
    decision_making_score: int
    impact_orientation_score: int
    strengths: List[str]
    areas_of_improvements: List[str]
