"""
Pydantic schemas for resume analysis
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class ResumeAnalysisRequest(BaseModel):
    """Request for resume analysis"""
    user_id: str
    session_id: Optional[str] = None
    # Files will be handled separately as UploadFile


class ResumeAnalysisResponse(BaseModel):
    """Response after submitting resume for analysis"""
    task_id: str
    status: str = "queued"
    message: Optional[str] = "Resume analysis queued"


class ResumeAnalysisStatusResponse(BaseModel):
    """Status of resume analysis task"""
    task_id: str
    status: str  # queued, processing, completed, failed
    progress: int = 0  # 0-100
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class SectionAnalysisResult(BaseModel):
    """Section analysis scores"""
    job_match_score: int
    format_and_structure: int
    content_quality: int
    length_and_conciseness: int
    keywords_optimization: int


class KeywordAnalysisResult(BaseModel):
    """Keyword analysis"""
    found_keywords: List[str]
    not_found_keywords: List[str]
    top_3_keywords: List[str]


class JobAlignmentResult(BaseModel):
    """Job alignment scores"""
    required_skills: int
    preferred_skills: int
    experience: int
    education: int
    insights: List[str]


class StrengthsAndImprovementsResult(BaseModel):
    """Strengths and areas of improvement"""
    candidate_strengths: List[str]
    candidates_areas_of_improvements: List[str]


class CompletedResumeAnalysis(BaseModel):
    """Complete resume analysis result"""
    company: str
    role: str
    section_analysis: SectionAnalysisResult
    keyword_analysis: KeywordAnalysisResult
    job_alignment: JobAlignmentResult
    strengths_and_improvements: StrengthsAndImprovementsResult
