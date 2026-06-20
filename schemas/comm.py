"""
CommByAI Pydantic schemas for request/response validation.
"""
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class UserSegment(str, Enum):
    YOUNG_LEARNER = "YoungLearner"
    MIDDLE_SCHOOL = "MiddleSchool"
    HIGH_SCHOOL = "HighSchool"
    COLLEGE = "College"
    WORKING_PROFESSIONAL = "WorkingProfessional"


class SkillDomain(str, Enum):
    GRAMMAR = "Grammar"
    VOCABULARY = "Vocabulary"
    SENTENCE_CONSTRUCTION = "SentenceConstruction"
    WRITING = "Writing"
    SPEAKING = "Speaking"
    SCENARIOS = "Scenarios"
    READING = "Reading"


class PlacementLevel(str, Enum):
    BEGINNER = "Beginner"
    INTERMEDIATE = "Intermediate"
    ADVANCED = "Advanced"


class DifficultyTier(int, Enum):
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5


class ActivityType(str, Enum):
    MCQ = "mcq"
    FILL_BLANK = "fill_blank"
    SENTENCE_REORDER = "sentence_reorder"
    WORD_LEARN = "word_learn"
    READING = "reading"
    WRITING_PROMPT = "writing_prompt"
    SPEAKING = "speaking"
    SCENARIO = "scenario"


# --- Request Models ---

class GenerateActivityRequest(BaseModel):
    user_segment: UserSegment
    placement_level: PlacementLevel
    skill_domain: SkillDomain
    difficulty_tier: DifficultyTier
    activity_type: ActivityType
    unit_context: str = Field(..., description="Topic/theme for this unit")


class OnboardingAnswer(BaseModel):
    domain: SkillDomain
    question: str
    user_answer: str
    correct_answer: str
    is_correct: bool


class OnboardingAssessRequest(BaseModel):
    user_segment: UserSegment
    answers: list[OnboardingAnswer]


class EvaluateWritingRequest(BaseModel):
    user_segment: UserSegment
    placement_level: PlacementLevel
    prompt_text: str
    user_response: str
    min_words: int = 50
    scenario_context: Optional[str] = None


class EvaluateSpeakingRequest(BaseModel):
    user_segment: UserSegment
    placement_level: PlacementLevel
    task_description: str
    transcription: str


class ScenarioTurnRequest(BaseModel):
    user_segment: UserSegment
    scenario_role: str
    scenario_description: str
    history: list[dict]  # list of {role: "user"|"ai", message: str}
    user_message: str
    turn_number: int
    total_turns: int


class ScenarioEndRequest(BaseModel):
    user_segment: UserSegment
    scenario_role: str
    scenario_description: str
    full_history: list[dict]


class XPAwardRequest(BaseModel):
    user_id: str
    activity_type: ActivityType
    difficulty_tier: DifficultyTier
    score: int = Field(..., ge=0, le=100)


# --- Response Models ---

class FeedbackError(BaseModel):
    original: str
    corrected: str
    explanation: str
    error_type: str


class WordCorrection(BaseModel):
    word: str
    suggestion: str
    explanation: str


class FeedbackReport(BaseModel):
    overall_score: int = Field(..., ge=0, le=100)
    grammar_score: Optional[int] = None
    clarity_score: Optional[int] = None
    tone_score: Optional[int] = None
    fluency_score: Optional[int] = None
    pronunciation_score: Optional[int] = None
    confidence_score: Optional[int] = None
    professionalism_score: Optional[int] = None
    errors: list[FeedbackError] = []
    word_corrections: list[WordCorrection] = []
    improvement_suggestions: list[str] = []
    model_answer: Optional[str] = None
    transcription: Optional[str] = None
    encouragement: str = ""
    key_strengths: list[str] = []
    areas_to_improve: list[str] = []


class PlacementResult(BaseModel):
    placement_levels: dict[str, PlacementLevel]  # domain → level
    initial_difficulty_tiers: dict[str, int]  # domain → 1-5


class ScenarioTurnResponse(BaseModel):
    ai_response: str
    is_complete: bool


class XPAwardResponse(BaseModel):
    xp_earned: int
    xp_total: int
    level: int
    leveled_up: bool


class BadgeAward(BaseModel):
    badge_id: str
    badge_name: str
    badge_description: str
    category: str


class BadgeCheckResponse(BaseModel):
    new_badges: list[BadgeAward] = []


class SkillTreeNodeResponse(BaseModel):
    unit_id: str
    title: str
    description: str
    domain: SkillDomain
    level: int = Field(..., ge=1, le=5)
    prerequisites: list[str] = []
    activity_count: int
    xp_reward: int
    icon: str
    status: str = "locked"  # locked | available | completed


class UnitActivityResponse(BaseModel):
    activity_id: str
    type: ActivityType
    domain: SkillDomain
    difficulty_tier: int
    xp_value: int
    instructions: str
    completed: bool = False


class DashboardResponse(BaseModel):
    level: int
    xp_total: int
    xp_to_next_level: int
    streak: int
    longest_streak: int
    daily_goal_xp: int
    daily_xp_earned: int
    badges: list[dict] = []
    domain_proficiency: dict[str, int] = {}  # domain → 0-100
    xp_history: list[dict] = []  # last 30 days [{date, xp}]


class LeaderboardEntry(BaseModel):
    rank: int
    user_id: str
    display_name: str
    avatar: Optional[str] = None
    weekly_xp: int


class LeaderboardResponse(BaseModel):
    entries: list[LeaderboardEntry] = []
    user_rank: Optional[int] = None
    user_weekly_xp: Optional[int] = None
