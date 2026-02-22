"""
Communication interview feedback workflow.
Speaking + Comprehension skills, interaction log feedback, strengths and improvements.
"""
from langgraph.graph import StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict, List, Literal

from ..utils import get_llm


class SpeakingSkills(BaseModel):
    """
    Evaluate speaking skills in a Communication Interview based on the interaction history.
    
    Score each skill on a precise 0-100 scale. Use granular values (e.g., 67, 73, 82) NOT rounded multiples of 10.
    
    Performance Bands:
    - 0: Insufficient data (fewer than 3 substantive responses) or major offense/misconduct
    - 1-35: Poor - Significant gaps, wrong approaches, or lack of basic understanding
    - 36-50: Below Average - Some understanding but major weaknesses
    - 51-60: Average - Adequate fundamentals, meets basic expectations
    - 61-70: Good - Solid knowledge and consistent performance
    - 71-80: Very Good - Strong mastery with minor areas for improvement
    - 81-90: Excellent - Expert-level understanding and application
    - 91-100: Outstanding - Exceptional mastery, flawless execution
    
    CRITICAL: Assign specific scores within ranges (e.g., 67, 73, 82) based on nuanced performance.
    Do NOT use only multiples of 10 (10, 20, 30, etc.). Be precise and granular.
    
    Before assigning any score above 0, ensure:
    1. The human has provided at least 3 meaningful responses related to that skill area.
    2. There has been sufficient back-and-forth discussion to gauge their ability.
    3. The human has demonstrated (or failed to demonstrate) the specific skill through their responses.
    """
    fluency: int = Field(..., description="Flow and smoothness of speech - absence of excessive hesitations, filler words, and breaks. Score 0-100 with granular precision (e.g., 68, 74, 83). 0 if insufficient discussion.")
    pronunciation: int = Field(..., description="Clarity of pronunciation, diction, and articulation. Score 0-100 with granular precision (e.g., 72, 79, 86). 0 if insufficient discussion.")
    vocabulary_range: int = Field(..., description="Breadth, richness, and appropriateness of vocabulary used. Score 0-100 with granular precision (e.g., 65, 77, 88). 0 if insufficient discussion.")
    sentence_construction: int = Field(..., description="Grammatical correctness and quality of sentence structure. Score 0-100 with granular precision (e.g., 71, 81, 92). 0 if insufficient discussion.")


class ComprehensionSkills(BaseModel):
    """
    Evaluate comprehension skills in a Communication Interview based on the interaction history.
    
    Score each skill on a precise 0-100 scale. Use granular values (e.g., 67, 73, 82) NOT rounded multiples of 10.
    
    Performance Bands:
    - 0: Insufficient data (fewer than 3 substantive responses) or major offense/misconduct
    - 1-35: Poor - Significant gaps, wrong approaches, or lack of basic understanding
    - 36-50: Below Average - Some understanding but major weaknesses
    - 51-60: Average - Adequate fundamentals, meets basic expectations
    - 61-70: Good - Solid knowledge and consistent performance
    - 71-80: Very Good - Strong mastery with minor areas for improvement
    - 81-90: Excellent - Expert-level understanding and application
    - 91-100: Outstanding - Exceptional mastery, flawless execution
    
    CRITICAL: Assign specific scores within ranges (e.g., 67, 73, 82) based on nuanced performance.
    Do NOT use only multiples of 10 (10, 20, 30, etc.). Be precise and granular.
    
    Before assigning any score above 0, ensure:
    1. The human has provided at least 3 meaningful responses related to that skill area.
    2. There has been sufficient back-and-forth discussion to gauge their ability.
    3. The human has demonstrated (or failed to demonstrate) the specific skill through their responses.
    """
    listening_comprehension: int = Field(..., description="Ability to correctly understand spoken/written questions and prompts. Score 0-100 with granular precision (e.g., 69, 76, 84). 0 if insufficient discussion.")
    reading_comprehension: int = Field(..., description="Ability to interpret and respond to written passages or scenarios. Score 0-100 with granular precision (e.g., 64, 78, 87). 0 if insufficient discussion.")
    contextual_understanding: int = Field(..., description="Grasping implied meaning, tone, and nuance in communication. Score 0-100 with granular precision (e.g., 71, 79, 89). 0 if insufficient discussion.")
    response_relevance: int = Field(..., description="Relevance, appropriateness, and accuracy of responses given. Score 0-100 with granular precision (e.g., 66, 75, 85). 0 if insufficient discussion.")


class CommunicationChatLogsFeedback(BaseModel):
    """
    For each Q&A pair in the communication interaction, mark the status and provide a comment.
    
    CRITICAL: answer_status MUST be one of these exact values for EACH interaction:
    - "cross-question answer" - If the interaction is part of cross-questioning or follow-up
    - "correct answer" - If the interviewee communicated clearly and effectively
    - "incorrect answer" - If the interviewee's response was unclear, off-topic, or poorly communicated
    - "partially-correct answer" - If the response was adequate but had communication issues
    
    Do NOT use descriptive text. Use ONLY the exact literal values above.
    """
    answer_status: List[Literal['cross-question answer', 'correct answer', 'incorrect answer', 'partially-correct answer']] = Field(
        ..., 
        description="List of status labels for each Q&A pair. MUST use exact values: 'cross-question answer', 'correct answer', 'incorrect answer', or 'partially-correct answer'"
    )
    comment: List[str] = Field(
        ..., 
        description="List of detailed comments explaining the communication assessment for each Q&A pair"
    )


class CommunicationStrengthsAndAreasOfImprovements(BaseModel):
    """
    Based on the interaction history between interviewer (AI) and interviewee (human) in a Communication Interview,
    provide 3 specific strengths and 3 specific areas for improvement in their speaking and comprehension skills.
    
    Address the interviewee in second person (e.g., "You demonstrated excellent...", "Your vocabulary range...").
    Be specific and actionable.
    """
    strength1: str = Field(..., description="1 crisp, specific strength the interviewee demonstrated in speaking or comprehension skills, addressed in second person.")
    strength2: str = Field(..., description="1 crisp, specific strength the interviewee demonstrated in speaking or comprehension skills, addressed in second person.")
    strength3: str = Field(..., description="1 crisp, specific strength the interviewee demonstrated in speaking or comprehension skills, addressed in second person.")
    areas_of_improvements1: str = Field(..., description="1 crisp, actionable area for improvement in speaking or comprehension skills, addressed in second person.")
    areas_of_improvements2: str = Field(..., description="1 crisp, actionable area for improvement in speaking or comprehension skills, addressed in second person.")
    areas_of_improvements3: str = Field(..., description="1 crisp, actionable area for improvement in speaking or comprehension skills, addressed in second person.")


class CommunicationIntState(TypedDict):
    history_log: str
    speaking: SpeakingSkills
    comprehension: ComprehensionSkills
    interaction_log_feedback: CommunicationChatLogsFeedback
    strengths_and_areas_of_improvements: CommunicationStrengthsAndAreasOfImprovements


def _speaking_node(speaking_llm):
    def _node(state: CommunicationIntState) -> CommunicationIntState:
        r = speaking_llm.invoke(state["history_log"])
        state["speaking"] = r
        return state
    return _node


def _comprehension_node(comprehension_llm):
    def _node(state: CommunicationIntState) -> CommunicationIntState:
        r = comprehension_llm.invoke(state["history_log"])
        state["comprehension"] = r
        return state
    return _node


def _chat_logs_node(chat_logs_llm):
    def _node(state: CommunicationIntState) -> CommunicationIntState:
        r = chat_logs_llm.invoke(state["history_log"])
        state["interaction_log_feedback"] = r
        return state
    return _node


def _strengths_node(strengths_llm):
    def _node(state: CommunicationIntState) -> CommunicationIntState:
        r = strengths_llm.invoke(state["history_log"])
        state["strengths_and_areas_of_improvements"] = r
        return state
    return _node


def build_communication_feedback_graph(google_api_key: str):
    llm = get_llm(google_api_key)
    speaking_llm = llm.with_structured_output(SpeakingSkills)
    comprehension_llm = llm.with_structured_output(ComprehensionSkills)
    chat_logs_llm = llm.with_structured_output(CommunicationChatLogsFeedback)
    strengths_llm = llm.with_structured_output(CommunicationStrengthsAndAreasOfImprovements)

    graph = StateGraph(CommunicationIntState)
    graph.add_node("speaking_skills", _speaking_node(speaking_llm))
    graph.add_node("comprehension_skills", _comprehension_node(comprehension_llm))
    graph.add_node("chat_logs_feedback", _chat_logs_node(chat_logs_llm))
    graph.add_node("strengths_and_areas_of_improvements", _strengths_node(strengths_llm))

    graph.add_edge("speaking_skills", "comprehension_skills")
    graph.add_edge("comprehension_skills", "chat_logs_feedback")
    graph.add_edge("chat_logs_feedback", "strengths_and_areas_of_improvements")
    graph.add_edge("strengths_and_areas_of_improvements", "__end__")
    graph.set_entry_point("speaking_skills")

    return graph.compile()
