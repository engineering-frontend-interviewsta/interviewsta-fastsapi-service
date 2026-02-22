"""
Debate interview feedback workflow.
Argumentation + Persuasion skills, interaction log feedback, strengths and improvements.
"""
from langgraph.graph import StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict, List, Literal

from ..utils import get_llm


class ArgumentationSkills(BaseModel):
    """
    Evaluate argumentation skills in a Debate Interview based on the interaction history.
    
    Score each skill on a precise 0-100 scale. Use granular values (e.g., 67, 73, 82) NOT rounded multiples of 10.
    
    Performance Bands:
    - 0: Insufficient data (fewer than 3 substantive responses) or major offense/misconduct
    - 1-35: Poor - Significant gaps, illogical arguments, or lack of basic reasoning
    - 36-50: Below Average - Some logical structure but major weaknesses
    - 51-60: Average - Adequate argumentation, meets basic expectations
    - 61-70: Good - Solid logical reasoning and consistent arguments
    - 71-80: Very Good - Strong argumentation with minor areas for improvement
    - 81-90: Excellent - Expert-level reasoning and evidence usage
    - 91-100: Outstanding - Exceptional mastery, flawless argumentation
    
    CRITICAL: Assign specific scores within ranges (e.g., 67, 73, 82) based on nuanced performance.
    Do NOT use only multiples of 10 (10, 20, 30, etc.). Be precise and granular.
    
    Before assigning any score above 0, ensure:
    1. The human has provided at least 3 meaningful responses related to that skill area.
    2. There has been sufficient back-and-forth discussion to gauge their ability.
    3. The human has demonstrated (or failed to demonstrate) the specific skill through their responses.
    """
    argument_structure: int = Field(..., description="Logical structure, clarity, and organization of arguments. Score 0-100 with granular precision (e.g., 68, 75, 84). 0 if insufficient discussion.")
    evidence_usage: int = Field(..., description="Quality, relevance, and effective use of evidence and supporting facts. Score 0-100 with granular precision (e.g., 71, 78, 87). 0 if insufficient discussion.")
    logical_reasoning: int = Field(..., description="Soundness of reasoning, validity of inferences, and coherent logical flow. Score 0-100 with granular precision (e.g., 64, 76, 89). 0 if insufficient discussion.")
    counterargument_handling: int = Field(..., description="Ability to address, rebut, and effectively counter opposing viewpoints. Score 0-100 with granular precision (e.g., 69, 77, 85). 0 if insufficient discussion.")


class PersuasionSkills(BaseModel):
    """
    Evaluate persuasion skills in a Debate Interview based on the interaction history.
    
    Score each skill on a precise 0-100 scale. Use granular values (e.g., 67, 73, 82) NOT rounded multiples of 10.
    
    Performance Bands:
    - 0: Insufficient data (fewer than 3 substantive responses) or major offense/misconduct
    - 1-35: Poor - Significant gaps, unconvincing arguments, or lack of rhetorical skill
    - 36-50: Below Average - Some persuasive elements but major weaknesses
    - 51-60: Average - Adequate persuasion, meets basic expectations
    - 61-70: Good - Solid persuasive techniques and consistent impact
    - 71-80: Very Good - Strong persuasion with minor areas for improvement
    - 81-90: Excellent - Expert-level rhetorical skills and audience awareness
    - 91-100: Outstanding - Exceptional mastery, highly compelling arguments
    
    CRITICAL: Assign specific scores within ranges (e.g., 67, 73, 82) based on nuanced performance.
    Do NOT use only multiples of 10 (10, 20, 30, etc.). Be precise and granular.
    
    Before assigning any score above 0, ensure:
    1. The human has provided at least 3 meaningful responses related to that skill area.
    2. There has been sufficient back-and-forth discussion to gauge their ability.
    3. The human has demonstrated (or failed to demonstrate) the specific skill through their responses.
    """
    persuasiveness: int = Field(..., description="Overall convincing power and impact of arguments. Score 0-100 with granular precision (e.g., 70, 77, 86). 0 if insufficient discussion.")
    rhetorical_skills: int = Field(..., description="Effective use of ethos (credibility), pathos (emotion), and logos (logic). Score 0-100 with granular precision (e.g., 66, 74, 83). 0 if insufficient discussion.")
    audience_awareness: int = Field(..., description="Adapting arguments and language to the audience and context. Score 0-100 with granular precision (e.g., 68, 79, 88). 0 if insufficient discussion.")
    conclusion_strength: int = Field(..., description="Strength, clarity, and impact of closing arguments and conclusions. Score 0-100 with granular precision (e.g., 72, 81, 91). 0 if insufficient discussion.")


class DebateChatLogsFeedback(BaseModel):
    """
    For each Q&A pair in the debate interaction, mark the status and provide a comment.
    
    CRITICAL: answer_status MUST be one of these exact values for EACH interaction:
    - "cross-question answer" - If the interaction is part of cross-questioning or follow-up
    - "correct answer" - If the interviewee presented a strong, well-reasoned argument
    - "incorrect answer" - If the interviewee's argument was weak, illogical, or factually wrong
    - "partially-correct answer" - If the argument had merit but lacked depth or had flaws
    
    Do NOT use descriptive text like "Interviewee's initial argument". Use ONLY the exact literal values above.
    """
    answer_status: List[Literal['cross-question answer', 'correct answer', 'incorrect answer', 'partially-correct answer']] = Field(
        ..., 
        description="List of status labels for each Q&A pair. MUST use exact values: 'cross-question answer', 'correct answer', 'incorrect answer', or 'partially-correct answer'"
    )
    comment: List[str] = Field(
        ..., 
        description="List of detailed comments explaining the assessment for each Q&A pair"
    )


class DebateStrengthsAndAreasOfImprovements(BaseModel):
    """
    Based on the interaction history between interviewer (AI) and interviewee (human) in a Debate Interview,
    provide 3 specific strengths and 3 specific areas for improvement in their argumentation and persuasion skills.
    
    Address the interviewee in second person (e.g., "You presented compelling...", "Your logical reasoning...").
    Be specific and actionable.
    """
    strength1: str = Field(..., description="1 crisp, specific strength the interviewee demonstrated in argumentation or persuasion skills, addressed in second person.")
    strength2: str = Field(..., description="1 crisp, specific strength the interviewee demonstrated in argumentation or persuasion skills, addressed in second person.")
    strength3: str = Field(..., description="1 crisp, specific strength the interviewee demonstrated in argumentation or persuasion skills, addressed in second person.")
    areas_of_improvements1: str = Field(..., description="1 crisp, actionable area for improvement in argumentation or persuasion skills, addressed in second person.")
    areas_of_improvements2: str = Field(..., description="1 crisp, actionable area for improvement in argumentation or persuasion skills, addressed in second person.")
    areas_of_improvements3: str = Field(..., description="1 crisp, actionable area for improvement in argumentation or persuasion skills, addressed in second person.")


class DebateIntState(TypedDict):
    history_log: str
    argumentation: ArgumentationSkills
    persuasion: PersuasionSkills
    interaction_log_feedback: DebateChatLogsFeedback
    strengths_and_areas_of_improvements: DebateStrengthsAndAreasOfImprovements


def _argumentation_node(llm):
    def _node(state: DebateIntState) -> DebateIntState:
        r = llm.invoke(state["history_log"])
        state["argumentation"] = r
        return state
    return _node


def _persuasion_node(llm):
    def _node(state: DebateIntState) -> DebateIntState:
        r = llm.invoke(state["history_log"])
        state["persuasion"] = r
        return state
    return _node


def _chat_logs_node(llm):
    def _node(state: DebateIntState) -> DebateIntState:
        r = llm.invoke(state["history_log"])
        state["interaction_log_feedback"] = r
        return state
    return _node


def _strengths_node(llm):
    def _node(state: DebateIntState) -> DebateIntState:
        r = llm.invoke(state["history_log"])
        state["strengths_and_areas_of_improvements"] = r
        return state
    return _node


def build_debate_feedback_graph(google_api_key: str):
    llm = get_llm(google_api_key)
    argumentation_llm = llm.with_structured_output(ArgumentationSkills)
    persuasion_llm = llm.with_structured_output(PersuasionSkills)
    chat_logs_llm = llm.with_structured_output(DebateChatLogsFeedback)
    strengths_llm = llm.with_structured_output(DebateStrengthsAndAreasOfImprovements)

    graph = StateGraph(DebateIntState)
    graph.add_node("argumentation_skills", _argumentation_node(argumentation_llm))
    graph.add_node("persuasion_skills", _persuasion_node(persuasion_llm))
    graph.add_node("chat_logs_feedback", _chat_logs_node(chat_logs_llm))
    graph.add_node("strengths_and_areas_of_improvements", _strengths_node(strengths_llm))

    graph.add_edge("argumentation_skills", "persuasion_skills")
    graph.add_edge("persuasion_skills", "chat_logs_feedback")
    graph.add_edge("chat_logs_feedback", "strengths_and_areas_of_improvements")
    graph.add_edge("strengths_and_areas_of_improvements", "__end__")
    graph.set_entry_point("argumentation_skills")

    return graph.compile()
