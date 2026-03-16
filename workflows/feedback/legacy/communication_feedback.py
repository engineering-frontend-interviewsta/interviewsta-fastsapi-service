"""
Communication interview feedback workflow.
Speaking + Comprehension skills, interaction log feedback, strengths and improvements.
"""
from langgraph.graph import StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict, List, Literal

from workflows.utils import get_llm


class SpeakingSkills(BaseModel):
    """Evaluate speaking skills in a Communication Interview."""
    fluency: int = Field(...)
    pronunciation: int = Field(...)
    vocabulary_range: int = Field(...)
    sentence_construction: int = Field(...)


class ComprehensionSkills(BaseModel):
    """Evaluate comprehension skills in a Communication Interview."""
    listening_comprehension: int = Field(...)
    reading_comprehension: int = Field(...)
    contextual_understanding: int = Field(...)
    response_relevance: int = Field(...)


class CommunicationChatLogsFeedback(BaseModel):
    answer_status: List[Literal['cross-question answer', 'correct answer', 'incorrect answer', 'partially-correct answer']] = Field(...)
    comment: List[str] = Field(...)


class CommunicationStrengthsAndAreasOfImprovements(BaseModel):
    strength1: str = Field(...)
    strength2: str = Field(...)
    strength3: str = Field(...)
    areas_of_improvements1: str = Field(...)
    areas_of_improvements2: str = Field(...)
    areas_of_improvements3: str = Field(...)


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
