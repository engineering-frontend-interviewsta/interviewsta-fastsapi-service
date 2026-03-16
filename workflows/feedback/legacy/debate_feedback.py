"""
Debate interview feedback workflow.
Argumentation + Persuasion skills, interaction log feedback, strengths and improvements.
"""
from langgraph.graph import StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict, List, Literal

from workflows.utils import get_llm


class ArgumentationSkills(BaseModel):
    """Evaluate argumentation skills in a Debate Interview."""
    argument_structure: int = Field(...)
    evidence_usage: int = Field(...)
    logical_reasoning: int = Field(...)
    counterargument_handling: int = Field(...)


class PersuasionSkills(BaseModel):
    """Evaluate persuasion skills in a Debate Interview."""
    persuasiveness: int = Field(...)
    rhetorical_skills: int = Field(...)
    audience_awareness: int = Field(...)
    conclusion_strength: int = Field(...)


class DebateChatLogsFeedback(BaseModel):
    answer_status: List[Literal['cross-question answer', 'correct answer', 'incorrect answer', 'partially-correct answer']] = Field(...)
    comment: List[str] = Field(...)


class DebateStrengthsAndAreasOfImprovements(BaseModel):
    strength1: str = Field(...)
    strength2: str = Field(...)
    strength3: str = Field(...)
    areas_of_improvements1: str = Field(...)
    areas_of_improvements2: str = Field(...)
    areas_of_improvements3: str = Field(...)


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
