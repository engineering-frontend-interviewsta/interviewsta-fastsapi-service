from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.messages import HumanMessage, AIMessage,BaseMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel,Field
import operator
from typing_extensions import TypedDict, List, Dict, Any, Optional
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
import requests
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import InMemorySaver
from typing import Annotated,Literal,Tuple
from workflows.utils import get_llm
from typing_extensions import TypedDict
import time
import getpass
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod


class AnalyticalSkills(BaseModel):
    """Evaluate analytical skills in a Case Study Interview."""
    problem_understanding: int = Field(...)
    hypothesis: int = Field(...)
    analysis: int = Field(...)
    synthesis: int = Field(...)

class BusinessImpactSkills(BaseModel):
    """Evaluate business impact skills in a Case Study Interview."""
    business_judgment: int = Field(...)
    creativity: int = Field(...)
    decision_making: int = Field(...)
    impact_orientation: int = Field(...)

class CaseStudyChatLogsFeedback(BaseModel):
    answer_status: List[Literal['cross-question answer','correct answer','incorrect answer','partially-correct answer']] = Field()
    comment: List[str] = Field()

class CaseStudyStrengthsAndAreasOfImprovements(BaseModel):
    strength1: str = Field(...)
    strength2: str = Field(...)
    strength3: str = Field(...)
    areas_of_improvements1: str = Field(...)
    areas_of_improvements2: str = Field(...)
    areas_of_improvements3: str = Field(...)


class CaseStudyIntState(TypedDict):
    history_log: str
    analytical: AnalyticalSkills
    business_impact: BusinessImpactSkills
    interaction_log_feedback: CaseStudyChatLogsFeedback
    strengths_and_areas_of_improvements: CaseStudyStrengthsAndAreasOfImprovements


def analytical_llm_Node(analytical_llm):
    def _Node(state:CaseStudyIntState) -> CaseStudyIntState:
        response = analytical_llm.invoke(state["history_log"])
        print(response)
        state["analytical"] = response
        return state
    return _Node

def business_impact_llm_Node(business_impact_llm):
    def _Node(state:CaseStudyIntState) -> CaseStudyIntState:
        response = business_impact_llm.invoke(state["history_log"])
        print(response)
        state["business_impact"] = response
        return state
    return _Node

def chat_logs_feedback_Node(chat_logs_feedback_llm):
    def _Node(state:CaseStudyIntState) -> CaseStudyIntState:
        response = chat_logs_feedback_llm.invoke(state["history_log"])
        print(response)
        state["interaction_log_feedback"] = response
        return state
    return _Node

def strengths_and_areas_of_improvements_llm_Node(strengths_and_areas_of_improvements_llm):
    def _Node(state:CaseStudyIntState) -> CaseStudyIntState:
        response = strengths_and_areas_of_improvements_llm.invoke(state["history_log"])
        print(response)
        state["strengths_and_areas_of_improvements"] = response
        return state
    return _Node


def build_case_study_feedback_graph(google_api_key:str):
    llm = get_llm(google_api_key)

    analytical_llm = llm.with_structured_output(AnalyticalSkills)
    business_impact_llm = llm.with_structured_output(BusinessImpactSkills)
    chat_logs_feedback_llm = llm.with_structured_output(CaseStudyChatLogsFeedback)
    strengths_and_areas_of_improvements_llm = llm.with_structured_output(CaseStudyStrengthsAndAreasOfImprovements)

    graph = StateGraph(CaseStudyIntState)
    graph.add_node("analytical", analytical_llm_Node(analytical_llm))
    graph.add_node("business_impact", business_impact_llm_Node(business_impact_llm))
    graph.add_node("chat_logs_feedback", chat_logs_feedback_Node(chat_logs_feedback_llm))   
    graph.add_node("strengths_and_areas_of_improvements", strengths_and_areas_of_improvements_llm_Node(strengths_and_areas_of_improvements_llm))

    graph.add_edge("analytical", "business_impact")
    graph.add_edge("business_impact", "chat_logs_feedback")
    graph.add_edge("chat_logs_feedback", "strengths_and_areas_of_improvements")
    graph.add_edge("strengths_and_areas_of_improvements", "__end__")

    graph.set_entry_point("analytical")
    agent = graph.compile()
    return agent
