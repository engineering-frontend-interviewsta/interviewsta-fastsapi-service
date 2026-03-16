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


class CommunicationSkills(BaseModel):
    """
    Evaluate communication skills in an HR Interview based on the interaction history.
    """
    clarity: int = Field(..., description="Clarity and articulation in communication - how well ideas are expressed. Score 0-100 with granular precision (e.g., 68, 76, 84).")
    confidence: int = Field(..., description="Confidence and assertiveness in responses and self-presentation. Score 0-100 with granular precision (e.g., 72, 79, 87).")
    structure: int = Field(..., description="Organization and logical structure of responses. Score 0-100 with granular precision (e.g., 65, 77, 88).")
    engagement: int = Field(..., description="Level of engagement, enthusiasm, and active participation. Score 0-100 with granular precision (e.g., 71, 81, 92).")

class CulturalFitSkills(BaseModel):
    """
    Evaluate cultural fit skills in an HR Interview based on the interaction history.
    """
    values: int = Field(..., description="Alignment with company values and ethical standards demonstrated in responses. Score 0-100 with granular precision (e.g., 69, 77, 85).")
    teamwork: int = Field(..., description="Teamwork orientation, collaboration skills, and ability to work with others. Score 0-100 with granular precision (e.g., 64, 73, 86).")
    growth: int = Field(..., description="Growth mindset, learning orientation, and adaptability. Score 0-100 with granular precision (e.g., 71, 78, 88).")
    initiative: int = Field(..., description="Proactiveness, self-motivation, and taking ownership. Score 0-100 with granular precision (e.g., 74, 82, 91).")

Label = Literal["correct", "incorrect", "partially-correct", "cross-question"]


class FeedbackItem(BaseModel):
    status: Label = Field(..., description="Mark the interaction status")
    comment: str = Field(..., description="Add any feedbacks.")



class ChatLogsFeedback(BaseModel):
    answer_status: Literal[
        'cross-question answer', 'correct answer', 'incorrect answer', 'partially-correct answer'] = Field()
    comment: str = Field()

class HR_Strengths_and_areas_of_improvements(BaseModel):
    """Strengths and areas of improvement for HR interview."""
    strength1: str = Field(...)
    strength2: str = Field(...)
    strength3: str = Field(...)
    areas_of_improvements1: str = Field(...)
    areas_of_improvements2: str = Field(...)
    areas_of_improvements3: str = Field(...)


class HRIntState(TypedDict):
    history_log: str
    communication_skills: CommunicationSkills
    cultural_skills: CulturalFitSkills
    strengths_and_areas_of_improvements: HR_Strengths_and_areas_of_improvements
    interaction_log_feedback: ChatLogsFeedback


def cultural_skills_llm_Node(cultural_skills_llm):
    def _Node(state:HRIntState) -> HRIntState:
        response = cultural_skills_llm.invoke(state["history_log"])
        state["cultural_skills"] = response
        return state
    return _Node

def communicational_skills_llm_Node(communicational_skills_llm):
    def _Node(state:HRIntState) -> HRIntState:
        response = communicational_skills_llm.invoke(state["history_log"])
        print(response)
        state["communication_skills"] = response
        return state
    return _Node

def strengths_and_areas_of_improvements_llm_Node(strengths_and_areas_of_improvements_llm):
     def _Node(state:HRIntState) -> HRIntState:
        response = strengths_and_areas_of_improvements_llm.invoke(state["history_log"])
        print(response)
        state["strengths_and_areas_of_improvements"] = response
        return state
     return _Node



def chat_logs_feedback_Node(feedback_llm):
    def _Node(state:HRIntState) -> HRIntState:
        _history = state["history_log"]
        response = feedback_llm.invoke(_history)
        print("This is the interaction log feedback",response)
        state["interaction_log_feedback"] = response
        return state

    return _Node


def build_hr_skills_feedback_graph(google_api_key:str):
    llm = get_llm(google_api_key)

    communicational_skills_llm = llm.with_structured_output(CommunicationSkills)
    cultural_skills_llm = llm.with_structured_output(CulturalFitSkills)
    strengths_and_areas_of_improvements_llm = llm.with_structured_output(HR_Strengths_and_areas_of_improvements)
    feedback_llm = llm.with_structured_output(ChatLogsFeedback)

    graph = StateGraph(HRIntState)
    graph.add_node("communication_skills", communicational_skills_llm_Node(communicational_skills_llm))
    graph.add_node("cultural_fit", cultural_skills_llm_Node(cultural_skills_llm))
    graph.add_node("strengths_and_areas_of_improvements", strengths_and_areas_of_improvements_llm_Node(strengths_and_areas_of_improvements_llm))
    graph.add_node("chat_logs_feedback",
                   chat_logs_feedback_Node(feedback_llm))

    graph.add_edge("communication_skills", "cultural_fit")
    graph.add_edge("cultural_fit", "strengths_and_areas_of_improvements")
    graph.add_edge("strengths_and_areas_of_improvements","chat_logs_feedback")
    graph.add_edge("chat_logs_feedback", "__end__")

    graph.set_entry_point("communication_skills")
    agent = graph.compile()

    return agent
