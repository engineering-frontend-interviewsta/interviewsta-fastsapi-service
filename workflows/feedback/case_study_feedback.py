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
from ..utils import get_llm
from typing_extensions import TypedDict
import time
import getpass
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod


class AnalyticalSkills(BaseModel):
    '''
    You need to assign scores for the following analytical skills categories based on interaction history between interviewer and interviewee.

    IMPORTANT: Only provide scores above 0 if there has been sufficient conversation and human responses to make an accurate assessment.
    If there are fewer than 3 substantive human responses or insufficient discussion about a particular skill area,
    assign 0 to indicate "Insufficient Data/Not Applicable".

    Scoring Guidelines:
    0 - Insufficient conversation between interviewer and interviewee/data to make an assessment, major offense occurred, completely wrong approach, or not applicable to the role.
    10,20,30,40 - Varying degrees of poor performance: unseriousness, plainly wrong, or barely comprehensive.
    50 - Average performance with basic understanding.
    60 - Decent performance with solid fundamentals.
    70 - Good performance with strong knowledge.
    80 - Great performance with deep understanding.
    90 - Amazing performance with expert-level insights.
    100 - Flawless performance with exceptional mastery.

    Before assigning any score above 0, ensure:
    1. The human has provided at least 3 meaningful responses related to that skill area.
    2. There has been sufficient back-and-forth discussion to gauge their knowledge.
    3. The human has demonstrated (or failed to demonstrate) the specific skill through their responses.
    '''
    problem_understanding: int = Field(..., description="The understanding of the problem")
    hypothesis: int = Field(..., description="The hypothesis of the problem")
    analysis: int = Field(..., description="The analysis of the problem")
    synthesis: int = Field(..., description="The synthesis of the problem")

class BusinessImpactSkills(BaseModel):
    '''
    You need to assign scores for the following business impact skills categories based on interaction history between interviewer and interviewee.

    IMPORTANT: Only provide scores above 0 if there has been sufficient conversation and human responses to make an accurate assessment.
    If there are fewer than 3 substantive human responses or insufficient discussion about a particular skill area,
    assign 0 to indicate "Insufficient Data/Not Applicable".

    Scoring Guidelines:
    0 - Insufficient conversation between interviewer and interviewee/data to make an assessment, major offense occurred, completely wrong approach, or not applicable to the role.
    10,20,30,40 - Varying degrees of poor performance: unseriousness, plainly wrong, or barely comprehensive.
    50 - Average performance with basic understanding.
    60 - Decent performance with solid fundamentals.
    70 - Good performance with strong knowledge.
    80 - Great performance with deep understanding.
    90 - Amazing performance with expert-level insights.
    100 - Flawless performance with exceptional mastery.

    Before assigning any score above 0, ensure:
    1. The human has provided at least 3 meaningful responses related to that skill area.
    2. There has been sufficient back-and-forth discussion to gauge their knowledge.
    3. The human has demonstrated (or failed to demonstrate) the specific skill through their responses.
    '''
    business_judgment: int = Field(..., description="The judgment of the business")
    creativity: int = Field(..., description="The creativity of the business")
    decision_making: int = Field(..., description="The decision making of the business")
    impact_orientation: int = Field(..., description="The impact orientation of the business")

class CaseStudyChatLogsFeedback(BaseModel):
    '''
    For a pair of interaction, first mark their status and followed up by comments.
    For status, mark them -
    "cross-question answer" - If the interaction is part of cross-questioning
    "correct answer" -  If the interviewee has answered correctly
    "incorrect answer" - If the interviewee has answered incorrectly
    "partially-correct answer" - If the interviewee has answered only partially correct
    For comment, add the comments to tell how the answer could've been improved if it is not correct
    '''
    answer_status: List[Literal['cross-question answer','correct answer','incorrect answer','partially-correct answer']] = Field()
    comment: List[str] = Field()

class CaseStudyStrengthsAndAreasOfImprovements(BaseModel):
    '''
    You are given interaction log between interviewer(ai) and interviewee(human).
    You need to first ensure that you have enough of a history to comment.
    '''
    strength1: str = Field(...,description="1 crisp points of strengths which you think interviewee(human) has in analytical and business impact skills strictly based on the question asked and answer provided and address them in second person.")
    strength2: str = Field(...,description="1 crisp points of strengths which you think interviewee(human) has in analytical and business impact skills strictly based on the question asked and answer provided and address them in second person.")
    strength3: str = Field(...,description="1 crisp points of strengths which you think interviewee(human) has in analytical and business impact skills strictly based on the question asked and answer provided and address them in second person.")
    areas_of_improvements1: str = Field(...,description="1 crisp points of areas of improvements which you think interviewee(human) has in analytical and business impact skills strictly based on the question asked and answer provided and address them in second person.")
    areas_of_improvements2: str = Field(...,description="1 crisp points of areas of improvements which you think interviewee(human) has in analytical and business impact skills strictly based on the question asked and answer provided and address them in second person.")
    areas_of_improvements3: str = Field(...,description="1 crisp points of areas of improvements which you think interviewee(human) has in analytical and business impact skills strictly based on the question asked and answer provided and address them in second person.")

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

class CaseStudyIntState(TypedDict):
    history_log: str = Field(...,description="Has list of base messages")
    analytical: AnalyticalSkills = Field(...,description="It has analytical scoring results")
    business_impact: BusinessImpactSkills = Field(...,description="It has business impact scoring results")
    interaction_log_feedback: CaseStudyChatLogsFeedback = Field(...,description="It has interaction log feedback results")
    strengths_and_areas_of_improvements: CaseStudyStrengthsAndAreasOfImprovements = Field(...,description="It has strengths and areas of improvements results")


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