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
    """
    Evaluate analytical skills in a Case Study Interview based on the interaction history.
    
    Score each skill on a precise 0-100 scale. Use granular values (e.g., 67, 73, 82) NOT rounded multiples of 10.
    
    Performance Bands:
    - 0: Insufficient data (fewer than 3 substantive responses) or major offense/misconduct
    - 1-35: Poor - Significant gaps, wrong analysis, or lack of basic analytical thinking
    - 36-50: Below Average - Some analytical ability but major weaknesses
    - 51-60: Average - Adequate analysis, meets basic expectations
    - 61-70: Good - Solid analytical thinking with consistent methodology
    - 71-80: Very Good - Strong analytical skills with minor areas for improvement
    - 81-90: Excellent - Expert-level analysis and problem-solving
    - 91-100: Outstanding - Exceptional mastery, flawless analytical reasoning
    
    CRITICAL: Assign specific scores within ranges (e.g., 67, 73, 82) based on nuanced performance.
    Do NOT use only multiples of 10 (10, 20, 30, etc.). Be precise and granular.
    
    Before assigning any score above 0, ensure:
    1. The human has provided at least 3 meaningful responses related to that skill area.
    2. There has been sufficient back-and-forth discussion to gauge their knowledge.
    3. The human has demonstrated (or failed to demonstrate) the specific skill through their responses.
    """
    problem_understanding: int = Field(..., description="Depth of understanding of the case study problem, key issues, and constraints. Score 0-100 with granular precision (e.g., 68, 76, 84). 0 if insufficient discussion.")
    hypothesis: int = Field(..., description="Quality of hypotheses formed, assumptions identified, and initial problem framing. Score 0-100 with granular precision (e.g., 72, 79, 87). 0 if insufficient discussion.")
    analysis: int = Field(..., description="Thoroughness and quality of analysis - data interpretation, pattern recognition, root cause identification. Score 0-100 with granular precision (e.g., 65, 77, 88). 0 if insufficient discussion.")
    synthesis: int = Field(..., description="Ability to synthesize information, connect insights, and form coherent conclusions. Score 0-100 with granular precision (e.g., 71, 81, 92). 0 if insufficient discussion.")

class BusinessImpactSkills(BaseModel):
    """
    Evaluate business impact skills in a Case Study Interview based on the interaction history.
    
    Score each skill on a precise 0-100 scale. Use granular values (e.g., 67, 73, 82) NOT rounded multiples of 10.
    
    Performance Bands:
    - 0: Insufficient data (fewer than 3 substantive responses) or major offense/misconduct
    - 1-35: Poor - Significant gaps, poor business judgment, or lack of strategic thinking
    - 36-50: Below Average - Some business awareness but major weaknesses
    - 51-60: Average - Adequate business thinking, meets basic expectations
    - 61-70: Good - Solid business acumen with consistent judgment
    - 71-80: Very Good - Strong business impact orientation with minor areas for improvement
    - 81-90: Excellent - Expert-level strategic thinking and business judgment
    - 91-100: Outstanding - Exceptional mastery, transformative business insights
    
    CRITICAL: Assign specific scores within ranges (e.g., 67, 73, 82) based on nuanced performance.
    Do NOT use only multiples of 10 (10, 20, 30, etc.). Be precise and granular.
    
    Before assigning any score above 0, ensure:
    1. The human has provided at least 3 meaningful responses related to that skill area.
    2. There has been sufficient back-and-forth discussion to gauge their knowledge.
    3. The human has demonstrated (or failed to demonstrate) the specific skill through their responses.
    """
    business_judgment: int = Field(..., description="Quality of business judgment, strategic thinking, and commercial awareness. Score 0-100 with granular precision (e.g., 69, 77, 85). 0 if insufficient discussion.")
    creativity: int = Field(..., description="Creative and innovative thinking in problem-solving and solution design. Score 0-100 with granular precision (e.g., 64, 73, 86). 0 if insufficient discussion.")
    decision_making: int = Field(..., description="Quality of decision-making process, trade-off analysis, and justification. Score 0-100 with granular precision (e.g., 71, 78, 88). 0 if insufficient discussion.")
    impact_orientation: int = Field(..., description="Focus on business impact, ROI, and measurable outcomes. Score 0-100 with granular precision (e.g., 74, 82, 91). 0 if insufficient discussion.")

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
    """
    Based on the interaction history between interviewer (AI) and interviewee (human) in a Case Study Interview,
    provide 3 specific strengths and 3 specific areas for improvement in their analytical and business impact skills.
    
    Focus on: problem understanding, hypothesis formation, analysis depth, synthesis, business judgment, creativity, decision-making, and impact orientation.
    Address the interviewee in second person (e.g., "You demonstrated strong...", "Your analysis of...").
    Be specific and actionable, strictly based on the questions asked and answers provided.
    """
    strength1: str = Field(..., description="1 crisp, specific strength in analytical or business impact skills, addressed in second person.")
    strength2: str = Field(..., description="1 crisp, specific strength in analytical or business impact skills, addressed in second person.")
    strength3: str = Field(..., description="1 crisp, specific strength in analytical or business impact skills, addressed in second person.")
    areas_of_improvements1: str = Field(..., description="1 crisp, actionable area for improvement in analytical or business impact skills, addressed in second person.")
    areas_of_improvements2: str = Field(..., description="1 crisp, actionable area for improvement in analytical or business impact skills, addressed in second person.")
    areas_of_improvements3: str = Field(..., description="1 crisp, actionable area for improvement in analytical or business impact skills, addressed in second person.")

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