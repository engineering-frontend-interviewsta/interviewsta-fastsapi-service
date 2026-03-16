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
    '''
    You need to assign the following range of numbers for the following communication skills categories based on interaction history between intrviewer and interviewee- \n\n
    0 - If some major offense have occured or not applicable
    10,20,30,40 - If it was varying degrees of unseriousness, plainly wrong or barely comprehensive
    50 - If they were average
    60 - If they did decent
    70 - If they did good
    80 - If they did great
    90 - If they did amazing
    100 - If they did flawlessly
    '''
    clarity: int = Field(..., description="The clarity in human communication")
    confidence: int = Field(..., description="The confidence in human communication")
    structure: int = Field(..., description="The structure in human communication")
    engagement: int = Field(..., description="The engagement in human communication")


class TechnicalSkills(BaseModel):
    """
    Evaluate technical skills in a Technical/Coding Interview based on the interaction history.
    
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
    2. There has been sufficient back-and-forth discussion to gauge their knowledge.
    3. The human has demonstrated (or failed to demonstrate) the specific skill through their responses.
    """
    programming_language: int = Field(..., description="Strength of fundamentals in programming language syntax, concepts, and best practices. Score 0-100 with granular precision (e.g., 68, 76, 84). 0 if insufficient discussion.")
    framework: int = Field(..., description="Knowledge and effective use of coding frameworks, libraries, and tools. Score 0-100 with granular precision (e.g., 72, 79, 87). 0 if insufficient discussion.")
    algorithms: int = Field(..., description="Understanding of algorithms, complexity analysis, and optimization techniques. Score 0-100 with granular precision (e.g., 65, 77, 88). 0 if insufficient discussion.")
    data_structures: int = Field(..., description="Knowledge and appropriate usage of data structures. Score 0-100 with granular precision (e.g., 71, 81, 92). 0 if insufficient discussion.")




class ProblemSolvingSkills(BaseModel):
    """
    Evaluate problem-solving skills in a Technical/Coding Interview based on the interaction history.
    
    Score each skill on a precise 0-100 scale. Use granular values (e.g., 67, 73, 82) NOT rounded multiples of 10.
    
    Performance Bands:
    - 0: Insufficient data (fewer than 3 substantive responses) or major offense/misconduct
    - 1-35: Poor - Significant gaps, wrong approaches, or lack of problem-solving ability
    - 36-50: Below Average - Some problem-solving but major weaknesses
    - 51-60: Average - Adequate approach, meets basic expectations
    - 61-70: Good - Solid problem-solving with consistent methodology
    - 71-80: Very Good - Strong analytical skills with minor areas for improvement
    - 81-90: Excellent - Expert-level problem-solving and optimization
    - 91-100: Outstanding - Exceptional mastery, flawless execution
    
    CRITICAL: Assign specific scores within ranges (e.g., 67, 73, 82) based on nuanced performance.
    Do NOT use only multiples of 10 (10, 20, 30, etc.). Be precise and granular.
    
    Before assigning any score above 0, ensure:
    1. The human has provided at least 3 meaningful responses related to that skill area.
    2. There has been sufficient back-and-forth discussion to gauge their knowledge.
    3. The human has demonstrated (or failed to demonstrate) the specific skill through their responses.
    """
    approach: int = Field(..., description="Quality of approach taken to solve technical problems - problem breakdown, strategy selection. Score 0-100 with granular precision (e.g., 69, 77, 85). 0 if insufficient discussion.")
    optimization: int = Field(..., description="Ability to optimize solutions for time/space complexity and efficiency. Score 0-100 with granular precision (e.g., 64, 73, 86). 0 if insufficient discussion.")
    debugging: int = Field(..., description="Ability to identify edge cases, bugs, and rectify code issues. Score 0-100 with granular precision (e.g., 71, 78, 88). 0 if insufficient discussion.")
    syntax: int = Field(..., description="Syntax correctness and code quality in the programming language. Score 0-100 with granular precision (e.g., 74, 82, 91). 0 if insufficient discussion.")

Label = Literal["correct", "incorrect", "partially-correct", "cross-question"]


class TechChatLogsFeedback(BaseModel):
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


class Tech_Strengths_and_areas_of_improvements(BaseModel):
    """
    Based on the interaction history between interviewer (AI) and interviewee (human) in a Technical Interview,
    provide 3 specific strengths and 3 specific areas for improvement in their technical and problem-solving skills.
    
    Focus on: programming language fundamentals, framework knowledge, algorithms, data structures, approach, optimization, debugging, and syntax.
    Address the interviewee in second person (e.g., "You demonstrated strong...", "Your algorithm choice...").
    Be specific and actionable, strictly based on the questions asked and answers provided.
    """
    strength1: str = Field(..., description="1 crisp, specific strength in technical or problem-solving skills, addressed in second person.")
    strength2: str = Field(..., description="1 crisp, specific strength in technical or problem-solving skills, addressed in second person.")
    strength3: str = Field(..., description="1 crisp, specific strength in technical or problem-solving skills, addressed in second person.")
    areas_of_improvements1: str = Field(..., description="1 crisp, actionable area for improvement in technical or problem-solving skills, addressed in second person.")
    areas_of_improvements2: str = Field(..., description="1 crisp, actionable area for improvement in technical or problem-solving skills, addressed in second person.")
    areas_of_improvements3: str = Field(..., description="1 crisp, actionable area for improvement in technical or problem-solving skills, addressed in second person.")

class TechIntState(TypedDict):
    history_log: str
    problem_solving: ProblemSolvingSkills
    technical: TechnicalSkills
    strengths_and_areas_of_improvements: Tech_Strengths_and_areas_of_improvements
    interaction_log_feedback: TechChatLogsFeedback

def problem_solving_llm_Node(problem_solving_llm):
    def _Node(state:TechIntState) -> TechIntState:
        response = problem_solving_llm.invoke(state["history_log"])
        state["problem_solving"] = response
        return state
    return _Node

def technical_llm_Node(technical_llm):
    def _Node(state:TechIntState) -> TechIntState:
        response = technical_llm.invoke(state["history_log"])
        print(response)
        state["technical"] = response
        return state
    return _Node

def strengths_and_areas_of_improvements_llm_Node(strengths_and_areas_of_improvements_llm):
     def _Node(state:TechIntState) -> TechIntState:
        response = strengths_and_areas_of_improvements_llm.invoke(state["history_log"])
        print(response)
        state["strengths_and_areas_of_improvements"] = response
        return state
     return _Node



def chat_logs_feedback_Node(feedback_llm):
    def _Node(state:TechIntState) -> TechIntState:
        _history = state["history_log"]
        response = feedback_llm.invoke(_history)
        print("This is the interaction log feedback",response)
        state["interaction_log_feedback"] = response
        return state

    return _Node


def build_tech_skills_feedback_graph(google_api_key:str):
    llm = get_llm(google_api_key)

    problem_solving_llm = llm.with_structured_output(ProblemSolvingSkills)
    technical_llm = llm.with_structured_output(TechnicalSkills)
    strengths_and_areas_of_improvements_llm = llm.with_structured_output(Tech_Strengths_and_areas_of_improvements)
    feedback_llm = llm.with_structured_output(TechChatLogsFeedback)

    graph = StateGraph(TechIntState)
    graph.add_node("problem_solving", problem_solving_llm_Node(problem_solving_llm))
    graph.add_node("technical", technical_llm_Node(technical_llm))
    graph.add_node("strengths_and_areas_of_improvements", strengths_and_areas_of_improvements_llm_Node(strengths_and_areas_of_improvements_llm))
    graph.add_node("chat_logs_feedback",
                   chat_logs_feedback_Node(feedback_llm))

    graph.add_edge("problem_solving", "technical")
    graph.add_edge("technical", "strengths_and_areas_of_improvements")
    graph.add_edge("strengths_and_areas_of_improvements","chat_logs_feedback")
    graph.add_edge("chat_logs_feedback", "__end__")

    graph.set_entry_point("problem_solving")
    agent = graph.compile()

    return agent
