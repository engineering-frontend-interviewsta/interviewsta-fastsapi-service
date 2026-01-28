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
    '''
    You need to assign scores for the following technical skills categories based on interaction history between interviewer and interviewee.

    IMPORTANT: Only provide scores above 0 if there has been sufficient conversation and human responses to make an accurate assessment.
    If there are fewer than 3 substantive human responses or insufficient discussion about a particular skill area,
    assign 0 to indicate "Insufficient Data/Not Applicable".

    Scoring Guidelines:
    0 - Insufficient conversation between interviewer and interviewee/data to make an assessment, major offense occurred, completely wrong approach, or not applicable to the role
    10,20,30,40 - Varying degrees of poor performance: unseriousness, plainly wrong, or barely comprehensive
    50 - Average performance with basic understanding
    60 - Decent performance with solid fundamentals
    70 - Good performance with strong knowledge
    80 - Great performance with deep understanding
    90 - Amazing performance with expert-level insights
    100 - Flawless performance with exceptional mastery

    Before assigning any score above 0, ensure:
    1. The human has provided at least 3 meaningful responses related to that skill area
    2. There has been sufficient back-and-forth discussion to gauge their knowledge
    3. The human has demonstrated (or failed to demonstrate) the specific skill through their responses
    '''
    programming_language: int = Field(..., description="The strength of fundamentals in programming language (0 if insufficient discussion)")
    framework: int = Field(..., description="Knowledge level of the coding framework (0 if insufficient discussion)")
    algorithms: int = Field(..., description="Knowledge level of algorithms and optimization (0 if insufficient discussion)")
    data_structures: int = Field(..., description="Knowledge level of data structures usage (0 if insufficient discussion)")




class ProblemSolvingSkills(BaseModel):
    '''
    You need to assign scores for the following problem-solving skills categories based on interaction history between interviewer and interviewee.

    IMPORTANT: Only provide scores above 0 if there has been sufficient conversation and human responses to make an accurate assessment.
    If there are fewer than 3 substantive human responses or insufficient discussion about a particular skill area,
    assign 0 to indicate "Insufficient Data/Not Applicable".

    Scoring Guidelines:
    0 - Insufficient conversation/data to make an assessment, major offense occurred, completely wrong approach, or not applicable to the role
    10,20,30,40 - Varying degrees of poor performance: unseriousness, plainly wrong, or barely comprehensive
    50 - Average performance with basic understanding
    60 - Decent performance with solid fundamentals
    70 - Good performance with strong knowledge
    80 - Great performance with deep understanding
    90 - Amazing performance with expert-level insights
    100 - Flawless performance with exceptional mastery

    Before assigning any score above 0, ensure:
    1. The human has provided at least 3 meaningful responses related to that skill area
    2. There has been sufficient back-and-forth discussion to gauge their knowledge
    3. The human has demonstrated (or failed to demonstrate) the specific skill through their responses
    '''
    approach: int = Field(..., description="The approach taken to solve technical problems (0 if insufficient discussion)")
    optimization: int = Field(..., description="The ability to optimize their solution (0 if insufficient discussion)")
    debugging: int = Field(..., description="The ability to find edge cases and rectify their code (0 if insufficient discussion)")
    syntax: int = Field(..., description="The syntax correctness of the programming language (0 if insufficient discussion)")

Label = Literal["correct", "incorrect", "partially-correct", "cross-question"]


# class FeedbackItem(BaseModel):
#     '''
#       For a pair of interaction, first mark their status and followed up by comments.
#       For status, mark them -
#       "cross-question" - If the interaction is part of cross-questioning
#       "correct" -  If the interviewee has answered correctly
#       "incorrect" - If the interviewee has answered incorrectly
#       "partially-correct" - If the interviewee has answered only partially correct
#       For comment, add the comments to tell how the answer could've been improved if it is not correct
#     '''
#     status: List[Label] = Field(..., description="Mark the interaction status")
#     comment: List[str] = Field(..., description="Add any feedbacks.")


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
    '''
    You are given interaction log between interviewer(ai) and interviewee(human).
    You need to first ensure that you have enough of a history to comment.
    '''
    strength1: str = Field(...,description="1 crisp points of strengths which you think interviewee(human) has in technical(programming language,framework knowledge,algorithms,data structures) and problem solving skills(approach,optimization,debugging and syntax) strictly based on the question asked and answer provided and address them in second person.")
    strength2: str = Field(...,description="1 crisp points of strengths which you think interviewee(human) has in technical(programming language,framework knowledge,algorithms,data structures) and problem solving skills(approach,optimization,debugging and syntax) strictly based on the question asked and answer provided and address them in second person.")
    strength3: str = Field(...,description="1 crisp points of strengths which you think interviewee(human) has in technical(programming language,framework knowledge,algorithms,data structures) and problem solving skills(approach,optimization,debugging and syntax) strictly based on the question asked and answer provided and address them in second person.")
    areas_of_improvements1: str = Field(...,description="1 crisp points of areas of improvements which you think interviewee(human) has in technical(programming language,framework knowledge,algorithms,data structures) and problem solving skills(approach,optimization,debugging and syntax)strictly based on the question asked and answer provided and address them in second person.")
    areas_of_improvements2: str = Field(...,description="1 crisp points of areas of improvements which you think interviewee(human) has in technical(programming language,framework knowledge,algorithms,data structures) and problem solving skills(approach,optimization,debugging and syntax)strictly based on the question asked and answer provided and address them in second person.")
    areas_of_improvements3: str = Field(...,description="1 crisp points of areas of improvements which you think interviewee(human) has in technical(programming language,framework knowledge,algorithms,data structures) and problem solving skills(approach,optimization,debugging and syntax)strictly based on the question asked and answer provided and address them in second person.")
# class InterviewAnalysisState(BaseModel):
#     history: List[Dict[Literal['human','ai'],str]] = Field(...,description="Log of interactions between interviewer and interviewee")
#     communication: CommunicationSkills = Field(...,description="Communication skills of the interviewee")

class TechIntState(TypedDict):
    history_log: str = Field(...,description="Has list of base messages")
    problem_solving: ProblemSolvingSkills = Field(...,description="It has problem solving scoring results")
    technical: TechnicalSkills = Field(...,description="It has technical scoring results")
    strengths_and_areas_of_improvements: Tech_Strengths_and_areas_of_improvements
    interaction_log_feedback: TechChatLogsFeedback

def problem_solving_llm_Node(problem_solving_llm):
    def _Node(state:TechIntState) -> TechIntState:
        response = problem_solving_llm.invoke(state["history_log"])
        # print(response.syntax)
        # print(type(response))
        # print(isinstance(response,ProblemSolvingSkills))
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
