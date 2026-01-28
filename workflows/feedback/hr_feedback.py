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
    clarity: int = Field(...,description="The clarity in human communication")
    confidence: int = Field(...,description="The confidence in human communication")
    structure: int = Field(...,description="The structure in human communication")
    engagement: int = Field(...,description="The engagement in human communication")

class CulturalFitSkills(BaseModel):
    '''
    You need to assign the following range of numbers for the following cultural fit skills categories based on interaction history between intrviewer and interviewee- \n\n
    0 - If some major offense have occured or not applicable
    10,20,30,40 - If it was varying degrees of unseriousness, plainly wrong or barely comprehensive
    50 - If they were average
    60 - If they did decent
    70 - If they did good
    80 - If they did great
    90 - If they did amazing
    100 - If they did flawlessly
    '''
    values: int = Field(...,description="The company values in human interaction")
    teamwork: int = Field(...,description="The teamwork in human interaction")
    growth: int = Field(...,description="Growth sentiments in human interaction")
    initiative: int = Field(...,description="The initiative in human interaction")

Label = Literal["correct", "incorrect", "partially-correct", "cross-question"]


class FeedbackItem(BaseModel):
    '''
      For a pair of interaction, first mark their status and followed up by comments.
      For status, mark them -
      "cross-question" - If the interaction is part of cross-questioning
      "correct" -  If the interviewee has answered correctly
      "incorrect" - If the interviewee has answered incorrectly
      "partially-correct" - If the interviewee has answered only partially correct
      For comment, add the comments to tell how the answer could've been improved if it is not correct
    '''
    status: Label = Field(..., description="Mark the interaction status")
    comment: str = Field(..., description="Add any feedbacks.")



class ChatLogsFeedback(BaseModel):
    '''
        For a pair of interaction, first mark their status and followed up by comments.
        For status, mark them -
        "cross-question answer" - If the interaction is part of cross-questioning
        "correct answer" -  If the interviewee has answered correctly
        "incorrect answer" - If the interviewee has answered incorrectly
        "partially-correct answer" - If the interviewee has answered only partially correct
        For comment, add the comments to tell how the answer could've been improved if it is not correct

    '''
    answer_status: Literal[
        'cross-question answer', 'correct answer', 'incorrect answer', 'partially-correct answer'] = Field()
    comment: str = Field()

class HR_Strengths_and_areas_of_improvements(BaseModel):
    '''
    You are given interaction log between interviewer(ai) and interviewee(human).
    You need to first ensure that you have enough of a history to comment.
    '''
    strength1: str = Field(...,description="1 crisp points of strengths which you think interviewee(human) has in Cultural(clarity,confidence,structure,engagement) and problem solving skills(values,teamwork,growth,initiative) strictly based on the question asked and answer provided and address them in second person.")
    strength2: str = Field(...,description="1 crisp points of strengths which you think interviewee(human) has in Cultural(clarity,confidence,structure,engagement) and problem solving skills(values,teamwork,growth,initiative) strictly based on the question asked and answer provided and address them in second person.")
    strength3: str = Field(...,description="1 crisp points of strengths which you think interviewee(human) has in Cultural(clarity,confidence,structure,engagement) and problem solving skills(values,teamwork,growth,initiative) strictly based on the question asked and answer provided and address them in second person.")
    areas_of_improvements1: str = Field(...,description="1 crisp points of areas of improvements which you think interviewee(human) has in Cultural(clarity,confidence,structure,engagement) and problem solving skills(values,teamwork,growth,initiative) strictly based on the question asked and answer provided and address them in second person.")
    areas_of_improvements2: str = Field(...,description="1 crisp points of areas of improvements which you think interviewee(human) has in Cultural(clarity,confidence,structure,engagement) and problem solving skills(values,teamwork,growth,initiative) strictly based on the question asked and answer provided and address them in second person.")
    areas_of_improvements3: str = Field(...,description="1 crisp points of areas of improvements which you think interviewee(human) has in Cultural(clarity,confidence,structure,engagement) and problem solving skills(values,teamwork,growth,initiative) strictly based on the question asked and answer provided and address them in second person.")
# class InterviewAnalysisState(BaseModel):
#     history: List[Dict[Literal['human','ai'],str]] = Field(...,description="Log of interactions between interviewer and interviewee")
#     communication: CommunicationSkills = Field(...,description="Communication skills of the interviewee")


class HRIntState(TypedDict):
    history_log: str = Field(...,description="Has list of base messages")
    communication_skills: CommunicationSkills = Field(...,description="It has problem solving scoring results")
    cultural_skills: CulturalFitSkills = Field(...,description="It has technical scoring results")
    strengths_and_areas_of_improvements: HR_Strengths_and_areas_of_improvements
    interaction_log_feedback: ChatLogsFeedback


def cultural_skills_llm_Node(cultural_skills_llm):
    def _Node(state:HRIntState) -> HRIntState:
        response = cultural_skills_llm.invoke(state["history_log"])
        # print(response.syntax)
        # print(type(response))
        # print(isinstance(response,ProblemSolvingSkills))
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
