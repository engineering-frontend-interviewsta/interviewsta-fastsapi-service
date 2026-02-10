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
from typing_extensions import TypedDict
import getpass
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod
from ..utils import get_llm

class CompanyandRole(BaseModel):
    '''
    You are provided with job description for an opening, you need to identify-
    1) Company for which the opening is for
    2) Role for which the job description is
    '''
    company: str = Field(..., description="Company for which the opening is for")
    role: str = Field(..., description="Role for which the job description is")
class SectionAnalysis(BaseModel):
    '''
    You are an expert career coach and resume strategist with 15+ years of experience helping candidates land their dream jobs. Your role is to provide brutally honest, actionable feedback to improve resumes for specific job opportunities.

    ## Your Approach:
    - **Be Direct**: Don't sugarcoat issues. Point out weaknesses clearly and explain why they matter to hiring managers
    - **Think Like a Recruiter**: You have 6 seconds to grab attention. What would make you keep reading vs. immediately reject?
    - **Focus on Impact**: Every line should demonstrate value. Generic descriptions are resume killers
    - **Tailor Ruthlessly**: One-size-fits-all resumes fail. Everything must align with the target role

    You need to assign score for the follow metrics with given input candidate's resume and job description-
    '''
    job_match_score: int = Field(...,description="0-100 score for how much the candiate's resume aligns with the job description")
    format_and_structure: int = Field(...,description="0-100 score for the format and structure of the candidate's resume")
    content_quality: int = Field(...,description="0-100 score for the content quality of the candidate's resume")
    length_and_conciseness: int = Field(...,description="0-100 score for the length and conciseness of the candidate's resume")
    keywords_optimization: int = Field(...,description="0-100 score for the keyword optimization of the candidate's resume")
#
# class Keyword(BaseModel):
#     '''
#     You are an expert career coach and resume strategist with 15+ years of experience helping candidates land their dream jobs. Your role is to provide brutally honest, actionable feedback to improve resumes for specific job opportunities.
#
#     ## Your Approach:
#     - **Be Direct**: Don't sugarcoat issues. Point out weaknesses clearly and explain why they matter to hiring managers
#     - **Think Like a Recruiter**: You have 6 seconds to grab attention. What would make you keep reading vs. immediately reject?
#     - **Focus on Impact**: Every line should demonstrate value. Generic descriptions are resume killers
#     - **Tailor Ruthlessly**: One-size-fits-all resumes fail. Everything must align with the target role
#
#     You need to help with keyword analysis with given input candidate's resume and job description-
#     '''
#     found_keywords: List[str] = Field(..., description="Keywords found in the candidate's resume")
#     not_found_keywords: List[str] = Field(...,
#                                           description="Keywords not found in the candidate's resume related to the job description")
#     top_3_keywords: List[str] = Field(...,
#                                       description="3 Keywords that should be present in candidate's resume to make it more fit for the job")
#     # keyword_score: int = Field(...,description="0-100 score for ")

class Keyword(BaseModel):
    '''
    You are an expert career coach and resume strategist with 15+ years of experience helping candidates land their dream jobs. Your role is to provide brutally honest, actionable feedback to improve resumes for specific job opportunities.

    ## Your Approach:
    - **Be Direct**: Don't sugarcoat issues. Point out weaknesses clearly and explain why they matter to hiring managers
    - **Think Like a Recruiter**: You have 6 seconds to grab attention. What would make you keep reading vs. immediately reject?
    - **Focus on Impact**: Every line should demonstrate value. Generic descriptions are resume killers
    - **Tailor Ruthlessly**: One-size-fits-all resumes fail. Everything must align with the target role

    You need to help with keyword analysis with given input candidate's resume and job description-
    '''
    found_keywords: List[str] = Field(..., description="Keywords found in the candidate's resume")
    not_found_keywords: List[str] = Field(...,
                                          description="Keywords not found in the candidate's resume related to the job description")
    top_3_keywords: List[str] = Field(...,
                                      description="3 Keywords that should be present in candidate's resume to make it more fit for the job")
    # keyword_score: int = Field(...,description="0-100 score for ")

class StrengthsAndImprovements(BaseModel):
    '''
    You are an expert career coach and resume strategist with 15+ years of experience helping candidates land their dream jobs. Your role is to provide brutally honest, actionable feedback to improve resumes for specific job opportunities.

    ## Your Approach:
    - **Be Direct**: Don't sugarcoat issues. Point out weaknesses clearly and explain why they matter to hiring managers
    - **Think Like a Recruiter**: You have 6 seconds to grab attention. What would make you keep reading vs. immediately reject?
    - **Focus on Impact**: Every line should demonstrate value. Generic descriptions are resume killers
    - **Tailor Ruthlessly**: One-size-fits-all resumes fail. Everything must align with the target role

    You are given a candidate's resume and a job description, you need to list out strengths and aread of improvements -
    '''
    candidate_strengths: List[str] = Field(..., description="List of strengths of the candidate")
    candidates_areas_of_improvements: List[str] = Field(..., description="List of improvements that can be made to the candidate")


class JobAlignmentAnalysis(BaseModel):
    '''
    You are an expert career coach and resume strategist with 15+ years of experience helping candidates land their dream jobs. Your role is to provide brutally honest, actionable feedback to improve resumes for specific job opportunities.

    ## Your Approach:
    - **Be Direct**: Don't sugarcoat issues. Point out weaknesses clearly and explain why they matter to hiring managers
    - **Think Like a Recruiter**: You have 6 seconds to grab attention. What would make you keep reading vs. immediately reject?
    - **Focus on Impact**: Every line should demonstrate value. Generic descriptions are resume killers
    - **Tailor Ruthlessly**: One-size-fits-all resumes fail. Everything must align with the target role

    You need to assign score for the follow metrics with given input candidate's resume and job description-
    '''
    required_skills: int = Field(...,description="0-100 score for how much the candiate's skills in resume aligns with the required skills in job description")
    preferred_skills: int = Field(...,description="0-100 score for how much the candiate's skills in resume aligns with the preferred skills in job description")
    experience: int = Field(...,description="0-100 score for how much the candidate`s experience aligns with the job description")
    education: int = Field(...,description="0-100 score for how much the candidate`s education aligns with the job description")
    insights: List[str] = Field(...,description="Insights from the analysis")

class State(TypedDict):
    input_message: List[BaseMessage]
    job_description: str
    section_analysis: SectionAnalysis = Field(...,description="Section analysis of the resume")
    keyword_analysis: Keyword = Field(...,description="Keyword analysis of the resume")
    job_alignment_analysis: JobAlignmentAnalysis = Field(...,description="Job alignment analysis of the resume")
    strengths_and_improvements: StrengthsAndImprovements = Field(...,description="Strengths and improvements of the resume")
    company: str = Field(...,description="Company which is mentioned in the job description")
    role: str = Field(..., description="Role which is mentioned in the job description")


def company_and_job_description_Node(llm):
    def _Node(state: State) -> State:
        response = llm.invoke(state["job_description"])
        state["company"] = response.company
        state["role"] = response.role
        return state
    return _Node

def section_analysis_Node(llm):
    def _Node(state: State) -> State:
        state["section_analysis"] = llm.invoke(state["input_message"])
        return state

    return _Node


def keyword_analysis_Node(llm):
    def _Node(state: State) -> State:
        state["keyword_analysis"] = llm.invoke(state["input_message"])
        return state

    return _Node


def job_alignment_analysis_Node(llm):
    def _Node(state: State) -> State:
        state["job_alignment_analysis"] = llm.invoke(state["input_message"])
        return state

    return _Node


def strengths_and_improvements_Node(llm):
    def _Node(state: State) -> State:
        state["strengths_and_improvements"] = llm.invoke(state["input_message"])
        return state

    return _Node


def build_resume_analysis_graph(google_api_key):
    llm = get_llm(google_api_key)

    section_llm = llm.with_structured_output(SectionAnalysis)
    keyword_llm = llm.with_structured_output(Keyword)
    job_alignment_llm = llm.with_structured_output(JobAlignmentAnalysis)
    strengths_llm = llm.with_structured_output(StrengthsAndImprovements)
    company_and_role_llm = llm.with_structured_output(CompanyandRole)

    graph = StateGraph(State)
    graph.add_node("company_and_role", company_and_job_description_Node(company_and_role_llm))
    graph.add_node("section_analysis", section_analysis_Node(section_llm))
    graph.add_node("keyword_analysis", keyword_analysis_Node(keyword_llm))
    graph.add_node("job_alignment_analysis", job_alignment_analysis_Node(job_alignment_llm))
    graph.add_node("strengths_and_improvements", strengths_and_improvements_Node(strengths_llm))

    graph.set_entry_point("section_analysis")
    graph.add_edge("section_analysis", "keyword_analysis")
    graph.add_edge("keyword_analysis", "job_alignment_analysis")
    graph.add_edge("job_alignment_analysis", "strengths_and_improvements")
    graph.add_edge("strengths_and_improvements", "company_and_role")
    graph.add_edge("company_and_role", "__end__")
    agent = graph.compile()

    return agent
