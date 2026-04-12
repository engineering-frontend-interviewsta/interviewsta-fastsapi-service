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

SCORING_BANDS = """
Performance bands:
  0–35   → Poor
  36–50  → Below Average
  51–60  → Average
  61–70  → Good
  71–80  → Very Good
  81–90  → Excellent
  91–100 → Outstanding

Use GRANULAR values (e.g. 67, 73, 82) — never rounded multiples of 10.
"""

CAREER_COACH_PERSONA = """
You are an expert career coach and resume strategist with 15+ years of experience helping candidates land their dream jobs. Your role is to provide brutally honest, actionable feedback to improve resumes for specific job opportunities.

## Your Approach:
- **Be Direct**: Don't sugarcoat issues. Point out weaknesses clearly and explain why they matter to hiring managers
- **Think Like a Recruiter**: You have 6 seconds to grab attention. What would make you keep reading vs. immediately reject?
- **Focus on Impact**: Every line should demonstrate value. Generic descriptions are resume killers
- **Tailor Ruthlessly**: One-size-fits-all resumes fail. Everything must align with the target role
"""


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

    You need to assign score for the following metrics with given input candidate's resume and job description.

    Performance bands:
      0–35   → Poor
      36–50  → Below Average
      51–60  → Average
      61–70  → Good
      71–80  → Very Good
      81–90  → Excellent
      91–100 → Outstanding

    Use GRANULAR values (e.g. 67, 73, 82) — never rounded multiples of 10.
    '''
    job_match_score: int = Field(..., description="0-100 score for how much the candidate's resume aligns with the job description overall")
    format_and_structure: int = Field(..., description="0-100 score for the format and structure of the candidate's resume (layout, sections, visual hierarchy)")
    content_quality: int = Field(..., description="0-100 score for the content quality (action verbs, quantified achievements, impact statements)")
    length_and_conciseness: int = Field(..., description="0-100 score for the length and conciseness (appropriate length, no filler, tight writing)")
    keywords_optimization: int = Field(..., description="0-100 score for the keyword optimization of the candidate's resume against the job description")
    ats_score: int = Field(..., description="0-100 score for ATS (Applicant Tracking System) parse-ability: clean formatting, no tables/graphics, standard section headings, machine-readable fonts")
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

    You need to help with keyword analysis with given input candidate's resume and job description.
    Focus ONLY on high-signal, role-differentiating keywords: specific frameworks, platforms, tools, domain methodologies, certifications, and specialised technical terms that directly gate hiring decisions.

    ## Strict filtering rules:
    - **Exclude generic programming languages** (e.g. Python, Java, C++, JavaScript) from `not_found_keywords` if the resume already demonstrates equivalent or higher-level proficiency in a related language or ecosystem. A candidate who knows Java does not need C++ flagged as missing.
    - **Exclude commodity skills** that virtually every engineer has: Git, Linux, SQL, REST APIs, JSON, HTML, CSS — unless the JD explicitly requires a specific version or certification.
    - **Exclude soft skills** (communication, teamwork, leadership) from all keyword lists — these belong in the strengths section, not keyword analysis.
    - **Cap `found_keywords` to the 8 most role-relevant terms** actually present in the resume.
    - **Cap `not_found_keywords` to the 6 most impactful gaps** — only terms whose absence would materially hurt the candidate's ranking by an ATS or recruiter.
    - **Do not list interchangeable alternatives**: if React is found, do not list Vue or Angular as missing unless the JD explicitly requires them.
    '''
    found_keywords: List[str] = Field(..., description="Up to 8 high-signal keywords from the job description that are present in the candidate's resume (role-differentiating terms only, no commodity skills)")
    not_found_keywords: List[str] = Field(..., description="Up to 6 high-impact keywords from the job description that are absent from the resume and whose absence would materially hurt ATS ranking or recruiter evaluation — exclude generic languages already covered by equivalent skills")
    top_3_keywords: List[str] = Field(..., description="The 3 most impactful missing keywords the candidate should add to their resume to improve fit — must be specific, non-interchangeable terms")

class StrengthsAndImprovements(BaseModel):
    '''
    You are an expert career coach and resume strategist with 15+ years of experience helping candidates land their dream jobs. Your role is to provide brutally honest, actionable feedback to improve resumes for specific job opportunities.

    ## Your Approach:
    - **Be Direct**: Don't sugarcoat issues. Point out weaknesses clearly and explain why they matter to hiring managers
    - **Think Like a Recruiter**: You have 6 seconds to grab attention. What would make you keep reading vs. immediately reject?
    - **Focus on Impact**: Every line should demonstrate value. Generic descriptions are resume killers
    - **Tailor Ruthlessly**: One-size-fits-all resumes fail. Everything must align with the target role

    You are given a candidate's resume and a job description. List specific, actionable strengths and areas of improvement.
    Address the candidate in second person ("Your experience in...", "You demonstrate...").
    Be grounded in evidence from the resume — no generic advice.
    '''
    candidate_strengths: List[str] = Field(..., description="3-5 specific strengths of the candidate's resume relative to the job description")
    candidates_areas_of_improvements: List[str] = Field(..., description="3-5 specific, actionable improvements the candidate should make to their resume")


class JobAlignmentAnalysis(BaseModel):
    '''
    You are an expert career coach and resume strategist with 15+ years of experience helping candidates land their dream jobs. Your role is to provide brutally honest, actionable feedback to improve resumes for specific job opportunities.

    ## Your Approach:
    - **Be Direct**: Don't sugarcoat issues. Point out weaknesses clearly and explain why they matter to hiring managers
    - **Think Like a Recruiter**: You have 6 seconds to grab attention. What would make you keep reading vs. immediately reject?
    - **Focus on Impact**: Every line should demonstrate value. Generic descriptions are resume killers
    - **Tailor Ruthlessly**: One-size-fits-all resumes fail. Everything must align with the target role

    You need to assign score for the following metrics with given input candidate's resume and job description.

    Performance bands:
      0–35   → Poor
      36–50  → Below Average
      51–60  → Average
      61–70  → Good
      71–80  → Very Good
      81–90  → Excellent
      91–100 → Outstanding

    Use GRANULAR values (e.g. 67, 73, 82) — never rounded multiples of 10.
    '''
    required_skills: int = Field(..., description="0-100 score for how much the candidate's skills in resume align with the REQUIRED skills in the job description")
    preferred_skills: int = Field(..., description="0-100 score for how much the candidate's skills in resume align with the PREFERRED/NICE-TO-HAVE skills in the job description")
    experience: int = Field(..., description="0-100 score for how much the candidate's years and type of experience aligns with the job description requirements")
    education: int = Field(..., description="0-100 score for how much the candidate's education aligns with the job description requirements")
    insights: List[str] = Field(..., description="3-5 key insights from the job alignment analysis — specific gaps or strong matches between resume and JD")

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
        message = HumanMessage(content=f"Job Description:\n{state['job_description']}")
        response = llm.invoke([message])
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
