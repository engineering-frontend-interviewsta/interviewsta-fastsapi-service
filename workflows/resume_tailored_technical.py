"""
Resume-tailored technical interview graph (separate from generic Technical).

Flow: Greeting → resume/project deep-dive (≈3–4 exchanges) → personal/fit (≈2–3) →
CS theory → coding → project discussion → closing job-match summary → End.
"""
from __future__ import annotations

from typing import Literal, TypeVar

from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph
from pydantic import BaseModel, Field

from workflows.technical import (
    TechnicalInterviewState,
    CodingProgress,
    ProjectProgress,
    TechnicalProgress,
    _format_job_context,
    create_before_coding,
    create_before_project,
    create_before_technical,
    create_coding_node,
    create_dummy_node,
    create_end_Node,
    create_project_node,
    create_route_to_coding,
    create_route_to_technical,
    create_technical_node,
    project_prompt,
    technical_prompt,
    technical_research_prompt,
)
from workflows.utils import get_llm

S = TypeVar("S")


resume_tailored_greeting_prompt = """
You are Glee, conducting a **resume-tailored technical** practice interview. Be warm, human, and professional.

Explain the flow clearly (high level only):
1) You will first discuss their **resume**—especially **projects and experience** (about 3–4 back-and-forths).
2) Then a few **personal / fit** questions (motivation, teamwork, why this role).
3) Then **CS fundamentals** (OS, DBMS, networks, etc., aligned with their background and the role).
4) A **live coding** segment.
5) A **deeper project / experience** discussion.
6) At the very end you will give a short **spoken summary** with **job match %** and **hire likelihood %** for this session.

Invite questions about the process. Do not start resume questions until they are ready (after any process Q&A).

Target role / JD context (do not read verbatim):
{job_context}

Resume (for context only; do not read verbatim):
{resume_text}
"""


def _resume_tailored_greeting_template(resume_text: str, job_context: str) -> ChatPromptTemplate:
    rt = (resume_text or "").strip() or "No resume text was provided."
    if len(rt) > 12000:
        rt = rt[:12000] + "\n[truncated]"
    jc = (job_context or "").strip() or _format_job_context("", "")
    return ChatPromptTemplate.from_messages(
        [("system", resume_tailored_greeting_prompt.format(resume_text=rt, job_context=jc))]
    )


resume_discussion_prompt = """
You are Glee in the **resume discussion** phase of a resume-tailored technical interview.

Your job for the next **3–4 substantive interviewer turns** (excluding short acknowledgements):
- Ground questions in **[RESUME]**—prioritize **projects**, internships, work experience, and tech stack.
- Tie questions to **[JOB_CONTEXT]** when natural (e.g. "How does X on your resume relate to what this role needs?").
- Go deeper on **one project at a time**: architecture, your contribution, trade-offs, failures, metrics.
- Do **not** ask full CS theory or coding problems yet—that comes later.
- Stay conversational; one clear question or prompt per turn.

[JOB_CONTEXT]
{job_context}

[RESUME]
{resume_text}
"""


personal_fit_prompt = """
You are Glee in the **personal / fit** phase (after resume discussion).

Ask about **2–3 short** personal or situational topics, for example:
- Why this role or transition; what draws them to this kind of work.
- Collaboration or conflict example (brief).
- How they learn or handle ambiguity.

Keep questions **non-technical**. Do not repeat detailed resume project probes. After ~2–3 exchanges, transition:
"Thanks for sharing. Next we'll move into some core technical questions."

[JOB_CONTEXT]
{job_context}
"""


resume_tailored_closing_prompt = """
You are Glee delivering the **final message** of a resume-tailored technical interview.

Using [RESUME], [JOB_CONTEXT], and the **entire conversation**, produce **one** closing assistant message that:
1. Thanks the candidate genuinely.
2. Gives two integer scores **0–100** (calibrated, evidence-based, not inflated):
   - **Job match score**: alignment of their background with the target role/JD.
   - **Hire likelihood (this interview only)**: how likely a panel would **advance** them based solely on this session—not a job offer prediction.
3. Adds **2–4 sentences** of concise rationale (strengths + gaps).

**Required lines** (exact labels, so downstream tools can parse if needed):
Job match: [N]%
Hire likelihood (this interview): [M]%

Do **not** ask a new interview question. This message closes the interview.

[JOB_CONTEXT]
{job_context}

[RESUME]
{resume_text}
"""


class OpeningProgressResumeTailored(BaseModel):
    send_to_which_node: Literal["Greeting", "Resume_discussion_before"] = Field(
        description="Route to 'Greeting' if the candidate still has questions about the process. "
        "Route to 'Resume_discussion_before' only when they are ready to begin the resume discussion."
    )


class ResumeDiscussionProgress(BaseModel):
    send_to_which_node: Literal["Resume_discussion", "Personal_fit_before"] = Field(
        description="Stay on 'Resume_discussion' until ~3–4 substantive resume/project Q&A rounds are done. "
        "Then route to 'Personal_fit_before'."
    )


class PersonalFitProgress(BaseModel):
    send_to_which_node: Literal["Personal_fit", "Technical_before"] = Field(
        description="Stay on 'Personal_fit' for ~2–3 personal/fit exchanges. Then route to 'Technical_before'."
    )


def create_resume_tailored_greeting_node(Greeting_llm):
    def _Node(state: S) -> S:
        if state["LastNode"] != "Greeting":
            jc = _format_job_context(state.get("job_title", ""), state.get("job_description", ""))
            tpl = _resume_tailored_greeting_template(state["resume"], jc)
            input_ = tpl.format_messages() + [{"role": "human", "content": "Start the interview now"}]
            state["messages"] = state["messages"] + input_
        response = Greeting_llm.invoke(state["messages"])
        state["messages"] = state["messages"] + [response]
        state["history"] = state["history"] + "\n" + "Interviewer-" + response.content
        state["LastNode"] = "Greeting"
        return state

    return _Node


def create_resume_discussion_node(llm):
    def _Node(state: S) -> S:
        if state["LastNode"] != "Resume_discussion":
            jc = _format_job_context(state.get("job_title", ""), state.get("job_description", ""))
            rt = (state.get("resume") or "").strip() or "No resume provided."
            if len(rt) > 12000:
                rt = rt[:12000] + "\n[truncated]"
            state["messages"][0].content = resume_discussion_prompt.format(job_context=jc, resume_text=rt)
        response = llm.invoke(state["messages"])
        state["messages"] = state["messages"] + [response]
        state["history"] = state["history"] + "\n" + "Interviewer-" + response.content
        state["LastNode"] = "Resume_discussion"
        return state

    return _Node


def create_personal_fit_node(llm):
    def _Node(state: S) -> S:
        if state["LastNode"] != "Personal_fit":
            jc = _format_job_context(state.get("job_title", ""), state.get("job_description", ""))
            state["messages"][0].content = personal_fit_prompt.format(job_context=jc)
        response = llm.invoke(state["messages"])
        state["messages"] = state["messages"] + [response]
        state["history"] = state["history"] + "\n" + "Interviewer-" + response.content
        state["LastNode"] = "Personal_fit"
        return state

    return _Node


def create_resume_tailored_summary_node(llm):
    def _Node(state: S) -> S:
        if state["LastNode"] != "Resume_tailored_summary":
            jc = _format_job_context(state.get("job_title", ""), state.get("job_description", ""))
            rt = (state.get("resume") or "").strip() or "No resume provided."
            if len(rt) > 12000:
                rt = rt[:12000] + "\n[truncated]"
            state["messages"][0].content = resume_tailored_closing_prompt.format(job_context=jc, resume_text=rt)
        response = llm.invoke(state["messages"])
        state["messages"] = state["messages"] + [response]
        state["history"] = state["history"] + "\n" + "Interviewer-" + response.content
        state["LastNode"] = "Resume_tailored_summary"
        return state

    return _Node


def create_route_opening_resume_tailored(Opening_llm):
    def _Node(state: S) -> Literal["Greeting", "Resume_discussion_before"]:
        response = Opening_llm.invoke(state["history"])
        return response.send_to_which_node

    return _Node


def create_route_resume_discussion(ResumeDisc_llm):
    def _Node(state: S) -> Literal["Resume_discussion", "Personal_fit_before"]:
        response = ResumeDisc_llm.invoke(state["history"])
        return response.send_to_which_node

    return _Node


def create_route_personal_fit(Personal_llm):
    def _Node(state: S) -> Literal["Personal_fit", "Technical_before"]:
        response = Personal_llm.invoke(state["history"])
        return response.send_to_which_node

    return _Node


def create_route_to_project_then_summary(ProjectProgress_llm):
    """After project phase, emit closing metrics node instead of jumping straight to End."""

    def _Node(state: S) -> Literal["Project", "Resume_tailored_summary"]:
        response = ProjectProgress_llm.invoke(state["history"])
        if response.send_to_which_node == "End":
            return "Resume_tailored_summary"
        return "Project"

    return _Node


def get_resume_tailored_technical_graph(google_api_key: str, tavily_api_key: str, checkpointer):
    _ = tavily_api_key
    llm = get_llm(google_api_key=google_api_key)
    workflow = StateGraph(TechnicalInterviewState)

    workflow.add_node("Greeting", create_resume_tailored_greeting_node(llm))
    workflow.add_node("Greeting_after", create_dummy_node())
    workflow.add_node("Resume_discussion_before", create_dummy_node())
    workflow.add_node("Resume_discussion", create_resume_discussion_node(llm))
    workflow.add_node("Resume_discussion_after", create_dummy_node())
    workflow.add_node("Personal_fit_before", create_dummy_node())
    workflow.add_node("Personal_fit", create_personal_fit_node(llm))
    workflow.add_node("Personal_fit_after", create_dummy_node())
    workflow.add_node("Technical_before", create_before_technical(llm))
    workflow.add_node("Technical", create_technical_node(llm))
    workflow.add_node("Technical_after", create_dummy_node())
    workflow.add_node("Coding_before", create_before_coding(llm))
    workflow.add_node("Coding", create_coding_node(llm))
    workflow.add_node("Coding_after", create_dummy_node())
    workflow.add_node("Project_before", create_before_project(llm))
    workflow.add_node("Project", create_project_node(llm))
    workflow.add_node("Project_after", create_dummy_node())
    workflow.add_node("Resume_tailored_summary", create_resume_tailored_summary_node(llm))
    workflow.add_node("Resume_tailored_summary_after", create_dummy_node())
    workflow.add_node("End", create_end_Node())

    workflow.set_entry_point("Greeting")
    workflow.add_edge("Greeting", "Greeting_after")
    workflow.add_conditional_edges(
        "Greeting_after",
        create_route_opening_resume_tailored(llm.with_structured_output(OpeningProgressResumeTailored)),
    )
    workflow.add_edge("Resume_discussion_before", "Resume_discussion")
    workflow.add_edge("Resume_discussion", "Resume_discussion_after")
    workflow.add_conditional_edges(
        "Resume_discussion_after",
        create_route_resume_discussion(llm.with_structured_output(ResumeDiscussionProgress)),
    )
    workflow.add_edge("Personal_fit_before", "Personal_fit")
    workflow.add_edge("Personal_fit", "Personal_fit_after")
    workflow.add_conditional_edges(
        "Personal_fit_after",
        create_route_personal_fit(llm.with_structured_output(PersonalFitProgress)),
    )
    workflow.add_edge("Technical_before", "Technical")
    workflow.add_edge("Technical", "Technical_after")
    workflow.add_conditional_edges(
        "Technical_after",
        create_route_to_technical(llm.with_structured_output(TechnicalProgress)),
    )
    workflow.add_edge("Coding_before", "Coding")
    workflow.add_edge("Coding", "Coding_after")
    workflow.add_conditional_edges(
        "Coding_after",
        create_route_to_coding(llm.with_structured_output(CodingProgress)),
    )
    workflow.add_edge("Project_before", "Project")
    workflow.add_edge("Project", "Project_after")
    workflow.add_conditional_edges(
        "Project_after",
        create_route_to_project_then_summary(llm.with_structured_output(ProjectProgress)),
    )
    workflow.add_edge("Resume_tailored_summary", "Resume_tailored_summary_after")
    workflow.add_edge("Resume_tailored_summary_after", "End")
    workflow.add_edge("End", "__end__")

    return workflow.compile(checkpointer=checkpointer)
