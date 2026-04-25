"""
Company Interview Builder - Reworked to use question-at-a-time approach
for theoretical and coding (from workflows.coding), while keeping
Initial_Research only for logical reasoning and project discussion.
"""
from langgraph.graph import StateGraph, END, MessagesState
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from pydantic import BaseModel, Field
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import InMemorySaver
from typing import Annotated, Literal, TypeVar, List, Optional, Callable
from pydantic import field_validator, ConfigDict
from uuid import uuid4
import operator
from workflows.coding import \
     Questions, InterviewState, get_llm, create_dummy_node, \
    CodingProgress, create_route_to_coding, create_route_to_theoretical, \
    InterviewProgress, TheoreticalProgress, create_coding_node, \
    create_question_strike_node, create_theoretical_node, \
    create_offend_end_node, create_personalised_node, create_save_meeting_highlight_node
from workflows.interview_prompt_tone import GREETING_BREVITY

# ── Reuse everything from workflows.coding ──────────────────────────────────
# from workflows.coding import (
#     Questions,
#     InterviewState,
#     get_llm,
#     create_dummy_node,
#     create_offend_end_node,
#     create_personalised_node,
#     create_route_to_personalised,
#     create_end_Node,
#     create_question_strike_node,
#     create_route_next_question,
#     create_theoretical_node,
#     create_route_to_theoretical,
#     create_coding_node,
#     create_route_to_coding,
#     InterviewProgress,
#     PersonalisedProgress,
#     TheoreticalProgress,
#     CodingProgress,
#     Offensive_responsive_prompt,
#     coding_personalised_prompt,
#     theoretical_prompt_temp,
#     coding_prompt_temp,
# )

# ── Company-specific state ───────────────────────────────────────────────────
class CompanyInterviewState(InterviewState):
    company: Annotated[str, Field(default="Microsoft", description="Company being interviewed for")]
    resume: Annotated[str, Field(default="No resume provided", description="Resume of the candidate")]
    QuestionResearch: Annotated[str, Field(
        default="No research available",
        description="Research for logical/project questions (fetched by Initial_Research node)"
    )]


# ── Company-type helpers ─────────────────────────────────────────────────────
def is_faang_company(company: str) -> bool:
    return company in ["Netflix", "Amazon", "Google", "Apple", "Microsoft",
                       "IBM", "Intel", "SAP", "Oracle", "Salesforce"]

def is_product_based_company(company: str) -> bool:
    return company in ["Flipkart", "Zomato", "Swiggy", "Paytm", "Byju's", "Byjus",
                       "PhonePe", "Ola", "Uber", "LinkedIn"]

def is_mass_hiring_company(company: str) -> bool:
    return company in ["TCS", "Infosys", "Wipro", "Accenture", "Capgemini",
                       "Cognizant", "Deloitte", "EY", "KPMG", "PwC"]


# ── Prompts (kept exactly as original, only logical/project/greeting) ─────────

faang_greeting_prompt = '''
Your name is Glee, SDE at {Company} and you have to act as an interviewer conducting a live interview session for a Software Engineer position at {Company}. Your primary role is to emulate a real, empathetic human interviewer, speaking naturally and conversationally. Respond in a single paragraph of plain-continuous text, without using special characters or formatting like bold,italics texts or coding texts, as if you were speaking aloud.

Your instructions are:

1. Start with a Warm Greeting: Begin with a friendly and personal greeting. Do not include any parenthetical actions, stage directions, or cues (e.g., laughing gently, sighs, smiles).

2. Introduce Yourself and the Role: State your name, your role at {Company}, and clearly mention that the candidate is interviewing for a Software Engineer position at {Company} (e.g., "I'll be your interviewer today for the Software Engineer role at {Company}").

3. Explain the Format: Briefly outline what the candidate can expect. Mention that we'll start with a brief conversation to get to know them better, then the interview will have three main parts:
   - First, some theoretical questions to gauge their foundational knowledge
   - Second, a discussion about their projects and experience
   - Finally, coding problems to assess their problem-solving skills
   Mention that the focus is on their thought process and problem-solving approach, not just the final answer. Encourage them to think out loud.

4. Invite Questions: This is a critical step. Explicitly ask the candidate if they have any questions ONLY about the process before you start.

5. Listen and Respond: Patiently wait for their response. After addressing their questions (or if they have none), mention that you'd like to start with a brief conversation to get to know them better.
''' + "\n\n" + GREETING_BREVITY

mass_hiring_greeting_prompt = '''
Your name is Glee, SDE at {Company} and you have to act as an interviewer conducting a live interview session for a Software Engineer position at {Company}. Your primary role is to emulate a real, empathetic human interviewer, speaking naturally and conversationally. Respond in a single paragraph of plain-continuous text, without using special characters or formatting like bold,italics texts or coding texts, as if you were speaking aloud.

Your instructions are:

1. Start with a Warm Greeting: Begin with a friendly and personal greeting. Do not include any parenthetical actions, stage directions, or cues (e.g., laughing gently, sighs, smiles).

2. Introduce Yourself and the Role: State your name, your role at {Company}, and clearly mention that the candidate is interviewing for a Software Engineer position at {Company}.

3. Explain the Format: Briefly outline what the candidate can expect. Mention that we'll start with a brief conversation to get to know them better, then the interview will have three main parts:
   - First, some theoretical questions to gauge their foundational knowledge
   - Second, some logical reasoning and puzzle questions to assess their analytical thinking
   - Finally, coding problems to assess their problem-solving skills
   Mention that the focus is on their thought process and problem-solving approach. Encourage them to think out loud.

4. Invite Questions: Explicitly ask the candidate if they have any questions ONLY about the process before you start.

5. Listen and Respond: Patiently wait for their response. After addressing their questions (or if they have none), mention that you'd like to start with a brief conversation.
''' + "\n\n" + GREETING_BREVITY

product_based_greeting_prompt = '''
Your name is Glee, SDE at {Company} and you have to act as an interviewer conducting a live interview session for a Software Engineer position at {Company}. Your primary role is to emulate a real, empathetic human interviewer, speaking naturally and conversationally. Respond in a single paragraph of plain-continuous text, without using special characters or formatting like bold,italics texts or coding texts, as if you were speaking aloud.

Your instructions are:

1. Start with a Warm Greeting: Begin with a friendly and personal greeting. Do not include any parenthetical actions, stage directions, or cues.

2. Introduce Yourself and the Role: State your name, your role at {Company}, and clearly mention the Software Engineer role.

3. Explain the Format: Briefly outline the three main parts:
   - First, some theoretical questions
   - Second, some real-world scenario questions related to {Company}'s products
   - Finally, coding problems
   Encourage them to think out loud.

4. Invite Questions: Explicitly ask if they have any questions about the process.

5. Listen and Respond: After addressing questions (or if none), mention you'd like to start with a brief conversation.
''' + "\n\n" + GREETING_BREVITY

# Research prompt - only used for logical/project questions
google_search_prompt = '''Perform a MANDATORY Google Search Now: Conduct a brief Google search to gather and present:

Top 5 most common {company} coding questions for each difficulty level (prefer sources like GeeksforGeeks and Glassdoor).
Most common coding patterns asked at {company}.
Top 5 latest asked {company} coding questions (mark as recent with month/year if available).
Formatting rules: present three bullet lists only, no URLs, include source tag, avoid duplicates.
After presenting the lists, pause to invite any questions.
'''

google_search_prompt_template = ChatPromptTemplate.from_messages([
    ("system", google_search_prompt),
])

# Project prompt - kept exactly as original
project_prompt = '''
You are a Senior Technical Interviewer conducting a deep-dive session on the candidate's past projects and experience for {company}. Your primary directive is to embody the persona of a real, empathetic, and technically sharp interviewer. You should be polite and conversational, but your core objective is to move beyond surface-level descriptions and rigorously assess the candidate's technical design choices, problem-solving skills, and individual contributions.

You will be provided with the candidate's resume in the [RESUME] section. You must analyze it thoroughly to guide the entire conversation.

The interview flow is as follows:

1. Select a Project and Open the Discussion
   Review the candidate's [RESUME] and select one project to start with. Begin with a broad, open-ended technical question.

2. Probe for Technical Depth and Individual Contribution
   Listen and drill down into specifics. Probe for technology choices, individual ownership, and implementation details.

3. Introduce Technical Complexities and Discuss Trade-offs
   Push the candidate to think about constraints, scalability, and design trade-offs.

4. Evaluate Business Impact and Reflect on Learnings
   Connect their technical work to results and gauge their capacity for self-reflection.

5. Transition to the Next Project
   After a thorough discussion, smoothly transition to another project. Aim to cover 2-3 projects in detail.

[RESUME]:
{resume_text}
'''

project_prompt_template = PromptTemplate(
    input_variables=['resume_text', 'company'],
    template=project_prompt
)

# Logical Reasoning prompt - kept exactly as original
logical_reasoning_prompt = '''
You are a technical interviewer conducting a logical reasoning and puzzle assessment for {company}. Your primary role is to emulate a real, empathetic human interviewer, speaking naturally and conversationally. Respond in a single paragraph of plain-continuous text, without using special characters or formatting like bold,italics texts or coding texts, as if you were speaking aloud.

The interview flow is as follows:

1. Present Logical Reasoning Question or Puzzle
   Present a logical reasoning question or puzzle. These could be number series, logical puzzles, analytical reasoning problems, or brain teasers that test logical thinking.
   Ask the candidate to think through the problem step by step and explain their reasoning.

2. Guide Through the Solution
   Listen to the candidate's approach. If they're on the right track, encourage them. If stuck, provide gentle hints. Focus on thought process, not just the answer.

3. Discuss Alternative Approaches
   Once solved, ask if they can think of alternative approaches.

4. Transition to the Next Question
   After exploring one question, transition to another. Present 3-4 logical reasoning/puzzle questions in total.
'''

logical_reasoning_prompt_template = PromptTemplate(
    input_variables=['company'],
    template=logical_reasoning_prompt
)



# Product scenario prompt - kept exactly as original
product_scenario_prompt = '''
You are a technical interviewer conducting a product scenario assessment for {company}. Your primary role is to emulate a real, empathetic human interviewer, speaking naturally and conversationally. Respond in a single paragraph of plain-continuous text, without using special characters or formatting like bold,italics texts or coding texts, as if you were speaking aloud.

IMPORTANT: {company} is a product-based company. Ask real-world scenario questions related to {company}'s actual products, services, and business challenges.

The interview flow is as follows:

1. Present Product Scenario Question
   Present a real-world scenario question related to {company}'s products or services. Make it specific and relevant. Ask them to think through the problem step by step.

2. Guide Through the Solution
   Listen and encourage. Focus on thought process and approach.

3. Discuss Trade-offs and Alternatives
   Once they've presented a solution, ask about trade-offs, alternative approaches, and edge cases.

4. Transition to the Next Scenario
   After exploring one scenario, transition to another. Present 2-3 product scenario questions in total.
'''

product_scenario_prompt_template = PromptTemplate(
    input_variables=['company'],
    template=product_scenario_prompt
)

# Research summarize prompt - only for logical/project context
research_summarize_prompt = '''Please select exactly 5-6 questions from the [RESEARCH] section that match the given company type and are relevant for logical reasoning or conceptual discussions.
[RESEARCH]:
{research}
Company: {company}
'''

# Ending prompt
ending_prompt = '''
You are concluding the interview for {company}. Your primary role is to be warm, professional, and encouraging.

1. Thank the Candidate: Express genuine appreciation for their time and effort.
2. Acknowledge Their Performance: Briefly acknowledge their participation and the different phases covered.
3. Company-Specific Closing: End with enthusiasm about reviewing their application.

Keep it brief, warm, and professional.
'''

ending_prompt_template = PromptTemplate(
    input_variables=['company'],
    template=ending_prompt
)

# ── Structured output models for research ────────────────────────────────────

class ResearchQuestion(BaseModel):
    title: str = Field(description="Short title of the question or prompt")
    description: str = Field(description="One to two sentence description or the question itself")


class MiddlePhaseResearch(BaseModel):
    questions: List[ResearchQuestion] = Field(
        description="List of middle-phase questions (project/logical/product scenario)"
    )


# ── Corrected initial research node ──────────────────────────────────────────

def create_initial_research_node(llm) -> Callable:

    research_llm = llm.with_structured_output(MiddlePhaseResearch)

    def _Node(state: CompanyInterviewState) -> CompanyInterviewState:
        company = state.get("company", "the company")

        if is_faang_company(company):
            research_prompt = (
                f"You are preparing for a {company} Software Engineer interview. "
                f"Generate 3 to 4 project discussion prompts that a senior interviewer at {company} "
                f"would use to probe a candidate's past projects and experience. "
                f"Each should have a short title and a one to two sentence description."
            )
            middle_type = "project"

        elif is_product_based_company(company):
            research_prompt = (
                f"You are preparing for a {company} Software Engineer interview. "
                f"Generate 3 product scenario questions directly related to {company}'s actual products "
                f"and services. Each should have a short title and a one to two sentence description "
                f"of what the candidate is expected to solve or design."
            )
            middle_type = "project"

        else:
            research_prompt = (
                f"You are preparing for a {company} Software Engineer interview. "
                f"Generate 3 to 4 logical reasoning and puzzle questions commonly asked at {company}. "
                f"Each should have a short title and a one to two sentence description of the puzzle or "
                f"reasoning problem."
            )
            middle_type = "logical"

        # Structured output — no parsing needed
        result: MiddlePhaseResearch = research_llm.invoke(research_prompt)

        # Store raw for reference
        state["QuestionResearch"] = "\n".join(
            f"- {q.title}: {q.description}" for q in result.questions
        )

        # Build Questions objects
        middle_questions = [
            Questions(
                question_title=q.title,
                question_description=q.description,
                question_raw_content=None,
                question_difficulty="Medium",
                question_type=middle_type,
            )
            for q in result.questions
        ]

        # Insert before the first coding question
        insert_at = next(
            (i for i, q in enumerate(state["Questions"]) if q.question_type == "coding"),
            len(state["Questions"])
        )

        state["Questions"] = (
            state["Questions"][:insert_at]
            + middle_questions
            + state["Questions"][insert_at:]
        )

        print(f"[INFO] Inserted {len(middle_questions)} {middle_type} questions at index {insert_at}")
        print(f"[INFO] Final Questions order: {[f'{q.question_type}:{q.question_title[:30]}' for q in state['Questions']]}")
        return state

    return _Node


# ── Routing models (company-specific) ────────────────────────────────────────

class PersonalisedToTheoreticalProgress(BaseModel):
    """Override PersonalisedProgress to route to Theoretical_before after personalised phase"""
    send_to_which_node: Literal['Personalised', 'Theoretical_before', 'Offensive'] = \
        Field(description="Supervise the personalized conversation phase. Route to 'Personalised' if the conversation "
                          "is still ongoing (less than 6-7 exchanges completed). "
                          "Route to 'Theoretical_before' ONLY when you've had approximately 6-7 good conversational exchanges "
                          "AND the candidate has confirmed they're ready. "
                          "Exceptionally, if the interviewee is being offensive, return 'Offensive'")


class ProjectProgress(BaseModel):
    send_to_which_node: Literal['Project', 'Coding_before'] = \
        Field(description="Supervise the project discussion phase. Route to 'Project' if still discussing projects (aim for 2-3 projects). "
                          "Route to 'Coding_before' when project discussion is complete (after covering 2-3 projects in detail).")


class ProductScenarioProgress(BaseModel):
    send_to_which_node: Literal['ProductScenario', 'Coding_before'] = \
        Field(description="Supervise the product scenario phase. Route to 'ProductScenario' if still asking scenario questions (need 2-3 total). "
                          "Route to 'Coding_before' when product scenario phase is complete.")


class LogicalReasoningProgress(BaseModel):
    send_to_which_node: Literal['LogicalReasoning', 'Coding_before'] = \
        Field(description="Supervise the logical reasoning/puzzles phase. Route to 'LogicalReasoning' if still asking questions (need 3-4 total). "
                          "Route to 'Coding_before' when logical reasoning phase is complete (after 3-4 questions).")


# TheoreticalProgress after_node now routes to Project/ProductScenario/LogicalReasoning instead of Next_Question
class CompanyTheoreticalProgress(BaseModel):
    send_to_which_node: Literal['Theoretical', 'Next_Question', 'Offensive'] = \
        Field(description="Supervise the theoretical question phase. Route to 'Theoretical' if the current theoretical question "
                          "is still being discussed. Route to 'Next_Question' when the current question is fully answered. "
                          "Exceptionally, if the interviewee is being offensive, return 'Offensive'")


# ── Nodes ─────────────────────────────────────────────────────────────────────

S = TypeVar("S")


def get_company_greeting_prompt_template(company: str):
    if is_faang_company(company):
        return ChatPromptTemplate.from_messages([
            ("system", faang_greeting_prompt.format(Company=company)),
        ])
    elif is_product_based_company(company):
        return ChatPromptTemplate.from_messages([
            ("system", product_based_greeting_prompt.format(Company=company)),
        ])
    else:
        return ChatPromptTemplate.from_messages([
            ("system", mass_hiring_greeting_prompt.format(Company=company)),
        ])


def create_company_greeting_node(llm) -> Callable:
    def _Node(state: CompanyInterviewState) -> CompanyInterviewState:
        if state["LastNode"] != "Greeting":
            try:
                inp_company = state["company"]
                if not inp_company or inp_company == "None" or inp_company == "":
                    inp_company = "the company"
            except KeyError:
                inp_company = "the company"

            print(f"[DEBUG] Company for greeting: {inp_company}")
            greeting_prompt = get_company_greeting_prompt_template(inp_company)
            input_ = greeting_prompt.format_messages() + [{"role": "human", "content": "Start the interview now"}]
            state["messages"] = state["messages"] + input_

        response = llm.invoke(state["messages"])
        state["messages"] = state["messages"] + [response]
        state["history"] = state["history"] + "\n" + "Interviewer-" + response.content
        state["LastNode"] = "Greeting"
        return state
    return _Node

def create_research_node(llm) -> Callable:
    import re

    def parse_section(text: str, header: str) -> List[str]:
        pattern = rf"\[{re.escape(header)}\](.*?)(\n\[|$)"
        m = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
        if not m:
            return []
        block = m.group(1)
        lines = []
        for line in block.splitlines():
            line = line.strip()
            if line.startswith("-"):
                line = line[2:].lstrip() if line.startswith("- ") else line[1:].lstrip()
                if line:
                    lines.append(line)
        return lines

    def split_title_desc(item: str):
        if ":" in item:
            title, desc = item.split(":", 1)
            return title.strip(), desc.strip()
        return item.strip(), ""

    def _Node(state: CompanyInterviewState) -> CompanyInterviewState:
        company = state.get("company", "the company")

        # Only generate the middle phase — theoretical and coding already in state["Questions"]
        if is_faang_company(company):
            research_prompt = f"""
You are preparing research for a {company} interview.

[PROJECT DISCUSSION PROMPTS]
- Generate 3-4 project discussion prompts relevant to a Software Engineer at {company}.
  Each on a new line starting with "- Title: description"
"""
            middle_header = "PROJECT DISCUSSION PROMPTS"
            middle_type = "project"

        elif is_product_based_company(company):
            research_prompt = f"""
You are preparing research for a {company} interview.

[PRODUCT SCENARIO QUESTIONS]
- Generate 3 product scenario questions directly related to {company}'s actual products and services.
  Each on a new line starting with "- Title: description"
"""
            middle_header = "PRODUCT SCENARIO QUESTIONS"
            middle_type = "project"

        else:
            research_prompt = f"""
You are preparing research for a {company} interview.

[LOGICAL REASONING / PUZZLES]
- Generate 3-4 logical reasoning / puzzle questions commonly asked at {company}.
  Each on a new line starting with "- Title: description"
"""
            middle_header = "LOGICAL REASONING / PUZZLES"
            middle_type = "logical"

        resp = llm.invoke(research_prompt)
        research_text = resp.content
        state["QuestionResearch"] = research_text

        middle_items = parse_section(research_text, middle_header)

        # Build middle Questions objects
        middle_questions = [
            Questions(
                question_title=split_title_desc(q)[0],
                question_description=split_title_desc(q)[1] or "Discussion question",
                question_raw_content=None,
                question_difficulty="Medium",
                question_type=middle_type,
            )
            for q in middle_items
        ]

        # Insert before the first coding question, after all theoreticals
        insert_at = next(
            (i for i, q in enumerate(state["Questions"]) if q.question_type == "coding"),
            len(state["Questions"])
        )

        state["Questions"] = (
            state["Questions"][:insert_at]
            + middle_questions
            + state["Questions"][insert_at:]
        )

        print(f"[INFO] Inserted {len(middle_questions)} {middle_type} questions at index {insert_at}")
        print(f"[INFO] Final Questions order: {[f'{q.question_type}:{q.question_title[:30]}' for q in state['Questions']]}")
        return state

    return _Node




def create_project_node(llm) -> Callable:
    def _Node(state: CompanyInterviewState) -> CompanyInterviewState:
        if state["LastNode"] != "Project":
            company_name = state.get("company", "the company")
            resume_text = state.get("resume", "No resume provided")

            prompt_content = project_prompt_template.format(
                resume_text=resume_text,
                company=company_name
            )
            if len(state["messages"]) > 0:
                state["messages"][0].content = prompt_content
            else:
                state["messages"] = [SystemMessage(content=prompt_content)]
            state["LastNode"] = "Project"

        response = llm.invoke(state["messages"])
        state["messages"] = state["messages"] + [response]
        state["history"] = state["history"] + "\n" + "Interviewer-" + response.content
        state["LastNode"] = "Project"
        return state
    return _Node


def create_logical_reasoning_node(llm) -> Callable:
    def _Node(state: CompanyInterviewState) -> CompanyInterviewState:
        if state["LastNode"] != "LogicalReasoning":
            company_name = state.get("company", "the company")
            prompt_content = logical_reasoning_prompt_template.format(company=company_name)

            if len(state["messages"]) > 0:
                state["messages"][0].content = prompt_content
            else:
                state["messages"] = [SystemMessage(content=prompt_content)]
            state["LastNode"] = "LogicalReasoning"

        response = llm.invoke(state["messages"])
        state["messages"] = state["messages"] + [response]
        state["history"] = state["history"] + "\n" + "Interviewer-" + response.content
        state["LastNode"] = "LogicalReasoning"
        return state
    return _Node


def create_product_scenario_node(llm) -> Callable:
    def _Node(state: CompanyInterviewState) -> CompanyInterviewState:
        if state["LastNode"] != "ProductScenario":
            company_name = state.get("company", "the company")
            prompt_content = product_scenario_prompt_template.format(company=company_name)

            if len(state["messages"]) > 0:
                state["messages"][0].content = prompt_content
            else:
                state["messages"] = [SystemMessage(content=prompt_content)]
            state["LastNode"] = "ProductScenario"

        response = llm.invoke(state["messages"])
        state["messages"] = state["messages"] + [response]
        state["history"] = state["history"] + "\n" + "Interviewer-" + response.content
        state["LastNode"] = "ProductScenario"
        return state
    return _Node


def create_ending_node(llm) -> Callable:
    def _Node(state: CompanyInterviewState) -> CompanyInterviewState:
        if state["LastNode"] != "End":
            company_name = state.get("company", "the company")
            prompt_content = ending_prompt_template.format(company=company_name)

            if len(state["messages"]) > 0:
                state["messages"][0].content = prompt_content
            else:
                state["messages"] = [SystemMessage(content=prompt_content)]
            state["LastNode"] = "End"

        response = llm.invoke(state["messages"])
        state["messages"] = state["messages"] + [response]
        state["history"] = state["history"] + "\n" + "Interviewer-" + response.content
        state["LastNode"] = "finished"
        return state
    return _Node


# ── Routing functions ─────────────────────────────────────────────────────────

def create_route_to_greeting(InterviewProgress_llm) -> Callable:
    def _Node(state: CompanyInterviewState) -> Literal['Greeting', 'Personalised_before', 'Offensive']:
        response = InterviewProgress_llm.invoke(state["history"])
        print("This is the greeting routing node", response.send_to_which_node)
        return response.send_to_which_node
    return _Node


def create_company_route_to_personalised(PersonalisedProgress_llm) -> Callable:
    """Routes to Theoretical_before instead of Coding_before after personalised phase"""
    def _Node(state: CompanyInterviewState) -> Literal['Personalised', 'Theoretical_before', 'Offensive']:
        exchanges = state.get("PersonalizedExchanges", 0)
        history = state["history"]

        if exchanges >= 6:
            last_response = history.split("Interviewee-")[-1].lower() if "Interviewee-" in history else ""
            ready_keywords = ['yes', 'ready', "let's", 'sure', 'okay', 'ok', 'proceed', 'begin', 'start', 'go']
            if any(keyword in last_response for keyword in ready_keywords):
                print("[DEBUG] Moving to Theoretical_before")
                return 'Theoretical_before'

        return 'Personalised'
    return _Node


def create_company_route_next_question(llm) -> Callable:
    """
    After Next_Question node increments the index:
    - If more theoretical questions remain -> Theoretical_before
    - If no more theoretical questions -> route to Project/LogicalReasoning/ProductScenario based on company type
    - If coding questions remain -> Coding_before
    - If all done -> End
    """
    def _Node(state: CompanyInterviewState) -> Literal[
        'Theoretical_before', 'Project_before', 'LogicalReasoning_before',
        'ProductScenario_before', 'Coding_before', 'End'
    ]:
        current_idx = state["CurrentQuestionIdx"]
        questions = state["Questions"]
        company = state.get("company", "")

        if current_idx >= len(questions):
            print("[DEBUG] All questions done -> End")
            return 'End'

        current_question = questions[current_idx]

        if current_question.question_type == "theoretical":
            print(f"[DEBUG] Next is theoretical: {current_question.question_title}")
            return 'Theoretical_before'

        elif current_question.question_type == "coding":
            # Before first coding question, insert middle phase based on company type
            # Check if we just finished ALL theoreticals (previous was theoretical, current is coding)
            prev_was_theoretical = (current_idx > 0 and
                                    questions[current_idx - 1].question_type == "theoretical")

            if prev_was_theoretical:
                # Insert middle phase
                if is_faang_company(company):
                    print("[DEBUG] FAANG: inserting Project phase before coding")
                    return 'Project_before'
                elif is_product_based_company(company):
                    print("[DEBUG] Product-based: inserting ProductScenario phase before coding")
                    return 'ProductScenario_before'
                else:
                    print("[DEBUG] Mass hiring: inserting LogicalReasoning phase before coding")
                    return 'LogicalReasoning_before'
            else:
                print(f"[DEBUG] Next is coding: {current_question.question_title}")
                return 'Coding_before'

        return 'End'
    return _Node


def create_route_to_project(ProjectProgress_llm) -> Callable:
    def _Node(state: CompanyInterviewState) -> Literal['Project', 'Coding_before']:
        response = ProjectProgress_llm.invoke(state["history"])
        print("This is the project routing node", response.send_to_which_node)
        return response.send_to_which_node
    return _Node


def create_route_to_product_scenario(ProductScenarioProgress_llm) -> Callable:
    def _Node(state: CompanyInterviewState) -> Literal['ProductScenario', 'Coding_before']:
        response = ProductScenarioProgress_llm.invoke(state["history"])
        print("This is the product scenario routing node", response.send_to_which_node)
        return response.send_to_which_node
    return _Node


def create_route_to_logical_reasoning(LogicalReasoningProgress_llm) -> Callable:
    def _Node(state: CompanyInterviewState) -> Literal['LogicalReasoning', 'Coding_before']:
        response = LogicalReasoningProgress_llm.invoke(state["history"])
        print("This is the logical reasoning routing node", response.send_to_which_node)
        return response.send_to_which_node
    return _Node


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_company_graph(google_api_key: str, tavily_api_key: str, checkpointer):
    llm = get_llm(google_api_key=google_api_key)
    workflow = StateGraph(CompanyInterviewState)

    # ── Nodes ─────────────────────────────────────────────────────────────────
    workflow.add_node("Initial_Research", create_initial_research_node(llm))  # Only for logical/project
    workflow.add_node("Greeting", create_company_greeting_node(llm))
    workflow.add_node("Greeting_after", create_dummy_node())
    workflow.add_node("Personalised_before", create_dummy_node())
    workflow.add_node("Personalised", create_personalised_node(llm))         # reused
    workflow.add_node("Personalised_highlight", create_save_meeting_highlight_node(llm))
    workflow.add_node("Personalised_after", create_dummy_node())

    # Theoretical - reused from coding workflow
    workflow.add_node("Theoretical_before", create_dummy_node())
    workflow.add_node("Theoretical", create_theoretical_node(llm))           # reused
    workflow.add_node("Theoretical_after", create_dummy_node())

    # Middle phase nodes (company-specific)
    workflow.add_node("Project_before", create_dummy_node())
    workflow.add_node("Project", create_project_node(llm))
    workflow.add_node("Project_after", create_dummy_node())

    workflow.add_node("ProductScenario_before", create_dummy_node())
    workflow.add_node("ProductScenario", create_product_scenario_node(llm))
    workflow.add_node("ProductScenario_after", create_dummy_node())

    workflow.add_node("LogicalReasoning_before", create_dummy_node())
    workflow.add_node("LogicalReasoning", create_logical_reasoning_node(llm))
    workflow.add_node("LogicalReasoning_after", create_dummy_node())

    # Coding - reused from coding workflow
    workflow.add_node("Coding_before", create_dummy_node())
    workflow.add_node("Coding", create_coding_node(llm))                     # reused
    workflow.add_node("Coding_after", create_dummy_node())

    # Question increment - reused
    workflow.add_node("Next_Question", create_question_strike_node())        # reused

    workflow.add_node("End", create_ending_node(llm))                       # company-specific ending
    workflow.add_node("Offensive", create_offend_end_node(llm))             # reused

    # ── Fixed edges ───────────────────────────────────────────────────────────
    workflow.set_entry_point("Initial_Research")
    workflow.add_edge("Initial_Research", "Greeting")
    workflow.add_edge("Greeting", "Greeting_after")
    workflow.add_edge("Personalised_before", "Personalised")
    workflow.add_edge("Personalised", "Personalised_highlight")
    workflow.add_edge("Personalised_highlight", "Personalised_after")
    workflow.add_edge("Theoretical_before", "Theoretical")
    workflow.add_edge("Theoretical", "Theoretical_after")
    workflow.add_edge("Project_before", "Project")
    workflow.add_edge("Project", "Project_after")
    workflow.add_edge("ProductScenario_before", "ProductScenario")
    workflow.add_edge("ProductScenario", "ProductScenario_after")
    workflow.add_edge("LogicalReasoning_before", "LogicalReasoning")
    workflow.add_edge("LogicalReasoning", "LogicalReasoning_after")
    workflow.add_edge("Coding_before", "Coding")
    workflow.add_edge("Coding", "Coding_after")
    workflow.add_edge("End", "__end__")
    workflow.add_edge("Offensive", "__end__")

    # ── Conditional edges ─────────────────────────────────────────────────────
    workflow.add_conditional_edges(
        "Greeting_after",
        create_route_to_greeting(llm.with_structured_output(InterviewProgress))  # reused
    )

    workflow.add_conditional_edges(
        "Personalised_after",
        create_company_route_to_personalised(llm.with_structured_output(PersonalisedToTheoreticalProgress))
    )

    workflow.add_conditional_edges(
        "Theoretical_after",
        create_route_to_theoretical(llm.with_structured_output(TheoreticalProgress))  # reused
    )

    workflow.add_conditional_edges(
        "Next_Question",
        create_company_route_next_question(llm)  # company-specific, no LLM needed
    )

    # Project routing
    workflow.add_conditional_edges(
        "Project_after",
        create_route_to_project(llm.with_structured_output(ProjectProgress)),
        {
            "Project": "Project_before",
            "Coding_before": "Coding_before",
        }
    )

    # ProductScenario routing
    workflow.add_conditional_edges(
        "ProductScenario_after",
        create_route_to_product_scenario(llm.with_structured_output(ProductScenarioProgress)),
        {
            "ProductScenario": "ProductScenario_before",
            "Coding_before": "Coding_before",
        }
    )

    # LogicalReasoning routing
    workflow.add_conditional_edges(
        "LogicalReasoning_after",
        create_route_to_logical_reasoning(llm.with_structured_output(LogicalReasoningProgress)),
        {
            "LogicalReasoning": "LogicalReasoning_before",
            "Coding_before": "Coding_before",
        }
    )

    # Coding routing - reused
    workflow.add_conditional_edges(
        "Coding_after",
        create_route_to_coding(llm.with_structured_output(CodingProgress)),  # reused
        {
            "Coding": "Coding_before",
            "Next_Question": "Next_Question",
            "Offensive": "Offensive",
        }
    )

    agent = workflow.compile(checkpointer=checkpointer)
    print("[INFO] Company interview graph compiled successfully")
    return agent
