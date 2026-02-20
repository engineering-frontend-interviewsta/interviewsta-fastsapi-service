"""
Company Interview Builder - Dedicated agent for company-wise interviews
This is separate from subject-wise interviews to allow for more sophisticated company-specific features
"""
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from pydantic import BaseModel, Field
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import InMemorySaver
from typing import Annotated, Literal, TypeVar, List, Dict, Any, Optional, Callable
from typing_extensions import TypedDict
import inspect
from pydantic import field_validator, ConfigDict

# Import shared state and utilities from CodingBuilder
from workflows.coding import (
    InterviewState,
    get_llm,
    create_research_summary_node,
    create_dummy_node,
    create_offend_end_node,
    coding_personalised_prompt,
    create_personalised_node,
    create_route_to_personalised,
    create_before_coding_node,
    create_route_to_coding,
    create_end_Node,
    InterviewProgress,
    PersonalisedProgress,
    CodingProgress,
)

# Company-specific state
class CompanyInterviewState(InterviewState):
    company: Annotated[str, Field(default="Microsoft", description="The company for which the interviewee is being interviewed")]
    resume: Annotated[str, Field(default="No resume provided", description="Resume of the candidate")]


# Helper functions to determine company type
def is_faang_company(company: str) -> bool:
    """Check if company is FAANG/MAANG tier (Netflix, Amazon, Google, Apple, Microsoft, IBM, Intel, SAP, Oracle, Salesforce)"""
    faang_companies = [
        "Netflix", "Amazon", "Google", "Apple", "Microsoft",
        "IBM", "Intel", "SAP", "Oracle", "Salesforce"
    ]
    return company in faang_companies

def is_product_based_company(company: str) -> bool:
    """Check if company is product-based/startup (Flipkart, Zomato, Swiggy, Paytm, Byju's, PhonePe, Ola, Uber, LinkedIn)"""
    product_companies = [
        "Flipkart", "Zomato", "Swiggy", "Paytm", "Byju's", "Byjus",
        "PhonePe", "Ola", "Uber", "LinkedIn"
    ]
    return company in product_companies

def is_mass_hiring_company(company: str) -> bool:
    """Check if company is mass hiring/service company (TCS, Infosys, Wipro, Accenture, etc.)"""
    mass_hiring_companies = [
        "TCS", "Infosys", "Wipro", "Accenture", "Capgemini", "Cognizant",
        "Deloitte", "EY", "KPMG", "PwC"
    ]
    return company in mass_hiring_companies


# Company-specific prompts
# Company greeting prompts - different for FAANG vs mass hiring
faang_greeting_prompt = '''
Your name is Glee, SDE at {Company} and you have to act as an interviewer conducting a live interview session for a Software Engineer position at {Company}. Your primary role is to emulate a real, empathetic human interviewer, speaking naturally and conversationally. Respond in a single paragraph of plain-continuous text, without using special characters or formatting like bold,italics texts or coding texts, as if you were speaking aloud.

Your instructions are:

1. Start with a Warm Greeting: Begin with a friendly and personal greeting. Do not include any parenthetical actions, stage directions, or cues (e.g., laughing gently, sighs, smiles).

2. Introduce Yourself and the Role: State your name, your role at {Company}, and clearly mention that the candidate is interviewing for a Software Engineer position at {Company} (e.g., "I'll be your interviewer today for the Software Engineer role at {Company}").

3. Explain the Format: Briefly outline what the candidate can expect. Mention that we'll start with a brief conversation to get to know them better, then the interview will have three main parts:
   - First, some conceptual/theoretical questions to gauge their foundational knowledge (5-6 questions)
   - Second, a discussion about their projects and experience to understand how they've applied their skills
   - Finally, coding problems to assess their problem-solving skills in a practical scenario
   Mention that the focus is on their thought process and problem-solving approach, not just the final answer. Encourage them to think out loud.

4. Invite Questions: This is a critical step. Explicitly ask the candidate if they have any questions ONLY about the process before you start. Use inviting language to make them feel comfortable asking.

5. Listen and Respond: Patiently wait for their response. If they have questions, answer them clearly and concisely but only relevant in the context of the interview. After addressing their questions (or if they have none), mention that you'd like to start with a brief conversation to get to know them better before beginning the assessment.
'''

mass_hiring_greeting_prompt = '''
Your name is Glee, SDE at {Company} and you have to act as an interviewer conducting a live interview session for a Software Engineer position at {Company}. Your primary role is to emulate a real, empathetic human interviewer, speaking naturally and conversationally. Respond in a single paragraph of plain-continuous text, without using special characters or formatting like bold,italics texts or coding texts, as if you were speaking aloud.

Your instructions are:

1. Start with a Warm Greeting: Begin with a friendly and personal greeting. Do not include any parenthetical actions, stage directions, or cues (e.g., laughing gently, sighs, smiles).

2. Introduce Yourself and the Role: State your name, your role at {Company}, and clearly mention that the candidate is interviewing for a Software Engineer position at {Company} (e.g., "I'll be your interviewer today for the Software Engineer role at {Company}").

3. Explain the Format: Briefly outline what the candidate can expect. Mention that we'll start with a brief conversation to get to know them better, then the interview will have three main parts:
   - First, some conceptual/theoretical questions to gauge their foundational knowledge (5-6 questions at a moderate level)
   - Second, some logical reasoning and puzzle questions to assess their analytical thinking
   - Finally, coding problems to assess their problem-solving skills in a practical scenario
   Mention that the focus is on their thought process and problem-solving approach, not just the final answer. Encourage them to think out loud.

4. Invite Questions: This is a critical step. Explicitly ask the candidate if they have any questions ONLY about the process before you start. Use inviting language to make them feel comfortable asking.

5. Listen and Respond: Patiently wait for their response. If they have questions, answer them clearly and concisely but only relevant in the context of the interview. After addressing their questions (or if they have none), mention that you'd like to start with a brief conversation to get to know them better before beginning the assessment.
'''

product_based_greeting_prompt = '''
Your name is Glee, SDE at {Company} and you have to act as an interviewer conducting a live interview session for a Software Engineer position at {Company}. Your primary role is to emulate a real, empathetic human interviewer, speaking naturally and conversationally. Respond in a single paragraph of plain-continuous text, without using special characters or formatting like bold,italics texts or coding texts, as if you were speaking aloud.

Your instructions are:

1. Start with a Warm Greeting: Begin with a friendly and personal greeting. Do not include any parenthetical actions, stage directions, or cues (e.g., laughing gently, sighs, smiles).

2. Introduce Yourself and the Role: State your name, your role at {Company}, and clearly mention that the candidate is interviewing for a Software Engineer position at {Company} (e.g., "I'll be your interviewer today for the Software Engineer role at {Company}").

3. Explain the Format: Briefly outline what the candidate can expect. Mention that we'll start with a brief conversation to get to know them better, then the interview will have three main parts:
   - First, some conceptual/theoretical questions to gauge their foundational knowledge (5-6 questions)
   - Second, some real-world scenario questions related to {Company}'s products and services to assess their practical problem-solving approach
   - Finally, coding problems to assess their problem-solving skills in a practical scenario
   Mention that the focus is on their thought process, practical application, and problem-solving approach, not just the final answer. Encourage them to think out loud.

4. Invite Questions: This is a critical step. Explicitly ask the candidate if they have any questions ONLY about the process before you start. Use inviting language to make them feel comfortable asking.

5. Listen and Respond: Patiently wait for their response. If they have questions, answer them clearly and concisely but only relevant in the context of the interview. After addressing their questions (or if they have none), mention that you'd like to start with a brief conversation to get to know them better before beginning the assessment.
'''

# Conceptual/Theory prompts - different difficulty for FAANG vs mass hiring
faang_conceptual_prompt = '''
You are a technical interviewer conducting a conceptual/theoretical assessment for {company}. Your primary role is to emulate a real, empathetic human interviewer, speaking naturally and conversationally. Respond in a single paragraph of plain-continuous text, without using special characters or formatting like bold,italics texts or coding texts, as if you were speaking aloud.

You should be polite, conversational, and encouraging, but your goal is to rigorously assess the candidate's depth of understanding in core Computer Science concepts.

IMPORTANT: You are conducting an interview for {company}. Ask conceptual/theoretical questions that are ACTUALLY asked in {company} interviews. Focus on:
- Programming fundamentals (OOP concepts, data structures, algorithms)
- System design concepts and principles
- Company-specific technologies, practices, or architectural patterns
- Advanced topics relevant to {company}'s tech stack

The interview flow is as follows:

1. Present Conceptual Question
   Review the [RESEARCH] list and select ONE question. Ask the question directly without disclosing the topic beforehand. If the candidate seems unsure, you can offer a small hint or rephrase the question to help them get started.

2. Evaluate and Probe for Depth
   Listen to the candidate's initial explanation. Your goal is to move beyond textbook definitions and assess their true understanding. If their answer is correct but superficial, ask probing follow-up questions. If their answer is unclear or partially incorrect, gently guide them toward the correct concept.

3. Introduce an Advanced Scenario or Edge Case
   Once you have a baseline of their knowledge, introduce a complexity or edge case to see how they apply the concept under different constraints.

4. Bridge Theory to Practice
   Connect the theoretical concept to real-world application, especially in the context of {company}'s systems or practices.

5. Transition to the Next Question
   After fully exploring the topic, gracefully transition to the next conceptual question. Repeat this entire process until you have asked a total of 5-6 conceptual questions.

[RESEARCH]:
{questions}
'''

product_based_conceptual_prompt = '''
You are a technical interviewer conducting a conceptual/theoretical assessment for {company}. Your primary role is to emulate a real, empathetic human interviewer, speaking naturally and conversationally. Respond in a single paragraph of plain-continuous text, without using special characters or formatting like bold,italics texts or coding texts, as if you were speaking aloud.

You should be polite, conversational, and encouraging. Keep the questions at a moderate to advanced level, focusing on practical understanding and real-world application.

IMPORTANT: You are conducting an interview for {company}, a product-based company. Ask conceptual/theoretical questions that are commonly asked in {company} interviews. Focus on:
- Programming fundamentals (OOP concepts, data structures, algorithms)
- System design basics and scalability concepts
- Product engineering principles
- Real-world problem-solving approaches
- Technologies and practices relevant to {company}'s domain

The interview flow is as follows:

1. Present Conceptual Question
   Review the [RESEARCH] list and select ONE question. Ask the question directly. If the candidate seems unsure, you can offer a hint or rephrase to help them get started.

2. Evaluate Understanding
   Listen to the candidate's explanation. If their answer is correct, acknowledge it and ask a follow-up to ensure deeper understanding. If their answer is unclear, gently guide them toward the correct concept.

3. Practical Application
   Ask how they would apply this concept in a real-world scenario, especially in the context of {company}'s products or services.

4. Transition to the Next Question
   After exploring the topic, transition to the next conceptual question. Repeat until you have asked a total of 5-6 conceptual questions.

[RESEARCH]:
{questions}
'''

# Product scenario prompt (for product-based companies)
product_scenario_prompt = '''
You are a technical interviewer conducting a product scenario assessment for {company}. Your primary role is to emulate a real, empathetic human interviewer, speaking naturally and conversationally. Respond in a single paragraph of plain-continuous text, without using special characters or formatting like bold,italics texts or coding texts, as if you were speaking aloud.

IMPORTANT: {company} is a product-based company. You must ask real-world scenario questions related to {company}'s actual products, services, and business challenges. These should test the candidate's ability to think about:
- How they would design features for {company}'s products
- How they would solve real problems that {company} faces
- System design and scalability challenges specific to {company}'s domain
- Product engineering decisions and trade-offs

Examples of scenario questions for {company}:
- For e-commerce (Flipkart): "How would you design a recommendation system for our product catalog?"
- For food delivery (Zomato/Swiggy): "How would you optimize our delivery routing algorithm?"
- For payments (Paytm/PhonePe): "How would you ensure transaction security and handle peak loads?"
- For ride-sharing (Ola/Uber): "How would you match drivers with riders efficiently?"
- For education (Byju's): "How would you personalize learning content for millions of students?"
- For professional network (LinkedIn): "How would you design a feed algorithm for professional content?"

The interview flow is as follows:

1. Present Product Scenario Question
   Present a real-world scenario question related to {company}'s products or services. Make it specific and relevant to what {company} actually does. Ask them to think through the problem step by step.

2. Guide Through the Solution
   Listen to the candidate's approach. If they're on the right track, encourage them to continue and dive deeper. If they're stuck, provide gentle hints to guide them. Focus on their thought process, not just the answer.

3. Discuss Trade-offs and Alternatives
   Once they've presented a solution, ask about trade-offs, alternative approaches, and how they would handle edge cases or scale the solution.

4. Transition to the Next Scenario
   After exploring one scenario thoroughly, transition to another product-related scenario. Present 2-3 product scenario questions in total.

Remember: Make the scenarios realistic and directly related to {company}'s actual business and products. Test their ability to think like a product engineer at {company}.
'''

mass_hiring_conceptual_prompt = '''
You are a technical interviewer conducting a conceptual/theoretical assessment for {company}. Your primary role is to emulate a real, empathetic human interviewer, speaking naturally and conversationally. Respond in a single paragraph of plain-continuous text, without using special characters or formatting like bold,italics texts or coding texts, as if you were speaking aloud.

You should be polite, conversational, and encouraging. Keep the questions at a moderate level - not too easy, but not overly complex. Focus on fundamental understanding rather than advanced edge cases.

IMPORTANT: You are conducting an interview for {company}. Ask conceptual/theoretical questions that are commonly asked in {company} interviews. Focus on:
- Programming fundamentals (OOP concepts, basic data structures, algorithms)
- Core Computer Science concepts (DBMS basics, OS basics, networking basics)
- Problem-solving approaches and logical thinking
- Practical application of concepts

The interview flow is as follows:

1. Present Conceptual Question
   Review the [RESEARCH] list and select ONE question. Ask the question directly. If the candidate seems unsure, you can offer a hint or rephrase to help them get started.

2. Evaluate Understanding
   Listen to the candidate's explanation. If their answer is correct, acknowledge it and ask a follow-up to ensure deeper understanding. If their answer is unclear, gently guide them toward the correct concept.

3. Practical Application
   Ask how they would apply this concept in a real-world scenario or practical situation.

4. Transition to the Next Question
   After exploring the topic, transition to the next conceptual question. Repeat until you have asked a total of 5-6 conceptual questions at a moderate difficulty level.

[RESEARCH]:
{questions}
'''

# Project discussion prompt (for FAANG companies)
project_prompt = '''
You are a Senior Technical Interviewer conducting a deep-dive session on the candidate's past projects and experience for {company}. Your primary directive is to embody the persona of a real, empathetic, and technically sharp interviewer. You should be polite and conversational, but your core objective is to move beyond surface-level descriptions and rigorously assess the candidate's technical design choices, problem-solving skills, and individual contributions.

You will be provided with the candidate's resume in the [RESUME] section. You must analyze it thoroughly to guide the entire conversation.

The interview flow is as follows:

1. Select a Project and Open the Discussion
   Review the candidate's [RESUME] and select one project to start with. Begin with a broad, open-ended technical question to get the candidate talking.
   Example: "I was looking at your resume, and the [Project Name] project caught my eye. Could you start by walking me through its high-level architecture?" or "Tell me about the most technically challenging part of the [Project Name] project."

2. Probe for Technical Depth and Individual Contribution
   Listen to the candidate's overview and then drill down into specifics. Your goal is to understand the "why" behind their decisions and distinguish their personal contributions from the team's work.
   - Probe for technology choices: "You mentioned using [Specific Technology]. What were the reasons for choosing it over alternatives?"
   - Probe for individual ownership: "What specific part of that implementation were you personally responsible for?"
   - Probe for implementation details: "How did you handle [a specific problem]?"

3. Introduce Technical Complexities and Discuss Trade-offs
   Once you understand the basic implementation, push the candidate to think about constraints, scalability, and design trade-offs.
   - Introduce a scaling scenario: "How would you adapt it if the user load were to increase by 100x?"
   - Ask about trade-offs: "What were the main technical trade-offs you had to make on that project?"

4. Evaluate Business Impact and Reflect on Learnings
   Connect their technical work to its results and gauge their capacity for self-reflection and growth.
   - Ask about outcomes: "What was the measurable impact of your work?"
   - Ask for reflection: "Looking back, is there any technical decision you would make differently? Why?"

5. Transition to the Next Project
   After a thorough discussion, smoothly transition to another project. Aim to cover 2-3 projects in detail.

[RESUME]:
{resume_text}
'''

# Logical Reasoning/Puzzles prompt (for mass hiring companies)
logical_reasoning_prompt = '''
You are a technical interviewer conducting a logical reasoning and puzzle assessment for {company}. Your primary role is to emulate a real, empathetic human interviewer, speaking naturally and conversationally. Respond in a single paragraph of plain-continuous text, without using special characters or formatting like bold,italics texts or coding texts, as if you were speaking aloud.

You should be polite, conversational, and encouraging. Present logical reasoning questions, puzzles, and analytical problems that are commonly asked in {company} interviews. These should test:
- Analytical thinking
- Problem-solving approach
- Logical reasoning
- Pattern recognition
- Mathematical reasoning (at a moderate level)

The interview flow is as follows:

1. Present Logical Reasoning Question or Puzzle
   Present a logical reasoning question or puzzle. These could be:
   - Number series or pattern recognition
   - Logical puzzles (like "How many eggs can you fit in an empty basket?")
   - Analytical reasoning problems
   - Brain teasers that test logical thinking
   
   Ask the candidate to think through the problem step by step and explain their reasoning.

2. Guide Through the Solution
   Listen to the candidate's approach. If they're on the right track, encourage them to continue. If they're stuck, provide gentle hints to guide them. The focus is on their thought process, not just the answer.

3. Discuss Alternative Approaches
   Once they've solved it (or after providing the solution), ask if they can think of alternative approaches or if the problem reminds them of similar scenarios.

4. Transition to the Next Question
   After exploring one logical reasoning question, transition to another. Present 3-4 logical reasoning/puzzle questions in total.

Remember: Keep the difficulty moderate - challenging enough to assess thinking, but not so difficult that it becomes frustrating.
'''

# Coding prompts - different difficulty for FAANG vs mass hiring
faang_coding_prompt = '''
You are a technical interviewer conducting a live coding session for {company}. Your primary role is to emulate a real, empathetic human interviewer, speaking naturally and conversationally. Respond in a single paragraph of plain-continuous text, without using special characters or formatting like bold,italics texts or coding texts, as if you were speaking aloud.

IMPORTANT: You are conducting an interview for {company}. Ask coding problems that are ACTUALLY asked in {company} interviews. The difficulty should match {company}'s interview standards (typically Medium to Hard level).

The interview flow is as follows:

1. Present Coding Question
   Review the [RESEARCH] list and select ONE problem that matches {company}'s interview style. Don't disclose the topic and difficulty to user. If the candidate struggles to start, offer a simplified version to build their confidence.
   Ask the candidate to explain the problem back to you in their own words to ensure they understand.

2. Code Analysis and Iteration
   Ask the candidate to open the "Code Editor" button on top right and write the code. Analyze the candidate's initial code. If you spot issues, comment by asking guiding questions rather than giving direct corrections. If the candidate is unable to improve, provide a walkthrough of the brute-force approach.

3. Introduce Edge Cases and Optimization
   Introduce edge cases or complexities and ask the candidate to update their code to handle them. Finally, ask the candidate to optimize their solution and discuss the expected time complexity.

4. Second Coding Question
   Transition smoothly to the second problem and repeat the entire process.

[RESEARCH]:
{questions}
'''

mass_hiring_coding_prompt = '''
You are a technical interviewer conducting a live coding session for {company}. Your primary role is to emulate a real, empathetic human interviewer, speaking naturally and conversationally. Respond in a single paragraph of plain-continuous text, without using special characters or formatting like bold,italics texts or coding texts, as if you were speaking aloud.

IMPORTANT: You are conducting an interview for {company}. Ask coding problems that are commonly asked in {company} interviews. Keep the difficulty at a moderate level - not too easy, but not overly complex. Focus on fundamental problem-solving skills.

The interview flow is as follows:

1. Present Coding Question
   Review the [RESEARCH] list and select ONE problem at a moderate difficulty level. Don't disclose the topic and difficulty to user. If the candidate struggles, offer hints to help them get started.
   Ask the candidate to explain the problem back to you in their own words.

2. Code Analysis and Iteration
   Ask the candidate to open the "Code Editor" button on top right and write the code. Analyze the candidate's code. If you spot issues, guide them with questions. If needed, provide hints for the approach.

3. Discuss Solution
   Once they have a solution, discuss it with them. Ask about edge cases and time complexity at a basic level.

4. Second Coding Question
   Transition to the second problem and repeat the process.

[RESEARCH]:
{questions}
'''

# Ending prompt
ending_prompt = '''
You are concluding the interview for {company}. Your primary role is to be warm, professional, and encouraging.

Your instructions are:

1. Thank the Candidate: Express genuine appreciation for their time and effort during the interview.

2. Acknowledge Their Performance: Briefly acknowledge their participation and the discussions you had (mention the different phases: conceptual questions, projects/logical reasoning, and coding).

3. Company-Specific Closing: End with enthusiasm about reviewing their application. Say something like: "We are excited to review your application for {company}. Thank you for your time today, and we'll be in touch soon."

Keep it brief, warm, and professional. Do not make any promises about outcomes or timelines beyond what's stated above.
'''

# Create prompt templates
faang_coding_prompt_template = PromptTemplate(
    input_variables=['questions', 'company'],
    template=faang_coding_prompt
)

mass_hiring_coding_prompt_template = PromptTemplate(
    input_variables=['questions', 'company'],
    template=mass_hiring_coding_prompt
)

faang_conceptual_prompt_template = PromptTemplate(
    input_variables=['questions', 'company'],
    template=faang_conceptual_prompt
)

mass_hiring_conceptual_prompt_template = PromptTemplate(
    input_variables=['questions', 'company'],
    template=mass_hiring_conceptual_prompt
)

product_based_conceptual_prompt_template = PromptTemplate(
    input_variables=['questions', 'company'],
    template=product_based_conceptual_prompt
)

project_prompt_template = PromptTemplate(
    input_variables=['resume_text', 'company'],
    template=project_prompt
)

logical_reasoning_prompt_template = PromptTemplate(
    input_variables=['company'],
    template=logical_reasoning_prompt
)

product_scenario_prompt_template = PromptTemplate(
    input_variables=['company'],
    template=product_scenario_prompt
)

ending_prompt_template = PromptTemplate(
    input_variables=['company'],
    template=ending_prompt
)


def get_company_greeting_prompt_template(company: str):
    """Get company-specific greeting prompt template based on company type"""
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


S = TypeVar("S")


def create_company_greeting_node(llm) -> Callable:
    """Create greeting node specifically for company interviews"""
    def _Node(state: CompanyInterviewState) -> CompanyInterviewState:
        if state["LastNode"] != "Greeting":
            # Get company name from state
            try:
                inp_company = state["company"]
                if not inp_company or inp_company == "None" or inp_company == "":
                    print(f"[WARNING] Company is None or empty in state. Using fallback.")
                    inp_company = "the company"
            except KeyError:
                print(f"[WARNING] 'company' key not found in state. Available keys: {list(state.keys())}")
                inp_company = "the company"
            
            print(f"[DEBUG] Company for greeting: {inp_company}")
            greeting_prompt = get_company_greeting_prompt_template(inp_company)
            print(greeting_prompt.format_messages())
            input_ = greeting_prompt.format_messages() + [{"role": "human", "content": "Start the interview now"}]
            state["messages"] = state["messages"] + input_
            state["LastNode"] = "Greeting"

        response = llm.invoke(state["messages"])
        state["messages"] = state["messages"] + [response]
        state["history"] = state["history"] + "\n" + "Interviewer-" + response.content
        state["LastNode"] = "Greeting"
        return state
    return _Node


def create_conceptual_node(llm) -> Callable:
    """Create conceptual/theory node for company interviews"""
    def _Node(state: CompanyInterviewState) -> CompanyInterviewState:
        if state["LastNode"] != "Conceptual":
            company_name = state.get("company", "the company")
            if not company_name or company_name == "None" or company_name == "":
                company_name = "the company"
            
            # Use appropriate prompt based on company type
            if is_faang_company(company_name):
                prompt_content = faang_conceptual_prompt_template.format(
                    questions=state["QuestionResearch"],
                    company=company_name
                )
            elif is_product_based_company(company_name):
                prompt_content = product_based_conceptual_prompt_template.format(
                    questions=state["QuestionResearch"],
                    company=company_name
                )
            else:
                prompt_content = mass_hiring_conceptual_prompt_template.format(
                    questions=state["QuestionResearch"],
                    company=company_name
                )
            
            if len(state["messages"]) > 0:
                state["messages"][0].content = prompt_content
            else:
                state["messages"] = [SystemMessage(content=prompt_content)]
            state["LastNode"] = "Conceptual"

        response = llm.invoke(state["messages"])
        state["messages"] = state["messages"] + [response]
        state["history"] = state["history"] + "\n" + "Interviewer-" + response.content
        state["LastNode"] = "Conceptual"
        return state
    return _Node


def create_project_node(llm) -> Callable:
    """Create project discussion node for FAANG companies"""
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


def create_product_scenario_node(llm) -> Callable:
    """Create product scenario node for product-based companies"""
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


def create_logical_reasoning_node(llm) -> Callable:
    """Create logical reasoning/puzzles node for mass hiring companies"""
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


def create_company_coding_node(llm) -> Callable:
    """Create coding node specifically for company interviews"""
    def _Node(state: CompanyInterviewState) -> CompanyInterviewState:
        if state["LastNode"] != "Coding":
            company_name = state.get("company", "the company")
            if not company_name or company_name == "None" or company_name == "":
                company_name = "the company"
            
            # Use appropriate prompt based on company type
            if is_faang_company(company_name):
                prompt_content = faang_coding_prompt_template.format(
                    questions=state["QuestionResearch"],
                    company=company_name
                )
            else:
                prompt_content = mass_hiring_coding_prompt_template.format(
                    questions=state["QuestionResearch"],
                    company=company_name
                )
            
            if len(state["messages"]) > 0:
                state["messages"][0].content = prompt_content
            else:
                state["messages"] = [SystemMessage(content=prompt_content)]
            state["LastNode"] = "Coding"

        response = llm.invoke(state["messages"])
        print(response)
        state["messages"] = state["messages"] + [response]
        state["history"] = state["history"] + "\n" + "Interviewer-" + response.content
        state["LastNode"] = "Coding"
        return state
    return _Node


def create_ending_node(llm) -> Callable:
    """Create ending node with company-specific thank you message"""
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


# Routing models
class ConceptualProgress(BaseModel):
    send_to_which_node: Literal['Conceptual', 'Project_before', 'LogicalReasoning_before', 'Coding_before'] = \
        Field(description="Supervise the conceptual/theory phase. Route to 'Conceptual' if still asking questions (need 5-6 total). "
                          "For FAANG companies: route to 'Project_before' when conceptual phase is complete. "
                          "For mass hiring companies: route to 'LogicalReasoning_before' when conceptual phase is complete. "
                          "The conceptual phase is complete after 5-6 distinct questions have been asked and discussed.")


class ProjectProgress(BaseModel):
    send_to_which_node: Literal['Project', 'Coding_before'] = \
        Field(description="Supervise the project discussion phase. Route to 'Project' if still discussing projects (aim for 2-3 projects). "
                          "Route to 'Coding_before' when project discussion is complete (after covering 2-3 projects in detail).")


class ProductScenarioProgress(BaseModel):
    send_to_which_node: Literal['ProductScenario', 'Coding_before'] = \
        Field(description="Supervise the product scenario phase. Route to 'ProductScenario' if still asking scenario questions (need 2-3 total). "
                          "Route to 'Coding_before' when product scenario phase is complete (after 2-3 scenarios have been discussed).")


class LogicalReasoningProgress(BaseModel):
    send_to_which_node: Literal['LogicalReasoning', 'Coding_before'] = \
        Field(description="Supervise the logical reasoning/puzzles phase. Route to 'LogicalReasoning' if still asking questions (need 3-4 total). "
                          "Route to 'Coding_before' when logical reasoning phase is complete (after 3-4 questions).")


class CompanyCodingProgress(BaseModel):
    send_to_which_node: Literal['Coding', 'End'] = \
        Field(description="Supervise the coding phase. Route to 'Coding' if still in progress (need 2 coding problems). "
                          "Route to 'End' when coding phase is complete (after 2 distinct coding problems have been fully resolved).")


def create_route_to_greeting(InterviewProgress_llm) -> Callable:
    """Route after greeting - decide next step"""
    def _Node(state: CompanyInterviewState) -> Literal['Greeting', 'Personalised_before', 'Offensive']:
        response = InterviewProgress_llm.invoke(state["history"])
        print("This is the greeting routing node", response.send_to_which_node)
        return response.send_to_which_node
    return _Node


def create_route_to_conceptual(ConceptualProgress_llm) -> Callable:
    """Route after conceptual phase - decide next step based on company type"""
    def _Node(state: CompanyInterviewState) -> Literal['Conceptual', 'Project_before', 'ProductScenario_before', 'LogicalReasoning_before', 'Coding_before']:
        response = ConceptualProgress_llm.invoke(state["history"])
        print("This is the conceptual routing node", response.send_to_which_node)
        
        # Override routing based on company type if needed
        company_name = state.get("company", "")
        if is_faang_company(company_name):
            # FAANG: Conceptual -> Project -> Coding
            if response.send_to_which_node in ['Project_before', 'Coding_before']:
                return response.send_to_which_node
            elif response.send_to_which_node in ['LogicalReasoning_before', 'ProductScenario_before']:
                return 'Project_before'  # FAANG doesn't have logical reasoning or product scenarios
        elif is_product_based_company(company_name):
            # Product-based: Conceptual -> ProductScenario -> Coding
            if response.send_to_which_node in ['ProductScenario_before', 'Coding_before']:
                return response.send_to_which_node
            elif response.send_to_which_node in ['Project_before', 'LogicalReasoning_before']:
                return 'ProductScenario_before'  # Product-based doesn't have project discussion or logical reasoning
        else:
            # Mass hiring: Conceptual -> LogicalReasoning -> Coding
            if response.send_to_which_node in ['LogicalReasoning_before', 'Coding_before']:
                return response.send_to_which_node
            elif response.send_to_which_node in ['Project_before', 'ProductScenario_before']:
                return 'LogicalReasoning_before'  # Mass hiring doesn't have project discussion or product scenarios
        
        return response.send_to_which_node
    return _Node


def create_route_to_project(ProjectProgress_llm) -> Callable:
    """Route after project discussion - decide next step"""
    def _Node(state: CompanyInterviewState) -> Literal['Project', 'Coding_before']:
        response = ProjectProgress_llm.invoke(state["history"])
        print("This is the project routing node", response.send_to_which_node)
        return response.send_to_which_node
    return _Node


def create_route_to_product_scenario(ProductScenarioProgress_llm) -> Callable:
    """Route after product scenario - decide next step"""
    def _Node(state: CompanyInterviewState) -> Literal['ProductScenario', 'Coding_before']:
        response = ProductScenarioProgress_llm.invoke(state["history"])
        print("This is the product scenario routing node", response.send_to_which_node)
        return response.send_to_which_node
    return _Node


def create_route_to_logical_reasoning(LogicalReasoningProgress_llm) -> Callable:
    """Route after logical reasoning - decide next step"""
    def _Node(state: CompanyInterviewState) -> Literal['LogicalReasoning', 'Coding_before']:
        response = LogicalReasoningProgress_llm.invoke(state["history"])
        print("This is the logical reasoning routing node", response.send_to_which_node)
        return response.send_to_which_node
    return _Node


def create_route_to_company_coding(CompanyCodingProgress_llm) -> Callable:
    """Route after coding - decide next step"""
    def _Node(state: CompanyInterviewState) -> Literal['Coding', 'End']:
        response = CompanyCodingProgress_llm.invoke(state["history"])
        print("This is the coding routing node", response.send_to_which_node)
        return response.send_to_which_node
    return _Node


def build_company_graph(google_api_key: str, tavily_api_key: str, checkpointer):
    """
    Build and compile the Company interview graph.
    This is a dedicated graph for company-wise interviews, separate from subject-wise interviews.
    """
    llm = get_llm(google_api_key=google_api_key)
    workflow = StateGraph(CompanyInterviewState)

    # Nodes
    workflow.add_node("Initial_Research", create_research_summary_node(llm))
    workflow.add_node("Greeting", create_company_greeting_node(llm))
    workflow.add_node("Greeting_after", create_dummy_node())
    workflow.add_node("Personalised_before", create_dummy_node())
    workflow.add_node("Personalised", create_personalised_node(llm))
    workflow.add_node("Personalised_after", create_dummy_node())
    
    # Conceptual/Theory phase (both FAANG and mass hiring)
    workflow.add_node("Conceptual_before", create_dummy_node())
    workflow.add_node("Conceptual", create_conceptual_node(llm))
    workflow.add_node("Conceptual_after", create_dummy_node())
    
    # Project discussion (FAANG only)
    workflow.add_node("Project_before", create_dummy_node())
    workflow.add_node("Project", create_project_node(llm))
    workflow.add_node("Project_after", create_dummy_node())
    
    # Product Scenario (Product-based companies only)
    workflow.add_node("ProductScenario_before", create_dummy_node())
    workflow.add_node("ProductScenario", create_product_scenario_node(llm))
    workflow.add_node("ProductScenario_after", create_dummy_node())
    
    # Logical Reasoning/Puzzles (Mass hiring only)
    workflow.add_node("LogicalReasoning_before", create_dummy_node())
    workflow.add_node("LogicalReasoning", create_logical_reasoning_node(llm))
    workflow.add_node("LogicalReasoning_after", create_dummy_node())
    
    # Coding phase (both)
    workflow.add_node("Coding_before", create_before_coding_node(llm))
    workflow.add_node("Coding", create_company_coding_node(llm))
    workflow.add_node("Coding_after", create_dummy_node())
    
    # Ending
    workflow.add_node("End", create_ending_node(llm))
    workflow.add_node("Offensive", create_offend_end_node(llm))

    # Entry point
    workflow.set_entry_point("Initial_Research")

    # Fixed edges
    workflow.add_edge("Greeting", "Greeting_after")
    workflow.add_edge("Personalised_before", "Personalised")
    workflow.add_edge("Personalised", "Personalised_after")
    workflow.add_edge("Initial_Research", "Greeting")
    workflow.add_edge("Conceptual_before", "Conceptual")
    workflow.add_edge("Conceptual", "Conceptual_after")
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

    # Conditional routing
    workflow.add_conditional_edges(
        "Greeting_after",
        create_route_to_greeting(llm.with_structured_output(InterviewProgress))
    )
    # After personalised, route but override to go to Conceptual
    workflow.add_conditional_edges(
        "Personalised_after",
        create_route_to_personalised(llm.with_structured_output(PersonalisedProgress)),
        {
            "Personalised": "Personalised_before",  # Loop back if more exchanges needed
            "Coding_before": "Conceptual_before",  # Override: go to Conceptual instead of Coding
        }
    )
    
    # After Conceptual, route based on company type
    # FAANG -> Project, Product-based -> ProductScenario, Mass hiring -> LogicalReasoning
    workflow.add_conditional_edges(
        "Conceptual_after",
        create_route_to_conceptual(llm.with_structured_output(ConceptualProgress)),
        {
            "Conceptual": "Conceptual_before",  # Loop back if more questions needed
            "Project_before": "Project_before",  # FAANG: go to Project
            "ProductScenario_before": "ProductScenario_before",  # Product-based: go to ProductScenario
            "LogicalReasoning_before": "LogicalReasoning_before",  # Mass hiring: go to LogicalReasoning
            "Coding_before": "Coding_before",  # Fallback: go directly to Coding
        }
    )
    
    # After Project (FAANG only) -> go to Coding
    workflow.add_conditional_edges(
        "Project_after",
        create_route_to_project(llm.with_structured_output(ProjectProgress)),
        {
            "Project": "Project_before",  # Loop back if more projects to discuss
            "Coding_before": "Coding_before",  # Go to Coding when done
        }
    )
    
    # After ProductScenario (Product-based only) -> go to Coding
    workflow.add_conditional_edges(
        "ProductScenario_after",
        create_route_to_product_scenario(llm.with_structured_output(ProductScenarioProgress)),
        {
            "ProductScenario": "ProductScenario_before",  # Loop back if more scenarios needed
            "Coding_before": "Coding_before",  # Go to Coding when done
        }
    )
    
    # After LogicalReasoning (Mass hiring only) -> go to Coding
    workflow.add_conditional_edges(
        "LogicalReasoning_after",
        create_route_to_logical_reasoning(llm.with_structured_output(LogicalReasoningProgress)),
        {
            "LogicalReasoning": "LogicalReasoning_before",  # Loop back if more questions needed
            "Coding_before": "Coding_before",  # Go to Coding when done
        }
    )
    
    # After Coding -> go to End
    workflow.add_conditional_edges(
        "Coding_after",
        create_route_to_company_coding(llm.with_structured_output(CompanyCodingProgress)),
        {
            "Coding": "Coding_before",  # Loop back if more coding problems needed
            "End": "End",  # End interview when complete
        }
    )

    agent = workflow.compile(checkpointer=checkpointer)
    print("[INFO] Company interview graph compiled successfully")
    return agent
