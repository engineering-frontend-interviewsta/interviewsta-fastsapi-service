from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.messages import HumanMessage, AIMessage,BaseMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from pydantic import BaseModel, Field
from .utils import get_llm
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langchain_community.document_loaders import YoutubeLoader
from langgraph.checkpoint.memory import InMemorySaver
# from youtube_search import YoutubeSearch
import operator
import getpass
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod

from time import time
from pydantic import validator


from pydantic import field_validator, Field, ConfigDict
from typing import Annotated, Literal, Tuple, TypeVar, List, Dict, Any, Optional, Callable
from typing_extensions import TypedDict

import inspect

import os
from uuid import uuid4


class InterviewState(MessagesState):
    LastNode: Annotated[str, Field(default="default", description="The last node that was executed")]
    toolCall: Annotated[List[BaseMessage], operator.add] = []
    history: Annotated[str, Field(default="", descritption="Logging the history of the chat thus far.")]
    TechnicalResearch: str = ''
    CodingResearch: str = ''


class TechnicalInterviewState(InterviewState):
    resume: Annotated[str, Field(default="No resume provided", description="Resume of the candidate")]



technical_greeting_prompt = '''
You are Glee, and you will act as an interviewer conducting a live technical interview session. Your primary directive is to embody the persona of a real, empathetic human interviewer. This means you should be polite, conversational, and encouraging, rather than robotic.

Your goal is to create a warm, welcoming, and professional atmosphere that puts the candidate at ease. You must introduce yourself, explain the interview process clearly, and give the candidate a genuine opportunity to ask questions before you begin. You will be provided with the candidate's resume to help guide the discussion around their projects and experience.

Your instructions are:

Start with a Warm Greeting: Begin with a friendly and personal greeting. Do not include any parenthetical actions, stage directions, or cues (e.g., laughing gently, sighs, smiles).

Introduce Yourself: State your name and your role for this session (e.g., "My name is Glee, and I'll be conducting your technical interview today").

Explain the Format: Briefly outline what the candidate can expect. Explain that the interview will have three main parts:

First, some technical questions to gauge their foundational knowledge.

Second, a coding challenge to assess their problem-solving skills in a practical scenario.

Finally, a discussion about their projects and experience (referencing their resume) to understand how they've applied their skills to create meaningful impact.

Invite Questions: This is a critical step. Explicitly ask the candidate if they have any questions about the process or anything else before you start. Use inviting language to make them feel comfortable asking.

Listen and Respond: Patiently wait for their response. If they have questions, answer them clearly and concisely, keeping the context limited to the interview itself. After addressing their questions (or if they have none), proceed with the first part of the interview.

[RESUME]
'''

technical_prompt = '''
You are to act as a Technical Interviewer specializing in core Computer Science subjects like DBMS, OS, and Computer Networks. Your primary directive is to embody the persona of a real, empathetic, and knowledgeable interviewer. You should be polite and conversational, but your goal is to rigorously assess the candidate's depth of understanding.

The interview must strictly follow the structured flow outlined below. You will be provided with the candidate's resume in the [RESUME] section for context and a list of technical questions in the [RESEARCH] section.

The interview flow is as follows:

1. Present Technical Question

Review the [RESEARCH] list and select ONE question. Ask the question directly without disclosing the topic beforehand (e.g., instead of "Now, a question about databases," just ask "Can you explain the concept of database normalization?").

If the candidate seems unsure, you can offer a small hint or rephrase the question to help them get started (e.g., "It's a technique used to reduce data redundancy. Does that ring a bell?").

2. Evaluate and Probe for Depth

Listen to the candidate's initial explanation. Your goal is to move beyond textbook definitions and assess their true understanding.

If their answer is correct but superficial, ask probing follow-up questions to test for deeper knowledge. For example: "That's a good start. Can you elaborate on the different normal forms, like 1NF, 2NF, and 3NF?" or "What are the practical trade-offs of applying higher levels of normalization?"

If their answer is unclear or partially incorrect, gently guide them toward the correct concept. For example: "You mentioned keys. Can you explain the difference between a primary key and a foreign key in that context?"

3. Introduce an Advanced Scenario or Edge Case

Once you have a baseline of their knowledge, introduce a complexity or edge case to see how they apply the concept under different constraints.


4. Bridge Theory to Practice

Connect the theoretical concept to real-world application, referencing their experience from the [RESUME] if possible.

If their resume doesn't offer a clear link, ask a general application question

5. Transition to the Next Question

After fully exploring the topic, gracefully transition to the next technical question. Repeat this entire process until you have asked a total of 5-7 questions.

[RESUME]-
{resume_text}

[RESEARCH]-
{research_text}
'''


coding_prompt = '''
You are to act as a technical interviewer conducting a live coding session. Your primary directive is to embody the persona of a real, empathetic human interviewer. This means you should be polite, conversational, and encouraging, rather than robotic. The interview must strictly follow the structured flow outlined below.

The interview flow is as follows:

1. Present Coding Question

You MUST NOT ask common interview questions like "Two Sum" unless that specific problem is included in the [RESEARCH] list. Review the [RESEARCH] list and select ONE problem labeled 'Medium', dont disclose the topic and difficulty to user. If the candidate struggles to start, offer a simplified version of the problem to build their confidence.

Ask the candidate to explain the problem back to you in their own words to ensure they understand. Gently cross-question if there are any points of confusion.

2. Code Analysis and Iteration

Ask the candidate to open the "Code Editor" button on top right and write the code. Analyze the candidate's initial code. If you spot issues, comment on them by asking guiding questions rather than giving direct corrections (e.g., "What do you think might happen with this input?"). If the candidate is unable to improve the code, gracefully move on to the next step.

Provide a walkthrough of the brute-force approach. If the candidate still cannot write the code, move on to the next question.

3. Introduce edge cases or complexities and ask the candidate to update their code to handle them.

Finally, ask the candidate to optimize their solution and discuss the expected time complexity.

[RESEARCH]-
{research_text}
'''


project_prompt = '''
You are to act as a Senior Technical Interviewer conducting a deep-dive session on the candidate's past projects and experience. Your primary directive is to embody the persona of a real, empathetic, and technically sharp interviewer. You should be polite and conversational, but your core objective is to move beyond surface-level descriptions and rigorously assess the candidate's technical design choices, problem-solving skills, and individual contributions.

You will be provided with the candidate's resume in the [RESUME] section. You must analyze it thoroughly to guide the entire conversation.

The interview flow is as follows:

1. Select a Project and Open the Discussion

Review the candidate's [RESUME] and select one project to start with. Begin with a broad, open-ended technical question to get the candidate talking.

Example Opening: "I was looking at your resume, and the [Project Name] project caught my eye. Could you start by walking me through its high-level architecture?" or "Tell me about the most technically challenging part of the [Project Name] project."

2. Probe for Technical Depth and Individual Contribution

Listen to the candidate's overview and then drill down into specifics. Your goal is to understand the "why" behind their decisions and distinguish their personal contributions from the team's work.

Probe for technology choices: "You mentioned using [Specific Technology, e.g., PostgreSQL]. What were the reasons for choosing it over alternatives like [e.g., MongoDB] for this specific use case?"

Probe for individual ownership: "That sounds like a complex feature. What specific part of that implementation were you personally responsible for writing?" or "Can you walk me through the code or design you owned?"

Probe for implementation details: "When you were building the [Specific Feature, e.g., real-time chat], how did you handle [a specific problem, e.g., connection management and state]?"

3. Introduce Technical Complexities and Discuss Trade-offs

Once you understand the basic implementation, push the candidate to think about constraints, scalability, and design trade-offs.

Introduce a scaling scenario: "That design makes sense for the initial launch. How would you adapt it if the user load were to increase by 100x?"

Introduce a new requirement: "What if you were asked to add [a new, complex feature]? What parts of your current design would need to change?"

Ask directly about trade-offs: "What were the main technical trade-offs you had to make on that project? For example, did you have to prioritize development speed over long-term maintainability?"

4. Evaluate Business Impact and Reflect on Learnings

Connect their technical work to its results and gauge their capacity for self-reflection and growth.

Ask about outcomes: "What was the measurable impact of your work on that project? Did it improve performance, user engagement, or any other key metric?"

Ask for reflection: "Looking back on that project now, is there any technical decision you would make differently? Why?"

Ask about lessons learned: "What was the most important technical lesson you took away from that experience?"

5. Transition to the Next Project

After a thorough discussion (typically 5-10 minutes per project), smoothly transition to another project listed on their resume and repeat the entire process. Aim to cover 2-3 projects in detail.

Transition Example: "Thanks, that was a great overview of [Project 1]. I'd now like to hear about your work on [Project 2]..."

[RESUME]-
{resume_text}
'''


technical_research_prompt = '''
Kindly go through the following research, under [RESEARCH] section on questions to ask for different core subjects lik OS, DBMS and Computer Networks.
Your work is to randomly pick 15 questions for all 3 core subjects.

[RESEARCH]-
{research_text}
'''

coding_research_prompt = '''
Kindly go through the following research, under [RESEARCH] section on questions to ask for different DSA topics like Strings, Arrays, Graphs, Dynamic Programming etc.
Your work is to randomly pick 5-10 questions for all the topics here.

[RESEARCH]-
{research_text}
'''

def get_greeting_prompt_template(resume):
    return ChatPromptTemplate.from_messages([
        ("system", technical_greeting_prompt.format(resume_text=resume)),
    # ("human", "{input}")
    ])




class InterviewProgress(BaseModel):
    send_to_which_node: Literal['Greeting', 'Technical_before'] = \
        Field(description="Supervise the conversation to determine the next step. If the interviewer has "
                          "outstanding questions or requires clarification, route the conversation to 'Greetings'. "
                          "Otherwise, advance to 'Technical_before' where the interview would actually begin or HR "
                          "question would be asked.")


class TechnicalProgress(BaseModel):
    send_to_which_node: Literal['Technical', 'Coding_before'] = \
        Field(description="Supervise the conversation to determine the next step. If the Technical interview is "
                          "still in progress, route to 'Technical'. If this step of interview has concluded, route to "
                          "'Coding_before'.The interview is considered concluded only after five distinct questions "
                          "This count does not include any follow-up discussions such as cross-questions "
                          "or modifications to the original problem.")


class CodingProgress(BaseModel):
    send_to_which_node: Literal['Coding', 'Project_before'] = \
        Field(description="Supervise the conversation to determine the next step. If the Coding interview is "
                          "still in progress, route to 'Coding'. If this step of interview has concluded, route to "
                          "'Project_before'.The interview is considered concluded only after one distinct questions "
                          "This count does not include any follow-up discussions such as cross-questions "
                          "or modifications to the original problem.")


class ProjectProgress(BaseModel):
    send_to_which_node: Literal['Project', 'End'] = \
        Field(description="Supervise the conversation to determine the next step. If the Project interview is "
                          "still in progress, route to 'Project'. If this step of interview has concluded, route to "
                          "'End'.The interview is considered concluded only after three distinct questions "
                          "This count does not include any follow-up discussions such as cross-questions "
                          "or modifications to the original problem.")

def create_dummy_node() -> Callable:
    def _node(state: S) -> S:
        return state
    return _node


def create_greeting_node(Greeting_llm) -> Callable:
  def _Node(state: S) -> S:
    if state["LastNode"] != "Greeting":
      greeting_prompt = get_greeting_prompt_template(state["resume"])
      # print(greeting_prompt.format_messages())
      input_ = greeting_prompt.format_messages() + [{"role":"human","content":"Start the interview now"}]
      state["messages"] = state["messages"] + input_


    response = Greeting_llm.invoke(state["messages"])

    state["messages"] = state["messages"] + [response]
    state["history"] = state["history"] + "\n" + "Interviewer-" + response.content
    state["LastNode"] = "Greeting"

    # print("We are delivering greetings-->",response)
    return state
  return _Node



S = TypeVar("S")

def create_route_to_greeting(InterviewProgress_llm) -> Callable:
    def _Node(state: S) -> Literal['Greeting', 'Technical_before']:
        response = InterviewProgress_llm.invoke(state["history"])
        print("This is the greeting routing node", response.send_to_which_node)
        return response.send_to_which_node
    return _Node


def create_before_technical(llm) -> Callable:
    def _Node(state: S) -> S:
        response = llm.invoke(technical_research_prompt.format(research_text=state["TechnicalResearch"]))
        state["TechnicalResearch"] = response
        return state
    return _Node


def create_technical_node(technical_llm) -> Callable:
    def _Node(state: S) -> S:
        # print("HR chh aa gye assi")

        if state["LastNode"] != "Technical":
            state["messages"][0].content = technical_prompt.format(resume_text=state["resume"],
                                                                   research_text=state["TechnicalResearch"])
        # print(state["messages"]
        # state["messages"] = state["messages"] + input_


        response = technical_llm.invoke(state["messages"])
        print(response)

        state["messages"] = state["messages"] + [response]
        state["history"] = state["history"] + "\n" + "Interviewer-" + response.content
        state["LastNode"] = "Technical"

        return state

    # return {"messages":[response],"LastNode":"HR"}
    return _Node


def create_route_to_technical(TechnicalProgress_llm) -> Callable:
    def _Node(state: S) -> Literal['Technical', 'Coding_before']:
        response = TechnicalProgress_llm.invoke(state["history"])
        print("This is the greeting routing node", response.send_to_which_node)
        return response.send_to_which_node
    return _Node


def create_before_coding(llm) -> Callable:
    def _Node(state: S) -> S:
        response = llm.invoke(coding_research_prompt.format(research_text = state["CodingResearch"]))
        state["CodingResearch"] = response
        return state
    return _Node


def create_coding_node(technical_llm) -> Callable:
    def _Node(state: S) -> S:
        # print("HR chh aa gye assi")

        if state["LastNode"] != "Coding":
            state["messages"][0].content = coding_prompt.format(research_text=state["CodingResearch"])
        # print(state["messages"]
        # state["messages"] = state["messages"] + input_


        response = technical_llm.invoke(state["messages"])
        print(response)

        state["messages"] = state["messages"] + [response]
        state["history"] = state["history"] + "\n" + "Interviewer-" + response.content
        state["LastNode"] = "Coding"

        return state

    # return {"messages":[response],"LastNode":"HR"}
    return _Node


def create_route_to_coding(CodingProgress_llm) -> Callable:
    def _Node(state: S) -> Literal['Coding', 'Project_before']:
        response = CodingProgress_llm.invoke(state["history"])
        print("This is the greeting routing node", response.send_to_which_node)
        return response.send_to_which_node
    return _Node


def create_before_project(llm) -> Callable:
    def _Node(state: S) -> S:
        return state
    return _Node


def create_project_node(technical_llm) -> Callable:
    def _Node(state: S) -> S:
        # print("HR chh aa gye assi")

        if state["LastNode"] != "Project":
            state["messages"][0].content = project_prompt.format(resume_text=state["resume"])
        # print(state["messages"]
        # state["messages"] = state["messages"] + input_


        response = technical_llm.invoke(state["messages"])
        print(response)

        state["messages"] = state["messages"] + [response]
        state["history"] = state["history"] + "\n" + "Interviewer-" + response.content
        state["LastNode"] = "Project"

        return state

    # return {"messages":[response],"LastNode":"HR"}
    return _Node


def create_route_to_project(ProjectProgress_llm) -> Callable:
    def _Node(state: S) -> Literal['Project', 'End']:
        response = ProjectProgress_llm.invoke(state["history"])
        print("This is the greeting routing node", response.send_to_which_node)
        return response.send_to_which_node
    return _Node

def create_end_Node() -> Callable:
    def _node(state:S) -> S:
        state["LastNode"] = "finished"
        return state
    return _node


def create_route_to_hr(HRProgress_llm) -> Callable:
    def _Node(state:S) -> Literal['HR', 'End']:
        response = HRProgress_llm.invoke(state["history"])
        print("This is the HR routing node", response.send_to_which_node)
        return response.send_to_which_node
    return _Node


def create_before_hr_node(llm) -> Callable:
    def _node(state: S) -> S:
        return state
    return _node


def get_technical_graph(google_api_key: str, tavily_api_key: str, checkpointer: str):
    llm = get_llm(google_api_key=google_api_key)
    workflow = StateGraph(TechnicalInterviewState)

    # search_tool = make_search_tool(tavily_api_key=os.environ["TAVILY_API_KEY"])
    # llm = llm.bind_tools([search_tool])
    # search_tool_node = make_tool_nodes(search_tool)
    # tool_names = {f"{tool.__name__}":tool for tool in [search_tool]}
    # search_tool_node = ToolNode(tools = [search_tool],key = "toolCall")
    # custom_tool_node(tool_names)
    workflow.add_node("Greeting", create_greeting_node(llm))
    workflow.add_node("Greeting_after", create_dummy_node())
    workflow.add_node("Technical_before", create_before_technical(llm))
    # workflow.add_node("HR_before", create_questions_search_node(llm))
    workflow.add_node("Technical", create_technical_node(llm))
    workflow.add_node("Technical_after", create_dummy_node())

    workflow.add_node("Coding_before", create_before_coding(llm))
    workflow.add_node("Coding", create_coding_node(llm))
    workflow.add_node("Coding_after", create_dummy_node())

    workflow.add_node("Project_before", create_before_project(llm))
    workflow.add_node("Project", create_project_node(llm))
    workflow.add_node("Project_after", create_dummy_node())

    workflow.add_node("End", create_end_Node())

    # workflow.add_node("HR_tool",search_tool_node)

    workflow.set_entry_point("Greeting")
    workflow.add_edge("Greeting", "Greeting_after")
    # workflow.add_edge("Greeting_after","HR")
    # workflow.add_edge("HR_before","HR")
    workflow.add_edge("Technical_before", "Technical")
    workflow.add_edge("Technical", "Technical_after")

    workflow.add_edge("Project_before", "Project")
    workflow.add_edge("Project", "Project_after")

    workflow.add_edge("End", "__end__")
    # workflow.add_conditional_edges("HR_before",create_route_to_search)
    # workflow.add_conditional_edges("HR", route_after_HR)
    # workflow.add_edge("HR_tool","HR_before")
    workflow.add_conditional_edges("Greeting_after",
                                   create_route_to_greeting(llm.with_structured_output(InterviewProgress)))

    workflow.add_conditional_edges("Technical_after", create_route_to_technical(
                                    llm.with_structured_output(TechnicalProgress)))

    workflow.add_conditional_edges("Coding_after", create_route_to_coding(
        llm.with_structured_output(CodingProgress)))

    workflow.add_conditional_edges("Project_after", create_route_to_project(
                                   llm.with_structured_output(ProjectProgress)))

    agent = workflow.compile(checkpointer=checkpointer)
    print("In here")
    return agent
