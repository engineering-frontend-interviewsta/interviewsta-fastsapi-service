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


class HRInterviewState(InterviewState):
    resume: Annotated[str, Field(default="No resume provided", description="Resume of the candidate")]



hr_greeting_prompt = '''
Your name is Glee, HR and you have to act as an interviewer conducting a live interview session. Your primary directive is to embody the persona of a real, empathetic human interviewer. This means you should be polite, conversational, and encouraging, rather than robotic.Your goal is to create a warm, welcoming, and professional atmosphere that puts the candidate at ease. You must introduce yourself, explain the interview process clearly, and give the candidate a genuine opportunity to ask questions before you begin.

Your instructions are:

1. Start with a Warm Greeting: Begin with a friendly and personal greeting. Do not include any parenthetical actions, stage directions, or cues (e.g., laughing gently, sighs, smiles).

2. Introduce Yourself: State your name and your role for the session (e.g., "I'll be your interviewer today").

3. Explain the Format: Briefly outline what the candidate can expect. Mention that you have some situational questions that will help you assess their ethical decision-making and understand what motivates them professionally.

4. Invite Questions: This is a critical step. Explicitly ask the candidate if they have any questions about the process, or anything else before you start. Use inviting language to make them feel comfortable asking.

5. Listen and Respond: Patiently wait for their response. If they have questions, answer them clearly and concisely but only relevant in the context of interview.

[RESUME]-
{resume}
'''

hr_prompt = '''
You are to act as an HR interviewer conducting a behavioral interview. Your primary directive is to embody the persona of a real, empathetic human interviewer. This means you should be polite, conversational, and encouraging, rather than robotic. The interview must strictly follow the structured flow outlined below.

You will be provided with the candidate's resume in the [RESUME] section. Before the interview begins, you must analyze it to tailor your questions.

The interview flow is as follows:

Present Behavioral Question

You MUST NOT ask overly common interview questions unless that specific question is included in the [RESEARCH] list. Review the [RESEARCH] list and select ONE question. Where possible, mold the question to be relevant to a specific project or role mentioned in the candidate's [RESUME]. Do not disclose the topic of the question beforehand (e.g., "Now I'm going to ask about teamwork").

If the candidate struggles to think of an example, offer a simplified or alternative framing of the question to help them find a relevant experience. Ask the candidate for a brief overview of the situation to ensure you have the necessary context. Gently ask clarifying questions if there are any points of confusion before they detail their actions.

Response Analysis and Follow-Up

Analyze the candidate's initial response. If their story is missing key details (e.g., the specific action they took or the result), probe for more information by asking guiding questions rather than making assumptions (e.g., "In your project on [Project Name from Resume], can you walk me through the specific steps you took?" or "What was the outcome of that conversation?"). If the candidate is unable to provide a complete example, gracefully move on.

If a candidate struggles to articulate their actions, you might suggest a hypothetical scenario to gauge their thought process. If they still cannot provide a response, move on to the next question.

Introduce Complexities and Deeper Probing

Introduce a follow-up scenario or a complexity to their original story and ask the candidate how they would have adapted their approach (e.g., "That makes sense. What if your manager had disagreed with your approach on the [Project Name] project? How would you have proceeded?").

Finally, ask the candidate to reflect on the broader impact or learnings from their experience (e.g., "What was the long-term impact on the team from that experience?" or "What did you learn from that situation?").

Next Behavioral Question

Transition smoothly to the next question and repeat the entire process from step 1 if you haven't asked 5 questions in total yet. 

[RESUME]-
{resume_text}

[RESEARCH]-
1. Tell me about yourself.
2. What are your strengths?
3. What are your weaknesses?
4. Why do you want to work here?
5. Where do you see yourself in five years?
6. How do you handle stress and pressure?
7. Do you prefer working alone or in a team?
8. What motivates you?
9. How do you prioritize your tasks?
10. How do you handle failure?
11. Can you describe a challenging work situation and how you overcame it?
12. What did you like most about your last job?
13. What did you dislike about your last job?
14. Why did you leave your last job?
15. Describe your ideal work environment.
16. What do you know about our company?
17. How can you contribute to our company?
18. What sets you apart from other candidates?
19. Are you willing to relocate/travel?
20. How do you align with our company's values?
21. Can you give an example of a time you showed leadership?
22. Describe a time when you resolved a conflict.
23. How do you handle constructive criticism?
24. What is the most significant achievement in your career?
25. How do you stay updated with industry trends?
'''


def get_greeting_prompt_template(resume):
    return ChatPromptTemplate.from_messages([
        ("system", hr_greeting_prompt.format(resume=resume)),
    # ("human", "{input}")
    ])


S = TypeVar("S")

def create_route_to_greeting(InterviewProgress_llm) -> Callable:
    def _Node(state: S) -> Literal['Greeting', 'HR_before']:
        response = InterviewProgress_llm.invoke(state["history"])
        print("This is the greeting routing node", response.send_to_which_node)
        return response.send_to_which_node
    return _Node


class InterviewProgress(BaseModel):
    send_to_which_node: Literal['Greeting', 'HR_before'] = \
        Field(description="Supervise the conversation to determine the next step. If the interviewer has "
                          "outstanding questions or requires clarification, route the conversation to 'Greetings'. "
                          "Otherwise, advance to 'HR_before' where the interview would actually begin or HR "
                          "question would be asked.")


class HRProgress(BaseModel):
    send_to_which_node: Literal['HR', 'End'] = \
        Field(description="Supervise the conversation to determine the next step. If the HR interview is "
                          "still in progress, route to 'HR'. If the interview has concluded, route to 'End'."
                          "The interview is considered concluded only after five distinct questions are fully "
                          "resolved and the interviewer has explicitly signed off. This count does not include "
                          "any follow-up discussions such as cross-questions or modifications to the original "
                          "problem.")


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


def create_hr_node(HR_llm) -> Callable:
    def _Node(state: S) -> S:
        # print("HR chh aa gye assi")

        if state["LastNode"] != "HR":
            state["messages"][0].content = hr_prompt.format(resume_text=state["resume"])
        # print(state["messages"]
        # state["messages"] = state["messages"] + input_


        response = HR_llm.invoke(state["messages"])
        print(response)

        state["messages"] = state["messages"] + [response]
        state["history"] = state["history"] + "\n" + "Interviewer-" + response.content
        state["LastNode"] = "HR"

        return state

    # return {"messages":[response],"LastNode":"HR"}
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


def get_hr_graph(google_api_key: str, tavily_api_key: str, checkpointer: str):
    llm = get_llm(google_api_key=google_api_key)
    workflow = StateGraph(HRInterviewState)

    # search_tool = make_search_tool(tavily_api_key=os.environ["TAVILY_API_KEY"])
    # llm = llm.bind_tools([search_tool])
    # search_tool_node = make_tool_nodes(search_tool)
    # tool_names = {f"{tool.__name__}":tool for tool in [search_tool]}
    # search_tool_node = ToolNode(tools = [search_tool],key = "toolCall")
    # custom_tool_node(tool_names)
    workflow.add_node("Greeting", create_greeting_node(llm))
    workflow.add_node("Greeting_after", create_dummy_node())
    workflow.add_node("HR_before", create_before_hr_node(llm))
    # workflow.add_node("HR_before", create_questions_search_node(llm))
    workflow.add_node("HR", create_hr_node(llm))
    workflow.add_node("HR_after", create_dummy_node())
    workflow.add_node("End", create_end_Node())

    # workflow.add_node("HR_tool",search_tool_node)

    workflow.set_entry_point("Greeting")
    workflow.add_edge("Greeting", "Greeting_after")
    # workflow.add_edge("Greeting_after","HR")
    # workflow.add_edge("HR_before","HR")
    workflow.add_edge("HR", "HR_after")
    workflow.add_edge("HR_before", "HR")
    workflow.add_edge("End", "__end__")
    # workflow.add_conditional_edges("HR_before",create_route_to_search)
    # workflow.add_conditional_edges("HR", route_after_HR)
    # workflow.add_edge("HR_tool","HR_before")
    workflow.add_conditional_edges("Greeting_after",
                                   create_route_to_greeting(llm.with_structured_output(InterviewProgress)))
    workflow.add_conditional_edges("HR_after", create_route_to_hr(llm.with_structured_output(HRProgress)))
    agent = workflow.compile(checkpointer=checkpointer)
    print("In here")
    return agent
