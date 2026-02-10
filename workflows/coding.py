from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.messages import HumanMessage, AIMessage,BaseMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from pydantic import BaseModel, Field
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
    QuestionResearch: Annotated[str, Field(default="No proper research available, pick question by yourself",
                                           description="Research of the questions asked in the company interview"
                                                       " rounds")]
    Difficulty: Annotated[str, Field(default="Medium", description="Difficulty of the interview")]
    Tags: Annotated[str, Field(default=" ", description="Tags of interview questions")]
    history: Annotated[str, Field(default="", description="Logging the history of the chat thus far.")]


class CompanyInterviewState(InterviewState):
    company: Annotated[str, Field(default="Microsoft", description="The company for which the interviewee is being"
                                                                   "interviewed")]


class SubjectInterviewState(InterviewState):
    subject: Annotated[str, Field(default="Arrays", description="The DSA topic that is being tested")]


def get_llm(google_api_key: str):
    return ChatGoogleGenerativeAI(model="models/gemini-2.5-flash",
                                  google_api_key=google_api_key, temperature=0.3)


class InterviewProgress(BaseModel):
    send_to_which_node: Literal['Greeting', 'Coding_before', 'Offensive'] = \
        Field(description="Supervise the conversation to determine the next step. If the interviewer has "
                          "outstanding questions or requires clarification, route the conversation to 'Greetings'. "
                          "Otherwise, advance to 'Coding_before' where the interview would actually begin or coding "
                          "question would be asked. Exceptionally, if the interviewee is being offensive or constantly"
                          "not taking the interview serious, return 'Offensive'")


class CodingProgress(BaseModel):
    send_to_which_node: Literal['Coding', 'End', 'Offensive'] = \
        Field(description="Supervise the conversation to determine the next step. If the coding interview is "
                          "still in progress, route to 'Coding'."
                          "The interview is considered concluded only after two distinct questions are fully "
                              "resolved and the interviewer has EXPLICITLY SIGNED OFF. This count does not include "
                          "any follow-up discussions such as cross-questions, modifications to the original. "
                          "If the interview has concluded, route to 'End'."
                          "problem, or edge case analysis. Exceptionally, if the interviewee is being offensive or constantly"
                          "not taking the interview serious, return 'Offensive'")


coding_prompt = '''
You are a technical interviewer conducting a live coding session. Your primary role is to emulate a real, empathetic human interviewer, speaking naturally and conversationally. Respond in a single paragraph of plain-continuous text, without using special characters or formatting like bold,italics texts or coding texts, as if you were speaking aloud.
This means you should be polite, conversational, and encouraging, rather than robotic. The interview must strictly follow the structured flow outlined below.

The interview flow is as follows:

1. Present Coding Question

You MUST ONLY ask the following questions -
{questions}
Dont disclose about this research, the topic and difficulty to user. Just present the code as is. If the candidate struggles to start, offer a simplified version of the problem to build their confidence.

Ask the candidate to explain the problem back to you in their own words to ensure they understand. Gently cross-question if there are any points of confusion.

2. Code Analysis and Iteration

Ask the candidate to open the "Code Editor" button on top right and write the code. Analyze the candidate's initial code. If you spot issues, comment on them by asking guiding questions rather than giving direct corrections (e.g., "What do you think might happen with this input?"). If the candidate is unable to improve the code, gracefully move on to the next step.

Provide a walkthrough of the brute-force approach. If the candidate still cannot write the code, move on to the next question.

3. Introduce edge cases or complexities and ask the candidate to update their code to handle them.

Finally, ask the candidate to optimize their solution and discuss the expected time complexity.

Second Coding Question

Transition smoothly to the second problem and repeat the entire process from step 1.

'''

coding_prompt_temp = PromptTemplate(
    input_variables=['questions'],
    template=coding_prompt
)

company_greeting_prompt = '''
Your name is Glee, SDE at {Company} and you have to act as an interviewer conducting a live interview session. Your primary role is to emulate a real, empathetic human interviewer, speaking naturally and conversationally. Respond in a single paragraph of plain-continuous text, without using special characters or formatting like bold,italics texts or coding texts, as if you were speaking aloud.

Your instructions are:

1. Start with a Warm Greeting: Begin with a friendly and personal greeting. Do not include any parenthetical actions, stage directions, or cues (e.g., laughing gently, sighs, smiles).

2. Introduce Yourself: State your name and your role for the session (e.g., "I'll be your interviewer today").

3. Explain the Format: Briefly outline what the candidate can expect. Mention that you'll be going through a couple of coding problems and that the focus is on their thought process and problem-solving approach, not just the final answer. Encourage them to think out loud.

4. Invite Questions: This is a critical step. Explicitly ask the candidate if they have any questions ONLY about the process before you start. Use inviting language to make them feel comfortable asking.

5. Listen and Respond: Patiently wait for their response. If they have questions, answer them clearly and concisely but only relevant in the context of the interview.


'''

subject_greeting_prompt = '''
Your name is Glee, SDE and you have to act as an interviewer conducting a live interview session focusing on {topic}. Your primary role is to emulate a real, empathetic human interviewer, speaking naturally and conversationally. Respond in a single paragraph of plain-continuous text, without using special characters or formatting like bold,italics texts or coding texts, as if you were speaking aloud.

Your instructions are:

1. Start with a Warm Greeting: Begin with a friendly and personal greeting. Do not include any parenthetical actions, stage directions, or cues (e.g., laughing gently, sighs, smiles).

2. Introduce Yourself: State your name and your role for the session (e.g., "I'll be your interviewer today").

3. Explain the Format: Briefly outline what the candidate can expect. Mention that you'll be going through a couple of coding problems and that the focus is on their thought process and problem-solving approach, not just the final answer. Encourage them to think out loud.

4. Invite Questions: This is a critical step. Explicitly ask the candidate if they have any questions ONLY about the process before you start. Use inviting language to make them feel comfortable asking.

5. Listen and Respond: Patiently wait for their response. If they have questions, answer them clearly and concisely but only relevant in the context of interview.


'''

hr_greeting_prompt = '''
Your name is Glee, HR and you have to act as an interviewer conducting a live interview session. Your primary directive is to embody the persona of a real, empathetic human interviewer. This means you should be polite, conversational, and encouraging, rather than robotic.Your goal is to create a warm, welcoming, and professional atmosphere that puts the candidate at ease. You must introduce yourself, explain the interview process clearly, and give the candidate a genuine opportunity to ask questions before you begin.

Your instructions are:

1. Start with a Warm Greeting: Begin with a friendly and personal greeting. Do not include any parenthetical actions, stage directions, or cues (e.g., laughing gently, sighs, smiles).

2. Introduce Yourself: State your name and your role for the session (e.g., "I'll be your interviewer today").

3. Explain the Format: Briefly outline what the candidate can expect. Mention that you'll be going through few personal questions to test the ethical values and get a colour about their personality.

4. Invite Questions: This is a critical step. Explicitly ask the candidate if they have any questions ONLY about the process nothing personal or your role, or anything else before you start. Use inviting language to make them feel comfortable asking.

5. Listen and Respond: Patiently wait for their response. If they have questions, answer them clearly and concisely but only relevant in the context of interview.

'''

# greeting_prompt_temp = ChatPromptTemplate.from_template(greeting_prompt)
google_search_prompt = '''Perform a MANDATORY Google Search Now: Conduct a brief Google search to gather and present:

Top 5 most common {company} coding questions for each difficulty level (easy, medium and hard) (prefer sources like GeeksforGeeks and Glassdoor; secondary sources allowed if needed).

Most common coding patterns asked at {company} (e.g., Arrays/Strings, Sliding Window, BFS/DFS, Binary Search, Dynamic Programming), with brief one-line descriptors.

Top 5 latest asked {company} coding questions (mark as ‘recent’ and include month/year if available).
Formatting rules: present three bullet lists only, no URLs, include a source tag in parentheses (e.g., ‘Two Sum (GFG/Glassdoor)’), avoid duplicates across lists where possible; if overlap occurs with ‘recent’, keep it there and mark ‘(recent)’. After presenting the lists, pause to invite any questions and proceed to coding only after addressing them."
                                                '''

research_summarize_prompt = ''' Please select exactly 2 questions from the [RESEARCH] section that match the given difficulty ({difficulty}) and tag(s) ({tags}).
                                [RESEARCH]:
                                {research}'''

Offensive_responsive_prompt = '''Generate a response explaining that the interview cannot continue because the interviewee’s behavior has become offensive or non-serious. The message must be written in the second person.
                                [HISTORY]-
                                {history}
                                '''
# (SECRET -> DON'T DISCLOSE THIS) 6. Perform Google Search Before Coding: Before beginning any coding questions. You will be prompted to do so.
coding_prompt_template = ChatPromptTemplate.from_messages([
    ("human",coding_prompt),
    # ("human", "{input}")
])


def get_greeting_prompt_template(interview_type, payload):
    if(interview_type == "Company"):
        return ChatPromptTemplate.from_messages([
            ("system", company_greeting_prompt.format(Company = payload)),
        # ("human", "{input}")
        ])

    return ChatPromptTemplate.from_messages([
            ("system", subject_greeting_prompt.format(topic = payload)),
        # ("human", "{input}")
        ])


google_search_prompt_template = ChatPromptTemplate.from_messages([
    ("system", google_search_prompt),
    # ("human", "Lets start")
])

S = TypeVar("S")

def create_research_summary_node(Summarize_llm) -> Callable:
    def _Node(State:S) -> S:
        research = State['QuestionResearch']
        difficulty = State['Difficulty']
        tags = State['Tags']
        prompt = research_summarize_prompt.format(research=research,difficulty=difficulty,tags=tags)
        response = Summarize_llm.invoke(prompt)
        State['QuestionResearch'] = response
        print("Research response", response)
        return State
    return _Node

def create_dummy_node() -> Callable:
    def _node(state: S) -> S:
        return state
    return _node

def create_offend_end_node(llm) -> Callable:
    def _Node(State: S) -> S:
        history = State['history']
        response = llm.invoke(Offensive_responsive_prompt.format(history=history))
        State['messages'].append(response)
        State['LastNode'] = "Offense"
        return State
    return _Node



class ToolNode(BaseModel):
    model_config = ConfigDict(extra='allow')
    tools: Annotated[List[Callable], Field(description="List of tools to be used")]
    key: Annotated[str, Field(description="Key in the state where the tool calls are to be made")]

    @field_validator("key")
    @classmethod
    def validate_key(cls, v):
        if not isinstance(v, str):
            raise ValueError("Key must be a string")
        return v

    @field_validator("tools")
    @classmethod
    def validate_tools(cls, v):
        for i, tool in enumerate(v):
            if not callable(tool):
                raise ValueError(f"Tool {i} is not a callable")
            if not inspect.isfunction(tool):
                raise ValueError(f"Tool {i} is not a function")
        return v

    def __init__(self, tools: List[Callable], key: str, *args, **kwargs):
        super().__init__(tools=tools, key=key, *args, **kwargs)
        self.tools = tools
        self.tool_names = {f"{tool.__name__}": tool for tool in tools}

    def __call__(self, state: S) -> S:
        latest_message = state[self.key][-1]
        if not getattr(latest_message, "tool_calls", None):
            return state

        output = []
        for tool_call in latest_message.tool_calls:
            tool_result = self.tool_names[tool_call["name"]](**tool_call["args"])
            output.append(
                ToolMessage(
                    content=str(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {self.key: output}


def make_search_tool(tavily_api_key: str, max_results: int = 5):
    search = TavilySearch(max_results=max_results, topic="general", tavily_api_key=tavily_api_key, include_answer=True)

    def get_google_search(query: str):
        "Call to perform google search online and get reliable results"
        return search.invoke({"query": query})

    return get_google_search



def create_route_to_greeting(InterviewProgress_llm) -> Callable:
    def _Node(state: S) -> Literal['Greeting', 'Coding_before']:
        response = InterviewProgress_llm.invoke(state["history"])
        print("This is the greeting routing node", response.send_to_which_node)
        return response.send_to_which_node
    return _Node


def create_greeting_node(interview_type, Greeting_llm) -> Callable:
  def _Node(state: S) -> S:
    if state["LastNode"] != "Greeting":
      inp_company = getattr(state, "company", None)
      inp_state = getattr(state, "subject", None)
      greeting_prompt = get_greeting_prompt_template(interview_type, inp_company or inp_state)
      print(greeting_prompt.format_messages())
      input_ = greeting_prompt.format_messages() + [{"role":"human","content":"Start the interview now"}]
      state["messages"] = state["messages"] + input_


    response = Greeting_llm.invoke(state["messages"])

    state["messages"] = state["messages"] + [response]
    state["history"] = state["history"] + "\n" + "Interviewer-" + response.content
    state["LastNode"] = "Greeting"

    # print("We are delivering greetings-->",response)
    return state
  return _Node


def create_coding_node(Coding_llm) -> Callable:
    def _Node(state: S) -> S:
        # print("Coding chh aa gye assi")

        if state["LastNode"] != "Coding":
            input_ = coding_prompt_template.format_messages(questions = state["QuestionResearch"])
            state["messages"][0].content = coding_prompt_temp.format(questions = state["QuestionResearch"])
        # print(state["messages"]
        # state["messages"] = state["messages"] + input_


        response = Coding_llm.invoke(state["messages"])
        print(response)

        state["messages"] = state["messages"] + [response]
        state["history"] = state["history"] + "\n" + "Interviewer-" + response.content
        state["LastNode"] = "Coding"

        return state

    # return {"messages":[response],"LastNode":"Coding"}
    return _Node


def create_end_Node() -> Callable:
    def _node(state:S) -> S:
        state["LastNode"] = "finished"
        print("This is the Last Node")
        return state
    return _node
def create_route_to_coding(CodingProgress_llm) -> Callable:
    def _Node(state:S) -> Literal['Coding', 'End']:
        response = CodingProgress_llm.invoke(state["history"])
        print("This is the coding routing node", response.send_to_which_node)
        return response.send_to_which_node
    return _Node

def create_before_coding_node(llm) -> Callable:
    def _Node(state: S) -> S:
        # print("We have reached here!")
        # prompt = '''You are given a [RESEARCH] peice about coding questions asked in coding interviews, your job is to make it presentable
        #         and concise(meaning you can pick 30-40 questions at random), you can categorize them by topics and after the questions,
        #         you can mark their difficulty.
        #         [RESEARCH]-
        #       '''
        # print("IDss")
        # response = llm.invoke(prompt + state["QuestionResearch"])
        # print(response.content)
        # state["QuestionResearch"] = response.content
        return state
    return _Node

def create_questions_search_node(search_llm) -> Callable:
    def _node(state: S) -> S:
        input_ = google_search_prompt_template.format_messages(company=state["company"])

        # if state["toolCall"]:
        #   input_ = input_ + "Here is the result from google search -> \n\n" +

        if state["LastNode"] == "Coding_before":
            print(state["toolCall"])
            response = search_llm.invoke(state["toolCall"])
            print("This is the response jii!", response)
            state["messages"] = state["messages"] + [HumanMessage(response.content)]
            return state

        else:
            # print("We are here!!")
            state["LastNode"] = "Coding_before"
            state["toolCall"] = input_

            state["toolCall"].append(AIMessage(content="", tool_calls=[{
                'name': 'get_google_search',
                'args': {'query': f'Top 5 latest {state["company"]} coding interview questions'},
                'id': str(uuid4())
            },
                {
                    'name': 'get_google_search',
                    'args': {
                        'query': f'Top 15 most common {state["company"]} coding interview questions GeeksforGeeks Glassdoor'},
                    'id': str(uuid4())
                },
                {
                    'name': 'get_google_search',
                    'args': {'query': f'Most common coding patterns asked at {state["company"]} interview'},
                    'id': str(uuid4())
                }]))

        return state

    return _node


def create_route_to_search(state) -> Literal['Coding_tool', 'Coding']:
    last = state["toolCall"][-1]
    if getattr(last, "tool_calls", None):
        print("What the fuck tool got called")
    return "Coding_tool" if getattr(last, "tool_calls", None) else "Coding"


def get_graph(input_type: str, google_api_key: str, tavily_api_key: str, checkpointer: str):
    llm = get_llm(google_api_key=google_api_key)
    workflow = StateGraph(CompanyInterviewState if input_type == "Company" else SubjectInterviewState)

    # search_tool = make_search_tool(tavily_api_key=os.environ["TAVILY_API_KEY"])
    # llm = llm.bind_tools([search_tool])
    # search_tool_node = make_tool_nodes(search_tool)
    # tool_names = {f"{tool.__name__}":tool for tool in [search_tool]}
    # search_tool_node = ToolNode(tools = [search_tool],key = "toolCall")
    # custom_tool_node(tool_names)
    workflow.add_node("Initial_Research", create_research_summary_node(llm))
    workflow.add_node("Greeting", create_greeting_node(input_type, llm))
    workflow.add_node("Greeting_after", create_dummy_node())
    workflow.add_node("Coding_before", create_before_coding_node(llm))
    # workflow.add_node("Coding_before", create_questions_search_node(llm))
    workflow.add_node("Coding", create_coding_node(llm))
    workflow.add_node("Coding_after", create_dummy_node())
    workflow.add_node("End", create_end_Node())
    workflow.add_node("Offensive", create_offend_end_node(llm))

    # workflow.add_node("Coding_tool",search_tool_node)

    workflow.set_entry_point("Initial_Research")
    workflow.add_edge("Greeting", "Greeting_after")
    # workflow.add_edge("Greeting_after","Coding")
    # workflow.add_edge("Coding_before","Coding")
    workflow.add_edge("Initial_Research", "Greeting")
    workflow.add_edge("Coding", "Coding_after")
    workflow.add_edge("Coding_before", "Coding")
    workflow.add_edge("End", "__end__")
    workflow.add_edge("Offensive", "__end__")
    # workflow.add_conditional_edges("Coding_before",create_route_to_search)
    # workflow.add_conditional_edges("Coding", route_after_coding)
    # workflow.add_edge("Coding_tool","Coding_before")
    workflow.add_conditional_edges("Greeting_after",
                                   create_route_to_greeting(llm.with_structured_output(InterviewProgress)))
    workflow.add_conditional_edges("Coding_after", create_route_to_coding(llm.with_structured_output(CodingProgress)))
    agent = workflow.compile(checkpointer=checkpointer)
    print("In here")
    return agent
