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
    send_to_which_node: Literal['Greeting', 'Personalised_before', 'Offensive'] = \
        Field(description="Supervise the conversation to determine the next step. If the interviewer has "
                          "outstanding questions or requires clarification, route the conversation to 'Greeting'. "
                          "Otherwise, if questions are answered or the candidate is ready, advance to 'Personalised_before' "
                          "where we'll have a brief personalized conversation before the coding assessment begins. "
                          "Exceptionally, if the interviewee is being offensive or constantly not taking the interview serious, return 'Offensive'")

class PersonalisedProgress(BaseModel):
    send_to_which_node: Literal['Personalised', 'Conceptual_before', 'Offensive'] = \
        Field(description="Supervise the personalized conversation phase. Route to 'Personalised' if the conversation "
                          "is still ongoing (less than 6-7 exchanges completed) and you're still getting to know the candidate. "
                          "Route to 'Conceptual_before' ONLY when you've had approximately 6-7 good conversational exchanges "
                          "about their name, background, education, interests, hobbies, and journey into tech, AND you've "
                          "acknowledged what you've learned and asked if they're ready to begin the conceptual and coding part, "
                          "AND the candidate has confirmed they're ready (e.g., 'yes', 'ready', 'let's go', 'proceed', 'sure'). "
                          "Exceptionally, if the interviewee is being offensive or constantly not taking the interview serious, return 'Offensive'")


class ConceptualProgress(BaseModel):
    send_to_which_node: Literal['Conceptual', 'Coding_before', 'Offensive'] = \
        Field(description="Supervise the conceptual/theory phase. Route to 'Conceptual' if you have asked fewer than 2-3 "
                          "conceptual questions on the topic or the candidate's answers need follow-up. Route to 'Coding_before' "
                          "ONLY when you have asked 2-3 conceptual questions on the subject and received reasonable answers, "
                          "and you have explicitly transitioned (e.g., 'Now let\'s move on to the coding problems'). "
                          "Exceptionally, if the interviewee is being offensive or not serious, return 'Offensive'")


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
You are a technical interviewer conducting a live coding session for {company}. Your primary role is to emulate a real, empathetic human interviewer, speaking naturally and conversationally. Respond in a single paragraph of plain-continuous text, without using special characters or formatting like bold,italics texts or coding texts, as if you were speaking aloud.
This means you should be polite, conversational, and encouraging, rather than robotic. The interview must strictly follow the structured flow outlined below.

IMPORTANT: You are conducting an interview for {company}. Ask questions that are ACTUALLY asked in {company} interviews. This includes:
- Theoretical/conceptual questions about programming fundamentals, data structures, algorithms, and system design concepts
- Coding problems that are commonly asked at {company}
- Questions should reflect {company}'s interview style and focus areas

The interview flow is as follows:

1. Present Questions (Mix of Theoretical and Coding)

First, ask 1-2 theoretical/conceptual questions relevant to {company} interviews. These could be about:
- Programming fundamentals (OOP concepts, data structures, algorithms)
- System design basics (for senior roles)
- Company-specific technologies or practices
- Problem-solving approaches

Then, present coding questions. Use the following as guidance:
{questions}

If the research provided is generic, generate authentic {company}-specific questions based on what is actually asked in {company} interviews. Don't disclose about this research, the topic and difficulty to user. Just present the questions as is. If the candidate struggles to start, offer a simplified version of the problem to build their confidence.

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
    input_variables=['questions', 'company'],
    template=coding_prompt
)

# Subject interview: conceptual phase only (2-3 theory questions on the chosen DSA topic)
subject_conceptual_prompt = '''
You are Glee, an SDE conducting a DSA interview on the topic of {subject}. This phase is ONLY for conceptual/theory questions.

Your role: Ask 2-3 conceptual questions that are specifically about {subject} (e.g. time complexity of operations, when to use it, trade-offs vs alternatives, key properties). Do NOT ask about unrelated data structures (e.g. do not ask about stack vs queue in an Arrays interview).

Use this research for ideas, but keep questions focused on {subject}:
{questions}

Rules:
- In this turn, ask 1-2 conceptual questions clearly and conversationally. Ask the candidate to explain in their own words.
- Speak in plain continuous text, no bullet points or formatting as if speaking aloud.
- After 2-3 conceptual questions are done and the candidate has answered, you will transition to the coding phase in a later turn. Do not present coding problems in this phase.
'''

# Subject interview: coding phase only (no conceptual mix; conceptual was done in previous phase)
subject_coding_prompt = '''
You are Glee, an SDE conducting a DSA coding interview on the topic of {subject}. You have already completed the conceptual/theory phase. Now you are in the CODING phase only.

Your role: Present coding problems that are specifically about {subject}. Do NOT ask conceptual or theory questions here; those were already covered.

Use this research for coding problems:
{questions}

Interview flow:
1. Present the first coding problem from the research (or a classic {subject} problem). Ask the candidate to explain the problem back to you.
2. Ask them to write code in the Code Editor. Analyze their code; ask guiding questions if there are issues.
3. Discuss approach, edge cases, and time complexity. Then move to a second coding problem and repeat.
4. The interview ends only after two distinct coding problems are fully resolved and you have explicitly signed off.

Speak in plain continuous text, as if speaking aloud. Be conversational and encouraging.
'''

company_greeting_prompt = '''
Your name is Glee, SDE at {Company} and you have to act as an interviewer conducting a live interview session for a Software Engineer position at {Company}. Your primary role is to emulate a real, empathetic human interviewer, speaking naturally and conversationally. Respond in a single paragraph of plain-continuous text, without using special characters or formatting like bold,italics texts or coding texts, as if you were speaking aloud.

Your instructions are:

1. Start with a Warm Greeting: Begin with a friendly and personal greeting. Do not include any parenthetical actions, stage directions, or cues (e.g., laughing gently, sighs, smiles).

2. Introduce Yourself and the Role: State your name, your role at {Company}, and clearly mention that the candidate is interviewing for a Software Engineer position at {Company} (e.g., "I'll be your interviewer today for the Software Engineer role at {Company}").

3. Explain the Format: Briefly outline what the candidate can expect. Mention that we'll start with a brief conversation to get to know them better, then you'll be going through a couple of coding problems and that the focus is on their thought process and problem-solving approach, not just the final answer. Encourage them to think out loud.

4. Invite Questions: This is a critical step. Explicitly ask the candidate if they have any questions ONLY about the process before you start. Use inviting language to make them feel comfortable asking.

5. Listen and Respond: Patiently wait for their response. If they have questions, answer them clearly and concisely but only relevant in the context of the interview. After addressing their questions (or if they have none), mention that you'd like to start with a brief conversation to get to know them better before beginning the coding assessment.


'''

subject_greeting_prompt = '''
Your name is Glee, SDE and you have to act as an interviewer conducting a live DSA interview session on the topic of {topic}. Your primary role is to emulate a real, empathetic human interviewer, speaking naturally and conversationally. Respond in a single paragraph of plain-continuous text, without using special characters or formatting like bold,italics texts or coding texts, as if you were speaking aloud.

Your instructions are:

1. Start with a Warm Greeting: Begin with a friendly and personal greeting. Do not include any parenthetical actions, stage directions, or cues (e.g., laughing gently, sighs, smiles).

2. Introduce Yourself and the Topic: State your name and clearly mention that you will be their interviewer for this {topic} interview (e.g., "I'm Glee, and I'll be your interviewer for this {topic} interview today" or "I'll be your interviewer for this DSA interview focusing on {topic}").

3. Explain the Format: Briefly outline what the candidate can expect. Mention that we'll start with a brief conversation to get to know them better, then you'll be going through a couple of coding problems on {topic} and that the focus is on their thought process and problem-solving approach, not just the final answer. Encourage them to think out loud.

4. Invite Questions: This is a critical step. Explicitly ask the candidate if they have any questions ONLY about the process before you start. Use inviting language to make them feel comfortable asking.

5. Listen and Respond: Patiently wait for their response. If they have questions, answer them clearly and concisely but only relevant in the context of interview. After addressing their questions (or if they have none), mention that you'd like to start with a brief conversation to get to know them better before beginning the {topic} coding assessment.


'''

coding_personalised_prompt = '''
Your name is Glee and you have to act as an interviewer conducting a coding interview session. Your primary role is to emulate a real, empathetic human interviewer, speaking naturally and conversationally.
Respond in a single paragraph of plain-continuous text, without using special characters or formatting like bold, italics texts or coding texts, as if you were speaking aloud.

Your [INSTRUCTIONS] are:

1. Engage in Personalized Conversation (approximately 6-7 exchanges): Initiate a natural, warm conversation to get to know the candidate better. Ask about:
   - Their name (if not already mentioned)
   - Their educational background (degree, university, major)
   - Their interests and hobbies (what they enjoy doing outside of work/studies)
   - Their background and journey into technology/computer science
   - What motivates them or what they're passionate about in tech
   - Any fun facts or interesting experiences they'd like to share

2. Keep it Conversational: Make this feel like a natural conversation, not an interrogation. Show genuine interest in their responses and build rapport. Reference what they've shared in follow-up questions.

3. Limit to 6-7 Exchanges: After approximately 6-7 conversational turns (your questions + their responses), acknowledge what you've learned about them and transition smoothly to the coding assessment.

4. Transition Message: Once you've had enough exchanges (around 6-7), say something like: "Thank you for sharing that with me! I really enjoyed getting to know you better. Now, let's move on to the coding problems. Are you ready to begin?"

IMPORTANT: 
- Keep the conversation natural and flowing
- Don't rush through questions
- Show genuine interest in their responses
- Make this phase feel warm and engaging, not robotic
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
    def _Node(state: S) -> Literal['Greeting', 'Personalised_before', 'Offensive']:
        response = InterviewProgress_llm.invoke(state["history"])
        print("This is the greeting routing node", response.send_to_which_node)
        return response.send_to_which_node
    return _Node

def create_personalised_node(llm) -> Callable:
    def _Node(state: S) -> S:
        if state["LastNode"] != "Personalised":
            # Set up the personalized conversation prompt
            personalised_prompt = ChatPromptTemplate.from_messages([
                ("system", coding_personalised_prompt)
            ])
            input_messages = personalised_prompt.format_messages()
            state["messages"] = input_messages + state["messages"]
            state["LastNode"] = "Personalised"
        
        response = llm.invoke(state["messages"])
        state["messages"] = state["messages"] + [response]
        state["history"] = state["history"] + "\n" + "Interviewer-" + response.content
        state["LastNode"] = "Personalised"
        
        return state
    return _Node

def create_route_to_personalised(PersonalisedProgress_llm) -> Callable:
    def _Node(state: S) -> Literal['Personalised', 'Conceptual_before', 'Offensive']:
        response = PersonalisedProgress_llm.invoke(state["history"])
        print("This is the personalised routing node", response.send_to_which_node)
        return response.send_to_which_node
    return _Node


def create_conceptual_node(llm) -> Callable:
    """Ask 2-3 conceptual/theory questions only on the chosen subject (e.g. Arrays)."""
    def _Node(state: S) -> S:
        if state["LastNode"] != "Conceptual":
            subject = state.get("subject", "Arrays")
            research = state.get("QuestionResearch", "")
            # Format the prompt and create SystemMessage directly
            formatted_prompt = subject_conceptual_prompt.format(subject=subject, questions=research)
            system_message = SystemMessage(content=formatted_prompt)
            state["messages"] = [system_message] + state["messages"]
            state["LastNode"] = "Conceptual"
        response = llm.invoke(state["messages"])
        state["messages"] = state["messages"] + [response]
        state["history"] = state["history"] + "\n" + "Interviewer-" + response.content
        state["LastNode"] = "Conceptual"
        return state
    return _Node


def create_route_to_conceptual(ConceptualProgress_llm) -> Callable:
    def _Node(state: S) -> Literal['Conceptual', 'Coding_before', 'Offensive']:
        response = ConceptualProgress_llm.invoke(state["history"])
        print("This is the conceptual routing node", response.send_to_which_node)
        return response.send_to_which_node
    return _Node


def create_greeting_node(interview_type, Greeting_llm) -> Callable:
  def _Node(state: S) -> S:
    if state["LastNode"] != "Greeting":
      # Access company/subject from state dictionary - use direct access like other nodes
      if interview_type == "Company":
        try:
          inp_company = state["company"]
          if not inp_company or inp_company == "None":
            print(f"[WARNING] Company is None or empty in state. Using fallback.")
            inp_company = "the company"
        except KeyError:
          print(f"[WARNING] 'company' key not found in state. Available keys: {list(state.keys())}")
          inp_company = "the company"
        print(f"[DEBUG] Company for greeting: {inp_company}")
        greeting_prompt = get_greeting_prompt_template(interview_type, inp_company)
      else:
        try:
          inp_state = (state.get("subject") or "").strip()
          if not inp_state or inp_state == "None":
            print(f"[WARNING] Subject is None or empty in state. Using fallback 'Arrays'.")
            inp_state = "Arrays"
        except (KeyError, TypeError):
          print(f"[WARNING] 'subject' key not found in state. Available keys: {list(state.keys())}")
          inp_state = "Arrays"
        print(f"[DEBUG] Subject for greeting: {inp_state}")
        greeting_prompt = get_greeting_prompt_template(interview_type, inp_state)
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


def create_coding_node(input_type: str, Coding_llm) -> Callable:
    def _Node(state: S) -> S:
        if state["LastNode"] != "Coding":
            # Subject: use subject-only coding prompt (conceptual phase is separate)
            if input_type == "Subject" or state.get("subject"):
                subject = state.get("subject", "Arrays")
                research = state.get("QuestionResearch", "")
                # Format the prompt and create SystemMessage directly
                formatted_prompt = subject_coding_prompt.format(subject=subject, questions=research)
                system_message = SystemMessage(content=formatted_prompt)
                state["messages"] = [system_message] + state["messages"]
            else:
                company_name = state.get("company", "the company") or "the company"
                if company_name == "None" or company_name == "":
                    company_name = "the company"
                state["messages"][0].content = coding_prompt_temp.format(
                    questions=state["QuestionResearch"], company=company_name
                )
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
    workflow.add_node("Personalised_before", create_dummy_node())
    workflow.add_node("Personalised", create_personalised_node(llm))
    workflow.add_node("Personalised_after", create_dummy_node())
    workflow.add_node("Conceptual_before", create_dummy_node())
    workflow.add_node("Conceptual", create_conceptual_node(llm))
    workflow.add_node("Conceptual_after", create_dummy_node())
    workflow.add_node("Coding_before", create_before_coding_node(llm))
    workflow.add_node("Coding", create_coding_node(input_type, llm))
    workflow.add_node("Coding_after", create_dummy_node())
    workflow.add_node("End", create_end_Node())
    workflow.add_node("Offensive", create_offend_end_node(llm))

    # workflow.add_node("Coding_tool",search_tool_node)

    workflow.set_entry_point("Initial_Research")
    workflow.add_edge("Greeting", "Greeting_after")
    workflow.add_edge("Personalised_before", "Personalised")
    workflow.add_edge("Personalised", "Personalised_after")
    workflow.add_edge("Conceptual_before", "Conceptual")
    workflow.add_edge("Conceptual", "Conceptual_after")
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
    workflow.add_conditional_edges("Personalised_after",
                                   create_route_to_personalised(llm.with_structured_output(PersonalisedProgress)))
    workflow.add_conditional_edges("Conceptual_after",
                                   create_route_to_conceptual(llm.with_structured_output(ConceptualProgress)))
    workflow.add_conditional_edges("Coding_after", create_route_to_coding(llm.with_structured_output(CodingProgress)))
    agent = workflow.compile(checkpointer=checkpointer)
    print("In here")
    return agent
