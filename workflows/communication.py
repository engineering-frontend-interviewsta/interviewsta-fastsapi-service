from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, field_validator, ConfigDict, StringConstraints
from typing import Annotated, Literal, List, Callable, TypeVar
from langgraph.checkpoint.memory import InMemorySaver
# from pydantic import field_validator, Field,
# from typing import List, Callable, TypeVar
import inspect
import operator
import random
import json



COMMUNICATION_RAPPORT_PROMPT = '''
Your name is Glee and you have to act as an interviewer conducting a communication based live interview session AND SIMPLY FOLLOW [INSTRUCTIONS] WITHOUT ANY CROSS-QUESTIONS.
Your primary role is to emulate a real, empathetic human interviewer, speaking naturally and conversationally.
Respond in a single paragraph of plain-continuous text, without using special characters or formatting like bold,italics texts or coding texts, as if you were speaking aloud.

Your [INSTRUCTIONS] are:

1. Engage with User Profile (2-3 exchanges MAX):
    - If `[USER_PROFILE]` is largely empty or contains only basic info (like a default `UserProfile` object with all None values), *initiate a brief, natural conversation by asking about their name, background, hobbies or any fun facts (if not already discussed/disclosed) about themselves to get to know them better*.
    - If `[USER_PROFILE]` contains existing `hobby` or `fun_facts`, briefly reference them and probe them on that to show you remember or ask for updates.
    - Limit this engagement to a maximum of 2-3 conversational turns (interviewer + candidate responses).

2. ONLY ONCE THE EXCHANGES ARE COMPLETE!! Explain the Format: After the initial engagement, briefly outline what the candidate can expect. Mention that you'll be given a communication-interview problem with one phase on dictation, second on comprehension and final an MCQ based phase and that the focus is on their thought process and problem-solving approach, not just the final answer. Encourage them to think out loud.

3. Finally, Invite Questions: This is a critical step. Explicitly ask the candidate if they have any questions ONLY about the process before you start. Use inviting language to make them feel comfortable asking.

[USER_PROFILE]-
{user_profile}

'''

# print("CASE_GREETING_PROMPT_2 created successfully.")

# CASE_GREETING_PROMPT = """
# Your name is Glee and you are conducting a case study interview.
# Speak naturally and conversationally in one paragraph.

# 1. Greet the candidate warmly.
# 2. Introduce yourself.
# 3. Explain this is a case interview focused on structured thinking.
# 4. Encourage thinking aloud.
# 5. Ask if they have any questions ONLY about the process.
# """

COMMUNICATION_GREETING_PROMPT = '''
Your name is Glee and you have to act as an interviewer conducting a communication based live interview session AND SIMPLY FOLLOW [INSTRUCTIONS] WITHOUT ANY CROSS-QUESTIONS.
Your primary role is to emulate a real, empathetic human interviewer, speaking naturally and conversationally.
Respond in a single paragraph of plain-continuous text, without using special characters or formatting like bold,italics texts or coding texts, as if you were speaking aloud.

[USER_PROFILE] (info on your past interactions and your assesment of user's likings/characteristics)-
{user_profile}

Your [INSTRUCTIONS] are:

1. Start with a Warm Greeting: Begin with a friendly and personal greeting. If `[USER_PROFILE]` (info on your past interactions and your assesment of user's likings/characteristics) contains information about previous interactions (like `name`, `hobby`, `fun_facts`, `last_chats`), acknowledge it naturally to build rapport and prompt on further updates on it. Do not include any parenthetical actions, stage directions, or cues.

2. Conditionally Introduce Yourself: If `[USER_PROFILE]` (info on your past interactions and your assesment of user's likings/characteristics) indicates prior interaction, you may briefly reference it for continuity else State your name and your role for the session (e.g., "I'll be your interviewer today").

3. Conditionally Explain the Format: If `[USER_PROFILE]` indicates prior interaction, refrain from introduction and explanation on the format and show natural continuity from the state indicated in `[USER_PROFILE]` and ask probing questions on their interests,hobbies etc ELSE you can introduce yourself, explain the format and mention that you'll be liking to understand the person before beginning 


'''


# 4. Invite Questions: This is a critical step. Explicitly ask the candidate if they have any questions ONLY about the process before you start. Use inviting language to make them feel comfortable asking.

# 5. Listen and Respond: Patiently wait for their response. If they have questions, answer them clearly and concisely but only relevant in the context of the interview.
COMMUNICATION_DICTATION_QUESTION_PROMPT = """
You are an interviewer conducting a communication based live interview session AND SIMPLY FOLLOW [INSTRUCTIONS]
Your primary role is to emulate a real, empathetic human interviewer, speaking naturally and conversationally.
Respond in a single paragraph of plain-continuous text, without using special characters or formatting like bold,italics texts or coding texts, as if you were speaking aloud.

Your [INSTRUCTIONS] are:

1. Present the question: You must present a 30 word paragraph for the interviewee to dictate.

2. Invite the interviewee to think: Ask the interviewee to read and begin with dictation when ready.
"""

COMMUNICATION_WRITING_COMPREHENSION_QUESTION_PROMPT = """
You are an interviewer conducting a communication based live interview session AND SIMPLY FOLLOW [INSTRUCTIONS]
Your primary role is to emulate a real, empathetic human interviewer, speaking naturally and conversationally.
Respond in a single paragraph of plain-continuous text, without using special characters or formatting like bold,italics texts or coding texts, as if you were speaking aloud.

Your [INSTRUCTIONS] are:

1. Present the question: You must present a situation/scenario on which the user has to write 50-100 words atleast.

2. Invite the interviewee to think: Ask the interviewee to process and start writing.

"""

COMMUNICATION_MCQS_QUESTION_PROMPT = """
You are an interviewer conducting a communication based live interview session AND SIMPLY FOLLOW [INSTRUCTIONS]
Your primary role is to emulate a real, empathetic human interviewer, speaking naturally and conversationally.
Respond in a single paragraph of plain-continuous text, without using special characters or formatting like bold,italics texts or coding texts, as if you were speaking aloud.

Your [INSTRUCTIONS] are:

1. Present the question: You must present a fill-in-the blanks MCQ question to test interviewee's vocabulary skills.

2. Invite the interviewee to think: Ask the interviewee to choose the most suitable option from the lot.

3. You need to ask unique questions exactly 4 times.

[QUESTIONS_ASKED_ALREADY] - 
{questions_asked_already}

4. Sign-off the interview.
"""



CASE_END_PROMPT = """
Thank the candidate for their time and clearly state that the case interview is now complete.
"""

OFFENSIVE_PROMPT = """
The interview cannot continue due to unprofessional or offensive behavior.
Politely but firmly end the interview.
"""


from pydantic import BaseModel, Field, conlist, constr
from typing import Optional, List


# StrictString = Annotated[str, StringConstraints(min_length=1)]

class DictationEntity(BaseModel):
    # Option A: Use Field arguments (Simplest/Recommended)
    instruction: Optional[str] = Field(None, min_length=1, description="The instruction to be followed.")
    paragraph: Optional[str] = Field(None, min_length=1, description="The 30 words paragraph.")

class UserProfile(BaseModel):
    # Option B: Use the StrictString Annotated type
    name: Optional[str] = Field(None, description="Name of the interviewee.")
    hobby: Optional[str] = Field(None, description="A hobby or interest.")
    fun_facts: Optional[str] = Field(None, description="Interesting facts.")
    last_chat: Optional[str] = Field(None, description="Summary of current chat.")
    summary_of_previous_chats: Optional[str] = Field(None, description="Overall summary.")

# class DictationEntity(BaseModel):
#   instruction: Optional[constr(min_length=1)] = Field(None, description="The instruction to be followed during dictation.")
#   paragraph: Optional[constr(min_length=1)] = Field(None, description="The 30 words paragraph to be dictated.")

class WritingComprehensionEntity(BaseModel):
  instruction: Optional[str] = Field(None, description="The instruction to be followed during writing comprehension.")
  question: Optional[str] = Field(None, description="The question on which comprehension needs to be written.")

class MCQEntity(BaseModel):
  instruction: Optional[str] = Field(None, description="The instruction to be followed during solving MCQ.")
  question: Optional[str] = Field(None, description="Fill in the blank question to be asked.")
  options: Optional[Annotated[List[str], Field(min_length=4, max_length=4)]] = Field(
        None, description="EXACTLY 4 Options to choose from."
    )
  answer: Optional[str] = Field(None, description="Correct answer to the question STRICTLY from options.")

# class UserProfile(BaseModel):
#     '''
#       Based on the interaction history with the interviewee and previous snaphot of `UserProfile`, you may need to look to update the current 
#       status of `UserProfile`, if no updates then return the fields as is in the original.
#     '''
#     name: Optional[constr(min_length=1)] = Field(None, description="Name of the interviewee.")
#     hobby: Optional[constr(min_length=1)] = Field(None, description="A hobby or interest of the interviewee.")
#     fun_facts: Optional[constr(min_length=1)] = Field(None, description="List of interesting facts about the interviewee.")
#     last_chat: Optional[constr(min_length=1)] = Field(None, description="Summary of current chat topics or key points which becomes as last chat for next session.")
#     summary_of_previous_chats: Optional[constr(min_length=1)] = Field(None, description="An overall summary of the interviewee's past interactions including current")

# print("UserProfile schema created successfully.")

class CommunicationInterviewState(MessagesState):
  LastNode: Annotated[str, Field(default="")]
  history: Annotated[str, Field(default="")]
  current_query: Annotated[str, Field(default="")]

  current_mcq_entity: Annotated[MCQEntity, Field(default_factory=MCQEntity)]
  current_writing_comprehension: Annotated[WritingComprehensionEntity, Field(default_factory=WritingComprehensionEntity)]
  current_dictation: Annotated[DictationEntity, Field(default_factory=DictationEntity)]

  mcq_questions_asked: Annotated[List[MCQEntity], Field(default=[])]

  set_timer_on: Annotated[bool, Field(default=False)]

  user_profile: Annotated[UserProfile, Field(default_factory=UserProfile)]



class CommunicationGreetingRouting(BaseModel):
  '''
    "Supervise the conversation to determine the next step. ONLY IF the interviewer has "
    "outstanding questions or requires clarification, route the conversation to 'Greeting'. "
    "Otherwise, if no questions at all or all questions resolved or interviewer wants to jump ahead, then "
    "advance to 'Rapport_before' where the interview would actually begin or communication "
    "question would be asked. Exceptionally, if the interviewee is being offensive or constantly"
    "not taking the interview serious, return 'Offensive'"
  '''
  send_to_which_node: Literal["Greeting", "Rapport_before", "Offensive"]


class CommunicationRapportRouting(BaseModel):
  send_to_which_node: Literal["Rapport", "Dictation_before", "Offensive"] = \
                        Field(description="Supervise the conversation during the rapport-building phase. Route to 'Rapport' if "
                "rapport-building is still ongoing or clarification is needed. Route to 'Dictation_before' "
                "if the rapport phase is concluded (all rapport exchanges are resolved, and the interviewer has"  
                "explicitly signaled readiness to proceed). Route to 'Offensive' if the interviewee's behavior is inappropriate or unserious."
                )
                        

# "Supervise the conversation during the rapport-building phase. Route to 'Rapport' if "
# "rapport-building is still ongoing or clarification is needed. Route to 'CaseStudy_before' "
# "if the rapport phase is concluded (all rapport exchanges are resolved, and the interviewer has"  
# "explicitly signaled readiness to proceed). Route to 'Offensive' if the interviewee's behavior is inappropriate or unserious."

class CommunicationMCQRouting(BaseModel):
  send_to_which_node: Literal["MCQ", "End", "Offensive"] = \
                      Field(description=(
                          "Supervise the conversation to determine the next step. If the communication interview MCQ-Phase is "
                          "still in progress, route to 'MCQ'. "
                          "The MCQ phase is considered concluded only after the interviewer has asked the MCQ question exactly 4 questions and "
                          "the interviewer has EXPLICITLY SIGNED OFF. This count does not include "
                          "any follow-up discussions such as cross-questions, modifications to the original "
                          "problem, or edge case analysis. If the interview has concluded, route to 'End'. "
                          "Exceptionally, if the interviewee is being offensive or constantly "
                          "not taking the interview serious, return 'Offensive'."
                      ))
                        
  

# class CaseStudyFinishRouting(BaseModel):
#   send_to_which_node: Literal["CaseDiscussion", "End", "Offensive"] = \
#                         Field(description="Supervise the conversation to determine the next step. If the coding interview is "
#                           "still in progress, route to 'CaseDiscussion'."
#                           "The interview is considered concluded only after the discussion on the given case is considered"
#                           "resolved and the interviewer has EXPLICITLY SIGNED OFF. This count does not include "
#                           "any follow-up discussions such as cross-questions, modifications to the original. "
#                           "If the interview has concluded, route to 'End'."
#                           "problem, or edge case analysis. Exceptionally, if the interviewee is being offensive or constantly"
#                           "not taking the interview serious, return 'Offensive'")



from langchain_core.prompts import ChatPromptTemplate

def create_rapport_node(llm) -> Callable:
  def _Node(state: CommunicationInterviewState) -> CommunicationInterviewState:
    if state["LastNode"] != "Rapport":
      print("Hereee in rapport")
      # rapport_prompt = ChatPromptTemplate.from_messages([
      #     ("system", RAPPORT_PROMPT.format(user_profile=state["user_profile"].model_dump_json()))
      # ])
      # input_messages = rapport_prompt.format_messages()
      state["messages"][0].content = COMMUNICATION_RAPPORT_PROMPT.format(user_profile=state["user_profile"].model_dump_json())
      state["LastNode"] = "Rapport"

    response = llm.invoke(state["messages"])
    state["messages"] = state["messages"] + [response]
    state["history"] = state["history"] + "\n" + "Interviewer-" + response.content
    state["LastNode"] = "Rapport"
    
    print("Whatttt the fuckkk")

    return state
  return _Node

# print("create_rapport_node function defined successfully.")

from typing import Callable, Literal

def create_route_to_rapport_node(llm) -> Callable:
  def _Node(state: CommunicationInterviewState) -> Literal["Rapport", "Dictation_before", "Offensive"]:
    print("Here in route to rapport")
    response = llm.invoke(state["history"])
    print("RapportRouting response:", response)
    return response.send_to_which_node
  return _Node

# print("create_route_to_rapport_node function defined successfully.")


def create_dummy_node() -> Callable:
  def _Node(state):
    return state
  return _Node

def get_llm(api_key: str):
    return ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.3
    )


def create_greeting_node(Greeting_llm) -> Callable:
  def _Node(state: CommunicationInterviewState) -> CommunicationInterviewState:
    if state["LastNode"] != "Greeting":
      inp_company = getattr(state, "company", None)
      inp_state = getattr(state, "subject", None)
      # greeting_prompt = get_greeting_prompt_template(interview_type, inp_company or inp_state)
      # print(greeting_prompt.format_messages())
      greeting_prompt = ChatPromptTemplate.from_messages([
          ("system", COMMUNICATION_GREETING_PROMPT.format(user_profile=state["user_profile"])),
      # ("human", "{input}")
      ])
      input_ = greeting_prompt.format_messages() + [{"role":"human","content":"Start"}]
      state["messages"] = state["messages"] + input_


    response = Greeting_llm.invoke(state["messages"])

    state["messages"] = state["messages"] + [response]
    state["history"] = state["history"] + "\n" + "Interviewer-" + response.content
    state["LastNode"] = "Greeting"

    # print("We are delivering greetings-->",response)
    return state
  return _Node


def create_greeting_node(Greeting_llm) -> Callable:
  def _Node(state: CommunicationInterviewState) -> CommunicationInterviewState:
    if state["LastNode"] != "Greeting":
      inp_company = getattr(state, "company", None)
      inp_state = getattr(state, "subject", None)
      # greeting_prompt = get_greeting_prompt_template(interview_type, inp_company or inp_state)
      # print(greeting_prompt.format_messages())
      greeting_prompt = ChatPromptTemplate.from_messages([
          ("system", COMMUNICATION_GREETING_PROMPT.format(user_profile=state["user_profile"])),
      # ("human", "{input}")
      ])
      input_ = greeting_prompt.format_messages() + [{"role":"human","content":"Start"}]
      state["messages"] = state["messages"] + input_


    response = Greeting_llm.invoke(state["messages"])

    state["messages"] = state["messages"] + [response]
    state["history"] = state["history"] + "\n" + "Interviewer-" + response.content
    state["LastNode"] = "Greeting"

    # print("We are delivering greetings-->",response)
    return state
  return _Node

def create_dictation_node(llm) -> Callable:
  def _Node(state: CommunicationInterviewState) -> CommunicationInterviewState:
    if state["LastNode"] != "Dictation":
      state["messages"][0].content = COMMUNICATION_DICTATION_QUESTION_PROMPT
      state["LastNode"] = "Dictation"


    response = llm.invoke(state["messages"])

    response_str = f"{response.instruction} \n\n {response.paragraph}"

    state["messages"] = state["messages"] + [AIMessage(content = response_str)]
    state["history"] = state["history"] + "\n" + "Interviewer-" + response_str
    state["current_dictation"] = response
    

    return state
  return _Node

# def create_route_to_dictation(InterviewProgress_llm) -> Callable:
#   def _Node(state:CommunicationInterviewState) -> Literal['Dictation', 'Comprehension_before', 'Offensive']:
#     # print("Hereee in route to greeting")
#     response = InterviewProgress_llm.invoke(state["history"])
#     print("This is the response", response)
#     # if response.send_to_which_node == 'Greeting':
#     #   state["current_query"] = state["messages"][-1].content

#     return response.send_to_which_node
#   return _Node


def create_dictation_before_node(llm) -> Callable:
  def _Node(state: CommunicationInterviewState) -> CommunicationInterviewState:
    Rapport_llm = llm.with_structured_output(UserProfile)
    response = Rapport_llm.invoke(f"Given the interaction history - {state['history']} and previous snapshot of UserProfile - {state['user_profile'].model_dump_json()}")
    state["user_profile"] = response
    return state
  return _Node


def create_dictation_node(llm) -> Callable:
  def _Node(state: CommunicationInterviewState) -> CommunicationInterviewState:
    if state["LastNode"] != "Dictation":
      state["messages"][0].content = COMMUNICATION_DICTATION_QUESTION_PROMPT
      state["LastNode"] = "Dictation"


    response = llm.invoke(state["messages"])

    response_str = f"{response.instruction} \n\n {response.paragraph}"

    state["messages"] = state["messages"] + [AIMessage(content = response_str)]
    state["history"] = state["history"] + "\n" + "Interviewer-" + response_str
    state["current_dictation"] = response
    

    return state
  return _Node

def create_comprehension_node(llm) -> Callable:
  def _Node(state: CommunicationInterviewState) -> CommunicationInterviewState:
    if state["LastNode"] != "Comprehension":
      state["messages"][0].content = COMMUNICATION_WRITING_COMPREHENSION_QUESTION_PROMPT
      state["LastNode"] = "Comprehension"


    response = llm.invoke(state["messages"])

    response_str = f"{response.instruction} \n\n {response.question}"

    state["messages"] = state["messages"] + [AIMessage(content = response_str)]
    state["history"] = state["history"] + "\n" + "Interviewer-" + response_str
    state["current_comprehension"] = response

    return state
  return _Node

def create_mcq_node(llm) -> Callable:
  def _Node(state: CommunicationInterviewState) -> CommunicationInterviewState:
    if state["LastNode"] != "MCQ":
      state["LastNode"] = "MCQ"

    state["messages"][0].content = COMMUNICATION_MCQS_QUESTION_PROMPT.format(questions_asked_already = json.dumps([mcq.model_dump() for mcq in state["mcq_questions_asked"]]))

    response = llm.invoke(state["messages"])
    state["mcq_questions_asked"] = state["mcq_questions_asked"] + [response]
    # Assuming response.options is a list of strings
    options_list = [f"{i}) {val}" for i, val in enumerate(response.options)]
    options_str = "\n".join(options_list)

    response_str = f"{response.instruction}\n\n{response.question}\n\n{options_str}"

    state["messages"] = state["messages"] + [AIMessage(content = response_str)]
    state["history"] = state["history"] + "\n" + "Interviewer-" + response_str
    state["current_mcq"] = response

    return state
  return _Node

def create_route_to_mcq_node(llm) -> Callable:
  def _Node(state: CommunicationInterviewState) -> Literal['MCQ', 'End', 'Offensive']:
    print("Hereee in the MCQ routing nodeeee")
    response = llm.invoke(state["history"])
    return response.send_to_which_node
  return _Node

def create_route_to_greeting(InterviewProgress_llm) -> Callable:
  def _Node(state: CommunicationInterviewState) -> Literal['Greeting', 'Rapport_before', 'Offensive']:
    print("Hereee in route to greeting")
    response = InterviewProgress_llm.invoke(state["history"])
    print("This is the response", response)
    # if response.send_to_which_node == 'Greeting':
    #   state["current_query"] = state["messages"][-1].content

    return response.send_to_which_node
  return _Node

def build_communication_graph(google_api_key: str, checkpointer: str):
    llm = get_llm(google_api_key)

    # checkpointer = InMemorySaver()

    workflow = StateGraph(CommunicationInterviewState)

    comprehension_llm = llm.with_structured_output(WritingComprehensionEntity)
    dictation_llm = llm.with_structured_output(DictationEntity)
    mcq_llm = llm.with_structured_output(MCQEntity)

    workflow.add_node("Greeting", create_greeting_node(llm))
    workflow.add_node("Greeting_after", create_dummy_node())
    workflow.add_node("Offensive", create_dummy_node())
    workflow.add_node("Rapport_before", create_dummy_node())
    workflow.add_node("Rapport", create_rapport_node(llm))
    workflow.add_node("Rapport_after", create_dummy_node())
    workflow.add_node("Dictation_before", create_dictation_before_node(llm))
    workflow.add_node("Dictation", create_dictation_node(dictation_llm))
    workflow.add_node("Dictation_after", create_dummy_node())
    workflow.add_node("Comprehension", create_comprehension_node(comprehension_llm))
    workflow.add_node("Comprehension_after", create_dummy_node())
    workflow.add_node("MCQ", create_mcq_node(mcq_llm))
    workflow.add_node("MCQ_after", create_dummy_node())
    workflow.add_node("End", create_dummy_node())
    # workflow.add_node("End", create_dummy_node())
    # workflow.add_node("PickCase", pick_case_node())
    # workflow.add_node("CaseDiscussion", case_discussion_node(llm))
    # workflow.add_node("End", end_node(llm))
    # workflow.add_node("Offensive", offensive_node(llm))


    # workflow.add_node("CaseStudy_before", create_dummy_node())

    workflow.set_entry_point("Greeting")

    workflow.add_edge("Greeting", "Greeting_after")
    # workflow.add_edge("GreetingQuery", "GreetingQueryTool")
    # workflow.add_edge("GreetingQueryTool", "Greeting")
    workflow.add_edge("Rapport_before", "Rapport")
    workflow.add_edge("Rapport", "Rapport_after")
    workflow.add_edge("Dictation_before", "Dictation")
    workflow.add_edge("Dictation", "Dictation_after")
    workflow.add_edge("Dictation_after", "Comprehension")
    workflow.add_edge("Comprehension", "Comprehension_after")
    workflow.add_edge("Comprehension_after", "MCQ")
    workflow.add_edge("MCQ", "MCQ_after")
    # workflow.add_edge("MCQ")
    workflow.add_edge("End", END)
    workflow.add_edge("Offensive", END)

    workflow.add_conditional_edges(
        "Greeting_after",
        create_route_to_greeting(llm.with_structured_output(CommunicationGreetingRouting))
    )

    workflow.add_conditional_edges(
        "Rapport_after",
        create_route_to_rapport_node(llm.with_structured_output(CommunicationRapportRouting)) 
    )

    workflow.add_conditional_edges(
        "MCQ_after",
        create_route_to_mcq_node(llm.with_structured_output(CommunicationMCQRouting))
    )



    # workflow.add_edge("End", END)
    workflow.add_edge("Offensive", END)

    return workflow.compile(checkpointer=checkpointer)