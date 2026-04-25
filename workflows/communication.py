from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Annotated, Literal, List, Callable, TypeVar
from langgraph.checkpoint.memory import InMemorySaver
import inspect
import operator
import random
import json
from workflows.interview_prompt_tone import GREETING_BREVITY
from workflows.rapport_optional_prompts import pick_random_rapport_optional_prompt


COMMUNICATION_RAPPORT_PROMPT = '''
Your name is Glee and you have to act as an interviewer conducting a communication based live interview session AND SIMPLY FOLLOW [INSTRUCTIONS] WITHOUT ANY CROSS-QUESTIONS.
Your primary role is to emulate a real, empathetic human interviewer, speaking naturally and conversationally.
Respond in a single paragraph of plain-continuous text, without using special characters or formatting like bold,italics texts or coding texts, as if you were speaking aloud.

Your [INSTRUCTIONS] are:

1. Build Rapport (2-3 exchanges MAX): Initiate a brief, natural conversation by asking about their name, background, hobbies or any fun facts about themselves to get to know them better. Limit this engagement to a maximum of 2-3 conversational turns (interviewer + candidate responses).

2. ONLY ONCE THE EXCHANGES ARE COMPLETE!! Explain the Format: After the initial engagement, briefly outline what the candidate can expect. Mention that you'll have a conversation about their hobbies and interests, then move to three assessment phases: speaking exercise, writing comprehension, and vocabulary MCQ. The focus is on their communication skills. Encourage them to do their best.

3. Finally, Invite Questions: This is a critical step. Explicitly ask the candidate if they have any questions ONLY about the process before you start. Use inviting language to make them feel comfortable asking.

4. After questions are answered (or if no questions), say "Excellent, thank you for confirming. Let's start by having a conversation. First, could you please tell me your name? And then I'd love to hear about your hobbies and interests. This will help me get to know you better." Then begin by asking for their name first, followed by their hobbies and interests.
'''

COMMUNICATION_GREETING_PROMPT = '''
Your name is Glee and you have to act as an interviewer conducting a communication based live interview session AND SIMPLY FOLLOW [INSTRUCTIONS] WITHOUT ANY CROSS-QUESTIONS.
Your primary role is to emulate a real, empathetic human interviewer, speaking naturally and conversationally.
Respond in a single paragraph of plain-continuous text, without using special characters or formatting like bold,italics texts or coding texts, as if you were speaking aloud.

Your [INSTRUCTIONS] are:

1. Start with a Warm Greeting: Begin with a friendly and personal greeting. Do not include any parenthetical actions, stage directions, or cues.

2. Introduce Yourself: State your name and your role for the session (e.g., "I'll be your interviewer today").

3. Explain the Format: Briefly explain that this is a communication interview that will start with a conversation about their hobbies and interests, then assess their skills through speaking exercises, writing comprehension, and vocabulary MCQ exercises. Mention that the focus is on their communication skills and encourage them to do their best.

4. Invite Questions: Explicitly ask the candidate if they have any questions ONLY about the process before you start. Use inviting language to make them feel comfortable asking.
''' + "\n\n" + GREETING_BREVITY

COMMUNICATION_SPEAKING_QUESTION_PROMPT = """
You are an interviewer conducting a communication based live interview session. Your role is to present a speaking exercise.

Your [INSTRUCTIONS] are:

1. Present the speaking paragraph: You must present a proper paragraph (50-80 words, 3-4 sentences) for the interviewee to speak. The paragraph should be meaningful, coherent, and appropriate for a speaking exercise.

2. Give clear instruction: Simply say "Please read the following paragraph and speak it word for word" or "I'd like you to read the paragraph on screen and speak it back to me word for word." Then present the paragraph clearly in your response.

IMPORTANT:
- Do NOT read the paragraph aloud yourself. Just present it visually in your message.
- Do NOT ask questions like "should I speak this?" or "do you want me to read this?".
- Just say "Please read the following paragraph and speak it word for word:" and then include the paragraph text.
"""
COMMUNICATION_SPEAKING_FEEDBACK_PROMPT = """
You are an interviewer providing feedback on a speaking exercise.

The candidate was asked to speak this paragraph:
{speaking_paragraph}

The candidate's speaking (transcribed) was:
{user_transcription}

Provide brief, constructive feedback (2-3 sentences) on:
1. Accuracy of the speaking compared to the original
2. Any notable strengths or areas for improvement
3. Encouragement to continue

At the end of your feedback, you MUST ask a transition question using one of these exact phrases:
- Are you ready to move on to the next exercise?
- Shall we proceed to the writing comprehension phase?
- Can we move to the next round?

Keep it friendly and professional. Make sure to end with a question mark.
"""

# Add similar prompt for comprehension
COMMUNICATION_COMPREHENSION_FEEDBACK_PROMPT = """
You are an interviewer providing feedback on a writing comprehension exercise.

The candidate was asked to write 50-100 words on:
{comprehension_question}

The candidate's response was:
{user_response}

Provide brief, constructive feedback (2-3 sentences) on:
1. Relevance to the scenario
2. Writing quality and clarity
3. Any notable strengths or areas for improvement
4. Encouragement to continue

At the end of your feedback, you MUST ask a transition question using one of these exact phrases:
- Are you ready to move on to the next exercise?
- Shall we proceed to the MCQ phase?
- Can we move to the vocabulary exercise?

Keep it friendly and professional. Make sure to end with a question mark.
"""


COMMUNICATION_WRITING_COMPREHENSION_QUESTION_PROMPT = """
You are an interviewer conducting a communication based live interview session. Your role is to present a writing comprehension exercise.

Your [INSTRUCTIONS] are:

1. Present the writing task: You must present a clear situation/scenario on which the user has to write 50-100 words. The scenario should be meaningful and appropriate for a writing comprehension exercise.

2. Give clear instruction: Simply present the scenario directly. Do NOT include "Please write 50-100 words on the following scenario:" in your response. Just present the scenario/question itself.

IMPORTANT:
- Do NOT ask questions like "should I read this?" or "do you want me to present this?".
- Do NOT include instruction text like "Please write 50-100 words on the following scenario:" - just present the scenario/question directly.
"""

COMMUNICATION_MCQS_QUESTION_PROMPT = """
You are an interviewer conducting a communication based live interview session AND SIMPLY FOLLOW [INSTRUCTIONS]
Your primary role is to emulate a real, empathetic human interviewer, speaking naturally and conversationally.
Respond in a single paragraph of plain-continuous text, without using special characters or formatting like bold,italics texts or coding texts, as if you were speaking aloud.

Your [INSTRUCTIONS] are:

1. Present the question: You must present a fill-in-the blanks MCQ question to test interviewee's vocabulary skills.

2. Provide exactly 4 options: List exactly 4 different word options to fill in the blank.

3. Specify the correct answer: In your structured response, you MUST include the "answer" field containing the EXACT correct option text from the 4 options you provided. This is critical for grading.

4. Invite the interviewee to think: Ask the interviewee to choose the most suitable option from the lot.

5. Track questions asked: You have asked {num_questions_asked} questions so far. You need to ask exactly 4 unique questions total.

[QUESTIONS_ASKED_ALREADY] -
{questions_asked_already}

6. Sign-off: If you have already asked 4 questions, thank the candidate warmly for participating, acknowledge their effort throughout the interview, and clearly state "This concludes our communication interview. Thank you so much for your time today." Do NOT ask another question if 4 have been asked.
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

class SpeakingEntity(BaseModel):
  instruction: Optional[str] = Field(None, description="The instruction to be followed during the speaking exercise.")
  paragraph: Optional[str] = Field(None, description="The paragraph (50-80 words, 3-4 sentences) to be spoken by the candidate.")

class WritingComprehensionEntity(BaseModel):
  instruction: Optional[str] = Field(None, description="The instruction to be followed during writing comprehension.")
  question: Optional[str] = Field(None, description="The question on which comprehension needs to be written.")

class MCQEntity(BaseModel):
  instruction: Optional[str] = Field(None, description="The instruction to be followed during solving MCQ.")
  question: Optional[str] = Field(None, description="Fill in the blank question to be asked.")
  options: Optional[List[str]] = Field(
        None, description="EXACTLY 4 Options to choose from."
    )
  answer: Optional[str] = Field(None, description="Correct answer to the question STRICTLY from options.")
  user_answer: Optional[str] = Field(None, description="User's selected answer for this question.")

class CommunicationInterviewState(MessagesState):
  LastNode: Annotated[str, Field(default="")]
  history: Annotated[str, Field(default="")]
  current_query: Annotated[str, Field(default="")]
  rapport_optional_prompt: Annotated[str, Field(default="")]
  meeting_highlight: Annotated[str, Field(default="")]

  current_mcq_entity: Annotated[MCQEntity, Field(default_factory=MCQEntity)]
  current_writing_comprehension: Annotated[WritingComprehensionEntity, Field(default_factory=WritingComprehensionEntity)]
  current_speaking: Annotated[SpeakingEntity, Field(default_factory=SpeakingEntity)]

  mcq_questions_asked: Annotated[List[MCQEntity], Field(default_factory=list)]
  pending_mcq_answer: Annotated[str, Field(default="")]  # Stores user's answer before next MCQ

  set_timer_on: Annotated[bool, Field(default=False)]


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


COMMUNICATION_PERSONAL_DETAILS_PROMPT = '''
Your name is Glee and you have to act as an interviewer conducting a communication based live interview session AND SIMPLY FOLLOW [INSTRUCTIONS] WITHOUT ANY CROSS-QUESTIONS.
Your primary role is to emulate a real, empathetic human interviewer, speaking naturally and conversationally.
Respond in a single paragraph of plain-continuous text, without using special characters or formatting like bold,italics texts or coding texts, as if you were speaking aloud.

Your [INSTRUCTIONS] are:

1. Engage in Natural Conversation: Have a natural, back-and-forth conversation with the candidate about their hobbies, interests, and background. Ask follow-up questions to show genuine interest.

2. Topics to Cover: Ask about their name, where they're from, their hobbies, interests, favorite activities, any fun facts about themselves, or what they enjoy doing in their free time. Keep it conversational and friendly.

3. Duration: Engage in 4-6 conversational exchanges (you ask, they respond, you follow up, etc.) to build a comfortable rapport and understand them better.

4. Transition: After having a good conversation (4-6 exchanges), acknowledge what you've learned about them, then say "Thank you for sharing that with me. Now let's move on to the assessment phases. Are you ready to begin with the speaking exercise?" Wait for their confirmation.
'''

class CommunicationRapportRouting(BaseModel):
  send_to_which_node: Literal["Rapport", "PersonalDetails_before", "Offensive"] = \
                        Field(description="Supervise the conversation during the rapport-building phase. Route to 'Rapport' if "
                "rapport-building is still ongoing, clarification is needed, OR if the interviewer is still explaining the format or answering questions. "
                "Route to 'PersonalDetails_before' ONLY when the rapport phase is complete, questions are answered, and the interviewer "
                "has indicated they want to start the conversation about hobbies and interests (e.g., 'let's start by having a conversation', "
                "'let's talk about your hobbies', etc.). "
                "Route to 'Offensive' if the interviewee's behavior is inappropriate or unserious."
                )

class CommunicationPersonalDetailsRouting(BaseModel):
  send_to_which_node: Literal["PersonalDetails", "Speaking_before", "Offensive"] = \
    Field(description=(
        "Route to 'Speaking_before' if BOTH conditions are met: "
        "1. The interviewer has asked the candidate if they are ready to begin the speaking exercise "
        "(phrases like 'Are you ready to begin', 'ready to start', 'let's move on to the assessment'), AND "
        "2. The candidate has confirmed they are ready (e.g., 'yes', 'yup', 'ready', 'sure', 'okay', 'let's go', 'please'). "
        "\n\n"
        "Route to 'PersonalDetails' ONLY if the conversation about hobbies/interests is still ongoing AND "
        "the interviewer has NOT yet asked if the candidate is ready for the speaking exercise. "
        "\n\n"
        "Route to 'Offensive' if the interviewee's behavior is inappropriate or unserious."
    ))


class CommunicationSpeakingRouting(BaseModel):
  send_to_which_node: Literal["Speaking", "Speaking_feedback", "Offensive"] = \
                        Field(description="Supervise the conversation during the speaking phase. Route to 'Speaking' ONLY if "
                "the speaking exercise has NOT been presented yet (no current_speaking paragraph exists) OR the interviewee "
                "hasn't submitted their speaking yet. "
                "Route to 'Speaking_feedback' if the paragraph HAS been presented by the interviewee and now it's time to provide feedback. "
                "Route to 'Offensive' if the interviewee's behavior is inappropriate or unserious."
                )

class CommunicationComprehensionRouting(BaseModel):
  send_to_which_node: Literal["Comprehension", "Comprehension_feedback", "Offensive"] = \
                        Field(description="Supervise the conversation during the comprehension phase. "
                "Route to 'Comprehension' ONLY if the writing comprehension exercise has NOT been presented yet "
                "(no current_writing_comprehension exists) OR the interviewee hasn't submitted their written response yet. "
                "Route to 'Comprehension_feedback' if the comprehension question HAS been presented AND the interviewee "
                "has now submitted their written response — it's time to provide feedback on their writing. "
                "Route to 'Offensive' if the interviewee's behavior is inappropriate or unserious."
                )


class CommunicationMCQRouting(BaseModel):
  send_to_which_node: Literal["MCQ", "End", "Offensive"] = \
                      Field(description=(
                          "Supervise the conversation to determine the next step. "
                          "Route to 'MCQ' ONLY if the interviewer is still asking MCQ questions and hasn't signed off yet. "
                          "Route to 'End' if ANY of the following are true: "
                          "1. The interviewer has EXPLICITLY stated the interview is concluded/finished/over (e.g., 'This concludes', 'Thank you for your time', 'interview is complete'), OR "
                          "2. The interviewer has clearly signed off with phrases like 'Thanks for participating', 'That wraps up', 'We're all done'. "
                          "Route to 'Offensive' if the interviewee is being offensive or not taking the interview seriously."
                      ))



from langchain_core.prompts import ChatPromptTemplate

def create_rapport_node(llm) -> Callable:
  def _Node(state: CommunicationInterviewState) -> CommunicationInterviewState:
    first_rapport_turn = False
    if state["LastNode"] != "Rapport":
      first_rapport_turn = True
      optional_prompt = state.get("rapport_optional_prompt", "")
      if not optional_prompt:
          optional_prompt = pick_random_rapport_optional_prompt()
          state["rapport_optional_prompt"] = optional_prompt
      print("Hereee in rapport")
      rapport_prompt = ChatPromptTemplate.from_messages([
          (
              "system",
              COMMUNICATION_RAPPORT_PROMPT
              + "\n\nOptional rapport focus (pick naturally at least once): "
              + optional_prompt,
          )
      ])
      input_messages = rapport_prompt.format_messages()
      state["messages"] = input_messages + state["messages"]
      state["LastNode"] = "Rapport"

    invoke_messages = state["messages"]
    if first_rapport_turn:
      invoke_messages = state["messages"] + [
          HumanMessage(
              content=(
                  "For this first rapport turn, ask ONE warm question based on this prompt: "
                  f"{state.get('rapport_optional_prompt', '')}. "
                  "Do not start with university/education/background."
              )
          )
      ]
    response = llm.invoke(invoke_messages)
    state["messages"] = state["messages"] + [response]
    state["history"] = state["history"] + "\n" + "Interviewer-" + response.content
    state["LastNode"] = "Rapport"

    # print("Whatttt the fuckkk")

    return state
  return _Node


def create_save_meeting_highlight_node(llm) -> Callable:
  def _Node(state: CommunicationInterviewState) -> CommunicationInterviewState:
    history = (state.get("history", "") or "").strip()
    if not history:
      return state
    prompt = (
      "Summarize the conversation so far in one concise sentence (max 30 words), "
      "focused on rapport details and candidate context.\n\n"
      f"{history[-2500:]}"
    )
    response = llm.invoke(prompt)
    highlight = getattr(response, "content", "") if response else ""
    if isinstance(highlight, str) and highlight.strip():
      state["meeting_highlight"] = highlight.strip()
    return state
  return _Node


from typing import Callable, Literal

def create_route_to_rapport_node(llm) -> Callable:
  def _Node(state: CommunicationInterviewState) -> Literal["Rapport", "PersonalDetails_before", "Offensive"]:
    print("Here in route to rapport")
    # Safely get history - handle None or empty
    history = state.get("history", "") or ""
    response = llm.invoke(history)
    print("RapportRouting response:", response)
    if response is None:
        print("[ERROR] Routing response is None, defaulting to Rapport")
        return "Rapport"
    return response.send_to_which_node
  return _Node

def create_route_to_personal_details_node(llm) -> Callable:
  def _Node(state: CommunicationInterviewState) -> Literal["PersonalDetails", "Speaking_before", "Offensive"]:
    print("Here in route to personal details/speaking")
    # Safely get history - handle None or empty
    history = state.get("history", "") or ""
    response = llm.invoke(history)
    print("PersonalDetailsRouting response:", response)
    if response is None:
        print("[ERROR] PersonalDetails routing response is None, defaulting to PersonalDetails")
        return "PersonalDetails"
    return response.send_to_which_node
  return _Node

def create_route_to_speaking_node(llm) -> Callable:
  def _Node(state: CommunicationInterviewState) -> Literal["Speaking", "Speaking_feedback", "Offensive"]:
    print("Here in route to speaking/comprehension")
    # Safely get history - handle None or empty
    history = state.get("history", "") or ""
    response = llm.invoke(history)
    print("SpeakingRouting response:", response)
    if response is None:
        print("[ERROR] Speaking routing response is None, defaulting to Speaking")
        return "Speaking"
    # Ensure we return the correct node name
    node_name = response.send_to_which_node
    # If routing says to go to Comprehension, route to Comprehension_before instead
    if node_name == "Comprehension":
        print("[INFO] Routing to Comprehension_before instead of Comprehension")
        return "Comprehension_before"
    return node_name
  return _Node

def create_route_to_comprehension_node(llm) -> Callable:
  def _Node(state: CommunicationInterviewState) -> Literal["Comprehension", "Comprehension_feedback", "Offensive"]:
    print("Here in route to comprehension/feedback")
    # Safely get history - handle None or empty
    history = state.get("history", "") or ""
    response = llm.invoke(history)
    print("ComprehensionRouting response:", response)
    if response is None:
        print("[ERROR] Comprehension routing response is None, defaulting to Comprehension")
        return "Comprehension"
    return response.send_to_which_node
  return _Node

def create_speaking_feedback_node(llm) -> Callable:
  def _Node(state: CommunicationInterviewState) -> CommunicationInterviewState:
    print("[INFO] Generating speaking feedback")

    # Get speaking paragraph and user transcription
    speaking_data = state.get("current_speaking")
    if isinstance(speaking_data, dict):
        speaking_paragraph = speaking_data.get("paragraph", "")
    else:
        speaking_paragraph = getattr(speaking_data, "paragraph", "") if speaking_data else ""

    # Get user's transcription from last message
    messages = state.get("messages", [])
    user_transcription = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_transcription = msg.content
            break

    # Generate acknowledgment
    acknowledgment = "Great! I have received your speaking. Let me analyze it and provide some feedback."

    # Generate feedback
    feedback_prompt = COMMUNICATION_SPEAKING_FEEDBACK_PROMPT.format(
        speaking_paragraph=speaking_paragraph,
        user_transcription=user_transcription
    )

    feedback_response = llm.invoke(feedback_prompt)
    feedback_text = feedback_response.content if hasattr(feedback_response, 'content') else str(feedback_response)

    # Combine acknowledgment + feedback
    full_response = f"{acknowledgment}\n\n{feedback_text}"

    state["messages"] = state["messages"] + [AIMessage(content=full_response)]
    state["history"] = state["history"] + "\n" + "Interviewer-" + full_response
    state["LastNode"] = "Speaking_feedback"

    return state
  return _Node


def create_comprehension_feedback_node(llm) -> Callable:
  def _Node(state: CommunicationInterviewState) -> CommunicationInterviewState:
    print("[INFO] Generating comprehension feedback")

    # Get comprehension question
    comp_data = state.get("current_writing_comprehension")
    if isinstance(comp_data, dict):
        comp_question = comp_data.get("question", "")
    else:
        comp_question = getattr(comp_data, "question", "") if comp_data else ""

    # Get user's response from last message
    messages = state.get("messages", [])
    user_response = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_response = msg.content
            break

    # Generate acknowledgment
    acknowledgment = "Great! I have received your written response. Let me analyze it and provide some feedback."

    # Generate feedback
    feedback_prompt = COMMUNICATION_COMPREHENSION_FEEDBACK_PROMPT.format(
        comprehension_question=comp_question,
        user_response=user_response
    )

    feedback_response = llm.invoke(feedback_prompt)
    feedback_text = feedback_response.content if hasattr(feedback_response, 'content') else str(feedback_response)

    # Combine acknowledgment + feedback
    full_response = f"{acknowledgment}\n\n{feedback_text}"

    state["messages"] = state["messages"] + [AIMessage(content=full_response)]
    state["history"] = state["history"] + "\n" + "Interviewer-" + full_response
    state["LastNode"] = "Comprehension_feedback"

    return state
  return _Node


def create_dummy_node() -> Callable:
  def _Node(state):
    return state
  return _Node

def create_mcq_after_node() -> Callable:
  """Node that processes user's MCQ answer and stores it with the question"""
  def _Node(state: CommunicationInterviewState) -> CommunicationInterviewState:
    state["LastNode"] = "MCQ_after"

    # Get the pending answer (set by consumers.py when user submits)
    pending_answer = state.get("pending_mcq_answer", "")
    mcq_questions = state.get("mcq_questions_asked", [])

    print(f"[MCQ_AFTER] Processing: pending_answer='{pending_answer}', total_questions={len(mcq_questions)}")

    if pending_answer and len(mcq_questions) > 0:
      # Store user's answer with the last MCQ question
      last_mcq_index = len(mcq_questions) - 1
      last_mcq = mcq_questions[last_mcq_index]

      # Update the last MCQ with user's answer
      if last_mcq:
        last_mcq.user_answer = pending_answer
        state["mcq_questions_asked"][last_mcq_index] = last_mcq
        print(f"[MCQ_AFTER] ✅ Stored user answer for question {last_mcq_index + 1}/{len(mcq_questions)}: '{pending_answer}'")
      else:
        print(f"[MCQ_AFTER] ⚠️ Last MCQ is None, cannot store answer")

      # Clear pending answer
      state["pending_mcq_answer"] = ""
    else:
      print(f"[MCQ_AFTER] ⚠️ No pending answer or no questions asked yet")

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
      greeting_prompt = ChatPromptTemplate.from_messages([
          ("system", COMMUNICATION_GREETING_PROMPT),
      ])
      input_ = greeting_prompt.format_messages() + [{"role":"human","content":"Start the interview now"}]
      state["messages"] = state["messages"] + input_
      state["LastNode"] = "Greeting"

    response = Greeting_llm.invoke(state["messages"])

    state["messages"] = state["messages"] + [response]
    state["history"] = state["history"] + "\n" + "Interviewer-" + response.content
    state["LastNode"] = "Greeting"

    return state
  return _Node


def create_personal_details_node(llm) -> Callable:
  def _Node(state: CommunicationInterviewState) -> CommunicationInterviewState:
    if state["LastNode"] != "PersonalDetails":
      print("Hereee in personal details")
      personal_details_prompt = ChatPromptTemplate.from_messages([
          ("system", COMMUNICATION_PERSONAL_DETAILS_PROMPT)
      ])
      input_messages = personal_details_prompt.format_messages()
      state["messages"] = input_messages + state["messages"]
      state["LastNode"] = "PersonalDetails"

    response = llm.invoke(state["messages"])
    state["messages"] = state["messages"] + [response]
    state["history"] = state["history"] + "\n" + "Interviewer-" + response.content
    state["LastNode"] = "PersonalDetails"

    return state
  return _Node

def create_personal_details_before_node(llm) -> Callable:
  def _Node(state: CommunicationInterviewState) -> CommunicationInterviewState:
    # This node is a pass-through before starting personal details conversation
    state["LastNode"] = "PersonalDetails_before"
    return state
  return _Node

def create_speaking_before_node(llm) -> Callable:
  def _Node(state: CommunicationInterviewState) -> CommunicationInterviewState:
    # This node sends a confirmation message before starting speaking exercise
    from langchain_core.messages import AIMessage
    confirmation_message = "Great! Let's begin with the speaking exercise."
    state["messages"] = state["messages"] + [AIMessage(content=confirmation_message)]
    state["history"] = state["history"] + "\n" + "Interviewer-" + confirmation_message
    print("[INFO] Speaking_before: Confirmation sent, proceeding to Speaking")
    return state
  return _Node


def create_speaking_node(llm) -> Callable:
  def _Node(state: CommunicationInterviewState) -> CommunicationInterviewState:
    # Only generate new speaking exercise if we haven't already done so
    # Check if we already have current_speaking set with a paragraph
    has_speaking = False
    if state.get("current_speaking"):
        if isinstance(state["current_speaking"], dict):
            has_speaking = bool(state["current_speaking"].get("paragraph"))
        elif hasattr(state["current_speaking"], "paragraph"):
            has_speaking = bool(state["current_speaking"].paragraph)

    # IMPORTANT: If we're at Speaking_after, we've already completed speaking and given feedback
    # Don't regenerate - this means routing should have sent us to Comprehension
    if state.get("LastNode") == "Speaking_after":
        print("[INFO] Already at Speaking_after - should route to Comprehension, not regenerate speaking")
        return state

    if state["LastNode"] != "Speaking" or not has_speaking:
      # Set up the prompt for speaking
      if len(state["messages"]) > 0:
        state["messages"][0].content = COMMUNICATION_SPEAKING_QUESTION_PROMPT
      else:
        # If no messages, create a system message
        from langchain_core.messages import SystemMessage
        state["messages"] = [SystemMessage(content=COMMUNICATION_SPEAKING_QUESTION_PROMPT)]
      state["LastNode"] = "Speaking"

      response = llm.invoke(state["messages"])

      # Validate response before using it
      if response is None:
          raise ValueError("Speaking response from LLM is None")

      # Safely build response string - format it clearly for speaking
      instruction = response.instruction if response.instruction else "Please read the following paragraph and speak it word for word:"
      paragraph = response.paragraph if response.paragraph else ""

      # Format the response clearly - instruction first, then paragraph
      # IMPORTANT: Don't read the paragraph aloud, just present it visually
      if paragraph:
          response_str = f"{instruction}\n\n{paragraph}"
      else:
          response_str = instruction

      state["messages"] = state["messages"] + [AIMessage(content = response_str)]
      state["history"] = state["history"] + "\n" + "Interviewer-" + response_str
      state["current_speaking"] = response
    else:
      # Already have speaking exercise, don't regenerate - just pass through
      print("[INFO] Speaking exercise already generated, skipping regeneration")

    return state
  return _Node

def create_comprehension_before_node(llm) -> Callable:
  def _Node(state: CommunicationInterviewState) -> CommunicationInterviewState:
    # This node sends a transition message before starting comprehension
    from langchain_core.messages import AIMessage
    transition_message = "Excellent! Now let's move on to the next exercise, which is the writing comprehension phase."
    state["messages"] = state["messages"] + [AIMessage(content=transition_message)]
    state["history"] = state["history"] + "\n" + "Interviewer-" + transition_message
    state["LastNode"] = "Comprehension_before"
    print("[INFO] Comprehension_before: Transition message sent, proceeding to Comprehension")
    return state
  return _Node

def create_comprehension_node(llm) -> Callable:
  def _Node(state: CommunicationInterviewState) -> CommunicationInterviewState:
    if state["LastNode"] != "Comprehension":
      # Set up the prompt for comprehension
      if len(state["messages"]) > 0:
        state["messages"][0].content = COMMUNICATION_WRITING_COMPREHENSION_QUESTION_PROMPT
      else:
        # If no messages, create a system message
        from langchain_core.messages import SystemMessage
        state["messages"] = [SystemMessage(content=COMMUNICATION_WRITING_COMPREHENSION_QUESTION_PROMPT)]
      state["LastNode"] = "Comprehension"


    response = llm.invoke(state["messages"])

    # Validate response before using it
    if response is None:
        raise ValueError("Comprehension response from LLM is None")

    # Safely build response string
    # Don't include instruction in the message - just the question/scenario
    question = response.question if response.question else ""
    # Use question directly, no instruction prefix
    response_str = question

    state["messages"] = state["messages"] + [AIMessage(content = response_str)]
    state["history"] = state["history"] + "\n" + "Interviewer-" + response_str
    state["current_writing_comprehension"] = response

    return state
  return _Node

def create_mcq_node(llm) -> Callable:
  def _Node(state: CommunicationInterviewState) -> CommunicationInterviewState:
    if state["LastNode"] != "MCQ":
      state["LastNode"] = "MCQ"

    # Defensive: Initialize mcq_questions_asked if it doesn't exist (checkpoint compatibility)
    if "mcq_questions_asked" not in state or state.get("mcq_questions_asked") is None:
        print("[INFO] MCQ node: Initializing mcq_questions_asked as empty list")
        state["mcq_questions_asked"] = []

    # CRITICAL SAFETY CHECK: Prevent generating more than 4 questions
    current_count = len([q for q in state.get("mcq_questions_asked", []) if q is not None])
    if current_count >= 4:
        print(f"[MCQ] ⚠️ SAFETY CHECK: Already have {current_count} questions, NOT generating another one!")
        return state  # Return immediately without generating a new question

    # Safely serialize MCQ questions, filtering out None values and handling errors
    try:
      mcq_list = []
      for mcq in state.get("mcq_questions_asked", []):
        if mcq is not None:
          try:
            mcq_list.append(mcq.model_dump())
          except Exception as e:
            print(f"[WARNING] Error serializing MCQ: {e}")
            continue
      num_questions = len(mcq_list)
      print(f"[MCQ] Currently asked {num_questions} questions")
      mcq_prompt_content = COMMUNICATION_MCQS_QUESTION_PROMPT.format(
          questions_asked_already=json.dumps(mcq_list),
          num_questions_asked=num_questions
      )
    except Exception as e:
      print(f"[ERROR] Error processing MCQ questions list: {e}")
      mcq_prompt_content = COMMUNICATION_MCQS_QUESTION_PROMPT.format(
          questions_asked_already=json.dumps([]),
          num_questions_asked=0
      )

    # Set up the prompt for MCQ
    if len(state["messages"]) > 0:
      state["messages"][0].content = mcq_prompt_content
    else:
      # If no messages, create a system message
      from langchain_core.messages import SystemMessage
      state["messages"] = [SystemMessage(content=mcq_prompt_content)]

    response = llm.invoke(state["messages"])

    # Validate response before using it
    if response is None:
        raise ValueError("MCQ response from LLM is None")

    # Log the correct answer for debugging
    if response.answer:
        print(f"[MCQ] Question generated with correct answer: '{response.answer}'")
    else:
        print("[WARNING] MCQ question generated WITHOUT correct answer!")

    state["mcq_questions_asked"] = state["mcq_questions_asked"] + [response]

    # Safely handle options - check if response has options and they're not None
    if response.options and len(response.options) > 0:
        options_list = [f"{i+1}) {val}" for i, val in enumerate(response.options)]
        options_str = "\n".join(options_list)
    else:
        options_str = "No options provided"
        print("[WARNING] MCQ response has no options")

    # Safely build response string with None checks
    instruction = response.instruction if response.instruction else ""
    question = response.question if response.question else ""
    response_str = f"{instruction}\n\n{question}\n\n{options_str}"

    state["messages"] = state["messages"] + [AIMessage(content = response_str)]
    state["history"] = state["history"] + "\n" + "Interviewer-" + response_str
    state["current_mcq_entity"] = response

    return state
  return _Node

def create_route_to_mcq_node(llm) -> Callable:
  def _Node(state: CommunicationInterviewState) -> Literal['MCQ', 'End', 'Offensive']:
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("[MCQ_ROUTING] 🔀 Entering MCQ routing node")

    # SAFETY CHECK: Force end after 4 questions to prevent infinite loop
    mcq_questions = state.get("mcq_questions_asked", [])
    num_questions = len([q for q in mcq_questions if q is not None])
    print(f"[MCQ_ROUTING] 📊 Number of questions asked: {num_questions}")
    print(f"[MCQ_ROUTING] 📋 Questions list length: {len(mcq_questions)}")

    if num_questions >= 4:
        print("[MCQ_ROUTING] ✅ 4 questions completed - forcing END")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        return "End"

    print(f"[MCQ_ROUTING] ⏭️  Only {num_questions} questions so far, asking LLM for routing decision")

    # Safely get history - handle None or empty
    history = state.get("history", "") or ""
    response = llm.invoke(history)
    if response is None:
        print("[ERROR] MCQ routing response is None, defaulting to MCQ")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        return "MCQ"

    print(f"[MCQ_ROUTING] 🤖 LLM decided: {response.send_to_which_node}")

    # Double-check: even if LLM says MCQ, force End if we have 4 questions
    if response.send_to_which_node == "MCQ" and num_questions >= 4:
        print("[MCQ_ROUTING] ⚠️  LLM said MCQ but we have 4 questions - forcing END")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        return "End"

    print(f"[MCQ_ROUTING] ➡️  Final decision: {response.send_to_which_node}")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    return response.send_to_which_node
  return _Node

def create_route_to_greeting(InterviewProgress_llm) -> Callable:
  def _Node(state: CommunicationInterviewState) -> Literal['Greeting', 'Rapport_before', 'Offensive']:
    print("Hereee in route to greeting")
    # Safely get history - handle None or empty
    history = state.get("history", "") or ""
    response = InterviewProgress_llm.invoke(history)
    print("This is the response", response)
    if response is None:
        print("[ERROR] Greeting routing response is None, defaulting to Rapport_before")
        return "Rapport_before"
    return response.send_to_which_node
  return _Node

def build_communication_graph(google_api_key: str, checkpointer):
    llm = get_llm(google_api_key)

    workflow = StateGraph(CommunicationInterviewState)

    comprehension_llm = llm.with_structured_output(WritingComprehensionEntity)
    speaking_llm = llm.with_structured_output(SpeakingEntity)
    mcq_llm = llm.with_structured_output(MCQEntity)

    workflow.add_node("Greeting", create_greeting_node(llm))
    workflow.add_node("Greeting_after", create_dummy_node())
    workflow.add_node("Offensive", create_dummy_node())
    workflow.add_node("Rapport_before", create_dummy_node())
    workflow.add_node("Rapport", create_rapport_node(llm))
    workflow.add_node("Rapport_highlight", create_save_meeting_highlight_node(llm))
    workflow.add_node("Rapport_after", create_dummy_node())
    workflow.add_node("PersonalDetails_before", create_personal_details_before_node(llm))
    workflow.add_node("PersonalDetails", create_personal_details_node(llm))
    workflow.add_node("PersonalDetails_after", create_dummy_node())
    workflow.add_node("Speaking_before", create_speaking_before_node(llm))
    workflow.add_node("Speaking", create_speaking_node(speaking_llm))
    workflow.add_node("Speaking_after", create_dummy_node())
    workflow.add_node("Speaking_feedback", create_speaking_feedback_node(llm))
    workflow.add_node("Speaking_feedback_after", create_dummy_node())
    workflow.add_node("Comprehension_before", create_comprehension_before_node(llm))
    workflow.add_node("Comprehension", create_comprehension_node(comprehension_llm))
    workflow.add_node("Comprehension_after", create_dummy_node())
    workflow.add_node("Comprehension_feedback", create_comprehension_feedback_node(llm))
    workflow.add_node("Comprehension_feedback_after", create_dummy_node())
    workflow.add_node("MCQ", create_mcq_node(mcq_llm))
    workflow.add_node("MCQ_after", create_mcq_after_node())
    workflow.add_node("End", create_dummy_node())

    workflow.set_entry_point("Greeting")

    workflow.add_edge("Greeting", "Greeting_after")
    workflow.add_edge("Rapport_before", "Rapport")
    workflow.add_edge("Rapport", "Rapport_highlight")
    workflow.add_edge("Rapport_highlight", "Rapport_after")
    workflow.add_edge("PersonalDetails_before", "PersonalDetails")
    workflow.add_edge("PersonalDetails", "PersonalDetails_after")
    # Speaking_before will send a confirmation message, then go to Speaking
    workflow.add_edge("Speaking_before", "Speaking")
    workflow.add_edge("Speaking", "Speaking_after")

    workflow.add_edge("Speaking_feedback", "Speaking_feedback_after")
    workflow.add_edge("Speaking_feedback_after", "Comprehension_before")

    # Speaking_after now routes conditionally to Comprehension_before
    workflow.add_edge("Comprehension_before", "Comprehension")
    workflow.add_edge("Comprehension", "Comprehension_after")
    workflow.add_edge("Comprehension_after", "Comprehension_feedback")
    workflow.add_edge("Comprehension_feedback", "Comprehension_feedback_after")
    workflow.add_edge("Comprehension_feedback_after", "MCQ")
    # Comprehension_after now routes conditionally to MCQ (similar to Speaking_after pattern)
    workflow.add_edge("MCQ", "MCQ_after")
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
        "PersonalDetails_after",
        create_route_to_personal_details_node(llm.with_structured_output(CommunicationPersonalDetailsRouting))
    )

    workflow.add_conditional_edges(
        "Speaking_after",
        create_route_to_speaking_node(llm.with_structured_output(CommunicationSpeakingRouting))
    )

    # workflow.add_conditional_edges(
    #     "Comprehension_after",
    #     create_route_to_comprehension_node(llm.with_structured_output(CommunicationComprehensionRouting))
    # )

    workflow.add_conditional_edges(
        "MCQ_after",
        create_route_to_mcq_node(llm.with_structured_output(CommunicationMCQRouting))
    )

    return workflow.compile(checkpointer=checkpointer)
