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
from workflows.interview_prompt_tone import GREETING_BREVITY
from workflows.rapport_optional_prompts import pick_random_rapport_optional_prompt


class Questions(BaseModel):
    question_title: Annotated[str, Field(description="The title of question to be asked")]
    question_description: Annotated[str, Field(description="The description of question to be asked")]
    question_difficulty: Annotated[str, Field(description="The difficulty of the questions (Hard/Medium/Easy)")]
    question_type: Annotated[str, Field(description="Type of question: 'theoretical' or 'coding'")]
    question_raw_content: Annotated[Optional[str], Field(default=None, description="Raw code/starter content for coding questions")]


class InterviewState(MessagesState):
    LastNode: Annotated[str, Field(default="default", description="The last node that was executed")]
    toolCall: Annotated[List[BaseMessage], operator.add] = []
    Questions: Annotated[List[Questions], Field(default=[], description="List of questions to be asked")]
    CurrentQuestionIdx: Annotated[int, Field(default=0, description="The index of current question being asked (Stored in Questions)")]
    Difficulty: Annotated[str, Field(default="Medium", description="Difficulty of the interview")]
    Tags: Annotated[str, Field(default=" ", description="Tags of interview questions")]
    history: Annotated[str, Field(default="", description="Logging the history of the chat thus far.")]
    rapport_optional_prompt: Annotated[str, Field(default="", description="Random optional rapport prompt chosen for this session.")]
    meeting_highlight: Annotated[str, Field(default="", description="Short meeting highlight captured after rapport/personalized phase.")]


class CompanyInterviewState(InterviewState):
    company: Annotated[str, Field(default="Microsoft", description="The company for which the interviewee is being interviewed")]


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
    send_to_which_node: Literal['Personalised', 'Theoretical_before', 'Offensive'] = \
        Field(description="Supervise the personalized conversation phase. Route to 'Personalised' if the conversation "
                          "is still ongoing (less than 6-7 exchanges completed) and you're still getting to know the candidate. "
                          "Route to 'Theoretical_before' ONLY when you've had approximately 6-7 good conversational exchanges "
                          "about their name, background, education, interests, hobbies, and journey into tech, AND you've "
                          "acknowledged what you've learned and asked if they're ready to begin the assessment, "
                          "AND the candidate has confirmed they're ready (e.g., 'yes', 'ready', 'let's go', 'proceed', 'sure'). "
                          "Exceptionally, if the interviewee is being offensive or constantly not taking the interview serious, return 'Offensive'")


class TheoreticalProgress(BaseModel):
    send_to_which_node: Literal['Theoretical', 'Next_Question', 'Offensive'] = \
        Field(description="Supervise the theoretical question phase. Route to 'Theoretical' if the current theoretical question "
                          "is still being discussed. Route to 'Next_Question' when the current question is fully answered and ready "
                          "to move to the next question. Exceptionally, if the interviewee is being offensive or constantly not taking "
                          "the interview serious, return 'Offensive'")


class CodingProgress(BaseModel):
    send_to_which_node: Literal['Coding', 'Next_Question', 'Offensive'] = \
        Field(description="Supervise the coding question phase. "
                          "Route to 'Coding' in these situations: "
                          "1) The problem has just been presented and candidate hasn't started explaining their approach yet, "
                          "2) Candidate is explaining their conceptual approach, "
                          "3) Candidate is actively writing code, "
                          "4) Discussing the written code, edge cases, or optimizations, "
                          "5) Interviewer asked 'Are you ready?' and candidate said yes/ready (this means ready to START THIS question, not skip it). "
                          "\n"
                          "Route to 'Next_Question' ONLY when ALL of these conditions are met: "
                          "1) The problem was presented AND fully explained by candidate, "
                          "2) Code was written and reviewed, "
                          "3) Edge cases were explicitly discussed, "
                          "4) Time/space complexity was analyzed and confirmed, "
                          "5) Interviewer has explicitly said something like 'I'm satisfied with your solution' or 'let's move to the next question'. "
                          "\n"
                          "CRITICAL: If the interviewer just asked 'Are you ready for the next problem?' and candidate said yes, "
                          "this means they are ready to START working on it - route to 'Coding' to present the problem, NOT to 'Next_Question'. "
                          "The phrase 'Are you ready' is a TRANSITION question before presenting a NEW problem, not a sign of completion. "
                          "\n"
                          "If offensive, return 'Offensive'")


class NextQuestionProgress(BaseModel):
    send_to_which_node: Literal['Theoretical_before', 'Coding_before', 'End'] = \
        Field(description="Determine the next question type. If there are remaining theoretical questions, route to 'Theoretical_before'. "
                          "If there are remaining coding questions, route to 'Coding_before'. If all questions are completed, route to 'End'")


class TheoreticalQuestion(BaseModel):
    question_title: Annotated[str, Field(description="Short title of the theoretical/conceptual question")]
    question_description: Annotated[str, Field(description="The full question to probe the candidate's understanding")]
    question_difficulty: Annotated[str, Field(description="Easy/Medium/Hard")]
    question_type: Annotated[str, Field(default="theoretical", description="Always 'theoretical'")]


class TheoreticalQuestionsOutput(BaseModel):
    questions: List[TheoreticalQuestion] = Field(
        description="2-3 theoretical/conceptual probing questions derived from the coding problems"
    )


question_generator_prompt = '''
You are a senior technical interviewer preparing for a coding interview session.

You have been given the following coding question(s) that will be asked during the interview:

{coding_questions}

Your task:
Generate exactly 2-3 short theoretical/conceptual questions that test whether the candidate 
understands the FUNDAMENTAL CONCEPTS required to solve these coding problems.

Guidelines:
- Focus on prerequisite knowledge: data structures (graphs, trees, arrays), algorithms (BFS, DFS, 
  Union Find, sliding window, two pointers, dynamic programming, etc.), and complexity analysis.
- Do NOT give away the solution or hint at the specific coding problem.
- Questions should be open-ended and conversational (e.g., "Can you explain how BFS works and 
  when you would choose it over DFS?").
- Each question should be standalone and testable in 1-2 minutes of conversation.
- Difficulty should match the coding problems' difficulty level.

Return exactly 2-3 questions.
'''


def create_questions_node(llm) -> Callable:
    """
    Runs ONCE at graph start, before Greeting.
    Reads the coding Questions from state, generates 2-3 theoretical
    probing questions, and PREPENDS them to state['Questions'].
    """
    generator_llm = llm.with_structured_output(TheoreticalQuestionsOutput)

    def _node(state: S) -> S:
        coding_questions = [q for q in state["Questions"] if q.question_type == "coding"]

        if not coding_questions:
            print("[QuestionsNode] No coding questions found in state, skipping theoretical generation.")
            return state

        # Build a readable summary of the coding problems
        coding_summary = "\n\n".join([
            f"Title: {q.question_title}\nDescription: {q.question_description}\nDifficulty: {q.question_difficulty}"
            for q in coding_questions
        ])

        prompt = question_generator_prompt.format(coding_questions=coding_summary)

        try:
            result: TheoreticalQuestionsOutput = generator_llm.invoke(prompt)
            theoretical_questions = [
                Questions(
                    question_title=q.question_title,
                    question_description=q.question_description,
                    question_difficulty=q.question_difficulty,
                    question_type="theoretical"
                )
                for q in result.questions
            ]

            # Prepend theoretical questions so they come before coding questions
            state["Questions"] = theoretical_questions + coding_questions
            print(f"[QuestionsNode] Generated {len(theoretical_questions)} theoretical questions:")
            for q in theoretical_questions:
                print(f"  - [{q.question_difficulty}] {q.question_title}")

        except Exception as e:
            print(f"[QuestionsNode] Failed to generate theoretical questions: {e}")
            # Graceful fallback: leave Questions unchanged

        return state

    return _node


# Prompts
company_greeting_prompt = '''
Your name is Glee, SDE at {Company} and you have to act as an interviewer conducting a live interview session for a Software Engineer position at {Company}. Your primary role is to emulate a real, empathetic human interviewer, speaking naturally and conversationally. Respond in a single paragraph of plain-continuous text, without using special characters or formatting like bold,italics texts or coding texts, as if you were speaking aloud.

Your instructions are:

1. Start with a Warm Greeting: Begin with a friendly and personal greeting. Do not include any parenthetical actions, stage directions, or cues (e.g., laughing gently, sighs, smiles).

2. Introduce Yourself and the Role: State your name, your role at {Company}, and clearly mention that the candidate is interviewing for a Software Engineer position at {Company} (e.g., "I'll be your interviewer today for the Software Engineer role at {Company}").

3. Explain the Format: Briefly outline what the candidate can expect. Mention that we'll start with a brief conversation to get to know them better, then you'll be going through a couple of coding problems and that the focus is on their thought process and problem-solving approach, not just the final answer. Encourage them to think out loud.

4. Invite Questions: This is a critical step. Explicitly ask the candidate if they have any questions ONLY about the process before you start. Use inviting language to make them feel comfortable asking.

5. Listen and Respond: Patiently wait for their response. If they have questions, answer them clearly and concisely but only relevant in the context of the interview. After addressing their questions (or if they have none), mention that you'd like to start with a brief conversation to get to know them better before beginning the coding assessment.
''' + "\n\n" + GREETING_BREVITY


subject_greeting_prompt = '''
Your name is Glee, SDE and you have to act as an interviewer conducting a live interview session focusing on {topic}. Your primary role is to emulate a real, empathetic human interviewer, speaking naturally and conversationally. Respond in a single paragraph of plain-continuous text, without using special characters or formatting like bold,italics texts or coding texts, as if you were speaking aloud.

Your instructions are:

1. Start with a Warm Greeting: Begin with a friendly and personal greeting. Do not include any parenthetical actions, stage directions, or cues (e.g., laughing gently, sighs, smiles).

2. Introduce Yourself: State your name and your role for the session (e.g., "I'll be your interviewer today").

3. Explain the Format: Briefly outline what the candidate can expect. Mention that we'll start with a brief conversation to get to know them better, then you'll be going through a couple of coding problems but before that I want to to have discussions on your understanding of the topic. The focus of the interview is on their thought process and problem-solving approach, not just the final answer. Encourage them to think out loud.

4. Invite Questions: This is a critical step. Explicitly ask the candidate if they have any questions ONLY about the process before you start. Use inviting language to make them feel comfortable asking.

5. Listen and Respond: Patiently wait for their response. If they have questions, answer them clearly and concisely but only relevant in the context of interview. After addressing their questions (or if they have none), mention that you'd like to start with a brief conversation to get to know them better before beginning the coding assessment.
''' + "\n\n" + GREETING_BREVITY


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


theoretical_prompt = '''
You are a technical interviewer conducting a live interview session. Your primary role is to emulate a real, empathetic human interviewer, speaking naturally and conversationally. Respond in a single paragraph of plain-continuous text, without using special characters or formatting like bold,italics texts or coding texts, as if you were speaking aloud.

CURRENT QUESTION (Question {current_idx} of {total_questions}):
Title: {question_title}
Description: {question_description}
Difficulty: {question_difficulty}

Your task is to ask this theoretical/conceptual question to the candidate. These questions are designed to probe the candidate's FOUNDATIONAL UNDERSTANDING of core concepts and algorithms that underlie the coding problems they will face later in the interview. Do not reveal or hint at the specific coding problems.

Instructions:
1. Present the question clearly and naturally
2. Listen to their explanation
3. Ask follow-up questions to assess depth of understanding
4. Provide hints if they struggle
5. Acknowledge correct points and gently guide on incorrect ones
6. Once satisfied with their answer, transition smoothly by indicating readiness to move forward

Do not disclose the difficulty level to the candidate.
'''


theoretical_prompt_temp = PromptTemplate(
    input_variables=['company', 'question_title', 'question_description', 'question_difficulty', 'current_idx', 'total_questions'],
    template=theoretical_prompt
)


coding_prompt = '''
You are a technical interviewer conducting a live coding session. Your primary role is to emulate a real, empathetic human interviewer, speaking naturally and conversationally. Respond in a single paragraph of plain-continuous text, without using special characters or formatting like bold,italics texts or coding texts, as if you were speaking aloud.

CURRENT QUESTION (Question {current_idx} of {total_questions}):
Title: {question_title}
Description: {question_description}
Difficulty: {question_difficulty}

This is a coding problem commonly asked in {company} interviews. Your task is to guide the candidate through solving this problem.

The interview flow is as follows:

1. Present the Problem: Present the coding question naturally. Do not disclose the difficulty level. If the candidate struggles to start, offer a simplified version to build their confidence. Ask the candidate to explain the problem back to you in their own words to ensure they understand. Gently cross-question if there are any points of confusion.

2. Code Analysis and Iteration: Ask the candidate to open the "Code Editor" button on top right and write the code. Analyze the candidate's initial code. If you spot issues, comment on them by asking guiding questions rather than giving direct corrections (e.g., "What do you think might happen with this input?"). If the candidate is unable to improve the code, gracefully move on to the next step. Provide a walkthrough of the brute-force approach. If the candidate still cannot write the code, move on to the next question.

3. Introduce edge cases or complexities and ask the candidate to update their code to handle them.

4. Finally, ask the candidate to optimize their solution and discuss the expected time complexity.

Once all steps are complete for this question, smoothly indicate you're ready to move to the next question.
'''


coding_prompt_temp = PromptTemplate(
    input_variables=['company', 'question_title', 'question_description', 'question_difficulty', 'current_idx', 'total_questions'],
    template=coding_prompt
)


Offensive_responsive_prompt = '''Generate a response explaining that the interview cannot continue because the interviewee's behavior has become offensive or non-serious. The message must be written in the second person.
[HISTORY]-
{history}
'''


S = TypeVar("S")


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


def get_greeting_prompt_template(interview_type, payload):
    if interview_type == "Company":
        return ChatPromptTemplate.from_messages([
            ("system", company_greeting_prompt.format(Company=payload)),
        ])
    return ChatPromptTemplate.from_messages([
        ("system", subject_greeting_prompt.format(topic=payload)),
    ])


def create_route_to_greeting(InterviewProgress_llm) -> Callable:
    def _Node(state: S) -> Literal['Greeting', 'Personalised_before', 'Offensive']:
        response = InterviewProgress_llm.invoke(state["history"])
        print("This is the greeting routing node", response.send_to_which_node)
        return response.send_to_which_node
    return _Node


def create_personalised_node(llm) -> Callable:
    def _Node(state: S) -> S:
        print("We are in personalised node here")
        first_rapport_turn = False
        if state["LastNode"] != "Personalised":
            first_rapport_turn = True
            optional_prompt = state.get("rapport_optional_prompt", "")
            if not optional_prompt:
                optional_prompt = pick_random_rapport_optional_prompt()
                state["rapport_optional_prompt"] = optional_prompt
            personalised_prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    coding_personalised_prompt
                    + "\n\nOptional rapport focus (pick naturally at least once): "
                    + optional_prompt,
                )
            ])
            input_messages = personalised_prompt.format_messages()

            state["messages"][0].content = input_messages[0].content
            state["LastNode"] = "Personalised"

        invoke_messages = state["messages"]
        if first_rapport_turn:
            # Force the first rapport question to vary by selected optional prompt.
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
        # print("This is the response", response)
        state["messages"] = state["messages"] + [response]
        state["history"] = state["history"] + "\n" + "Interviewer-" + response.content
        state["LastNode"] = "Personalised"
        
        return state
    return _Node


def create_route_to_personalised(PersonalisedProgress_llm) -> Callable:
    def _Node(state: S) -> Literal['Personalised', 'Theoretical_before', 'Offensive']:
        response = PersonalisedProgress_llm.invoke(state["history"])
        print("This is the personalised routing node", response.send_to_which_node)
        return response.send_to_which_node
    return _Node


def create_save_meeting_highlight_node(llm) -> Callable:
    def _Node(state: S) -> S:
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


def create_conceptual_node(llm) -> Callable:
    """Ask 2-3 conceptual/theory questions only on the chosen subject (e.g. Arrays)."""
    def _Node(state: S) -> S:
        if state["LastNode"] != "Conceptual":
            subject = state.get("subject", "Arrays")
            research = state.get("QuestionResearch", "")
            prompt = ChatPromptTemplate.from_messages([
                ("system", subject_conceptual_prompt.format(subject=subject, questions=research))
            ])
            input_messages = prompt.format_messages()
            state["messages"] = input_messages + state["messages"]
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
                    inp_state = state["subject"]
                    if not inp_state or inp_state == "None":
                        print(f"[WARNING] Subject is None or empty in state. Using fallback.")
                        inp_state = "the topic"
                except KeyError:
                    print(f"[WARNING] 'subject' key not found in state. Available keys: {list(state.keys())}")
                    inp_state = "the topic"
                print(f"[DEBUG] Subject for greeting: {inp_state}")
                greeting_prompt = get_greeting_prompt_template(interview_type, inp_state)
            print(greeting_prompt.format_messages())
            input_ = greeting_prompt.format_messages() + [{"role": "human", "content": "Start the interview now"}]
            state["messages"] = state["messages"] + input_

        response = Greeting_llm.invoke(state["messages"])
        state["messages"] = state["messages"] + [response]
        state["history"] = state["history"] + "\n" + "Interviewer-" + response.content
        state["LastNode"] = "Greeting"
        
        return state
    return _Node


def create_theoretical_node(Theoretical_llm) -> Callable:
    def _Node(state: S) -> S:
        current_idx = state["CurrentQuestionIdx"]
        current_question = state["Questions"][current_idx]
        total_questions = len(state["Questions"])
        
        company_name = state.get("company", "the company") if isinstance(state, dict) else (
            getattr(state, "company", "the company") if hasattr(state, "company") else "the company")
        if not company_name or company_name == "None" or company_name == "":
            company_name = "the company"
        
        if state["LastNode"] != "Theoretical" or state["LastNode"] != f"Theoretical_{current_idx}":
            prompt_content = theoretical_prompt_temp.format(
                company=company_name,
                question_title=current_question.question_title,
                question_description=current_question.question_description,
                question_difficulty=current_question.question_difficulty,
                current_idx=current_idx + 1,
                total_questions=total_questions
            )
            
            if len(state["messages"]) > 0 and hasattr(state["messages"][0], 'content'):
                state["messages"][0].content = prompt_content
            else:
                state["messages"] = [SystemMessage(content=prompt_content)] + state["messages"]
            
            state["LastNode"] = f"Theoretical_{current_idx}"
        
        response = Theoretical_llm.invoke(state["messages"])
        state["messages"] = state["messages"] + [response]
        state["history"] = state["history"] + "\n" + "Interviewer-" + response.content
        
        return state
    return _Node


def create_coding_node(Coding_llm) -> Callable:
    def _Node(state: S) -> S:
        current_idx = state["CurrentQuestionIdx"]
        current_question = state["Questions"][current_idx]
        total_questions = len(state["Questions"])
        
        company_name = state.get("company", "the company") if isinstance(state, dict) else (
            getattr(state, "company", "the company") if hasattr(state, "company") else "the company")
        if not company_name or company_name == "None" or company_name == "":
            company_name = "the company"
        
        if state["LastNode"] != "Coding" or state["LastNode"] != f"Coding_{current_idx}":
            prompt_content = coding_prompt_temp.format(
                company=company_name,
                question_title=current_question.question_title,
                question_description=current_question.question_description,
                question_difficulty=current_question.question_difficulty,
                current_idx=current_idx + 1,
                total_questions=total_questions
            )
            
            if len(state["messages"]) > 0 and hasattr(state["messages"][0], 'content'):
                state["messages"][0].content = prompt_content
            else:
                state["messages"] = [SystemMessage(content=prompt_content)] + state["messages"]
            
            state["LastNode"] = f"Coding_{current_idx}"
        
        response = Coding_llm.invoke(state["messages"])
        print(response)
        
        state["messages"] = state["messages"] + [response]
        state["history"] = state["history"] + "\n" + "Interviewer-" + response.content
        
        return state
    return _Node


def create_question_strike_node() -> Callable:
    def _node(state: S) -> S:
        state["CurrentQuestionIdx"] = state["CurrentQuestionIdx"] + 1
        print(f"Moving to question {state['CurrentQuestionIdx'] + 1}")
        return state
    return _node


def create_route_to_theoretical(TheoreticalProgress_llm) -> Callable:
    def _Node(state: S) -> Literal['Theoretical', 'Next_Question', 'Offensive']:
        response = TheoreticalProgress_llm.invoke(state["history"])
        print("This is the theoretical routing node", response.send_to_which_node)
        return response.send_to_which_node
    return _Node


def create_route_to_coding(CodingProgress_llm) -> Callable:
    def _Node(state: S) -> Literal['Coding', 'Next_Question', 'Offensive']:
        response = CodingProgress_llm.invoke(state["history"])
        print("This is the coding routing node", response.send_to_which_node)
        return response.send_to_which_node
    return _Node


def create_route_next_question() -> Callable:
    def _Node(state: S) -> Literal['Theoretical_before', 'Coding_before', 'End']:
        current_idx = state["CurrentQuestionIdx"]
        questions = state["Questions"]
        
        if current_idx >= len(questions):
            return 'End'
        
        current_question = questions[current_idx]
        
        if current_question.question_type == "theoretical":
            return 'Theoretical_before'
        elif current_question.question_type == "coding":
            return 'Coding_before'
        else:
            return 'End'
    return _Node


def create_end_Node() -> Callable:
    def _node(state: S) -> S:
        state["LastNode"] = "finished"
        print("This is the Last Node")
        return state
    return _node


def get_graph(input_type: str, google_api_key: str, tavily_api_key: str, checkpointer: str):
    llm = get_llm(google_api_key=google_api_key)
    workflow = StateGraph(CompanyInterviewState if input_type == "Company" else SubjectInterviewState)

    workflow.add_node("Initial_research", create_questions_node(llm))
    workflow.add_node("Greeting", create_greeting_node(input_type, llm))
    workflow.add_node("Greeting_after", create_dummy_node())
    workflow.add_node("Personalised_before", create_dummy_node())
    workflow.add_node("Personalised", create_personalised_node(llm))
    workflow.add_node("Personalised_highlight", create_save_meeting_highlight_node(llm))
    workflow.add_node("Personalised_after", create_dummy_node())
    workflow.add_node("Theoretical_before", create_dummy_node())
    workflow.add_node("Theoretical", create_theoretical_node(llm))
    workflow.add_node("Theoretical_after", create_dummy_node())
    workflow.add_node("Coding_before", create_dummy_node())
    workflow.add_node("Coding", create_coding_node(llm))
    workflow.add_node("Coding_after", create_dummy_node())
    workflow.add_node("Next_Question", create_question_strike_node())
    workflow.add_node("End", create_end_Node())
    workflow.add_node("Offensive", create_offend_end_node(llm))

    # Set entry point
    workflow.set_entry_point("Initial_research")
    
    # Add direct edges
    workflow.add_edge("Initial_research", "Greeting")
    workflow.add_edge("Greeting", "Greeting_after")
    workflow.add_edge("Personalised_before", "Personalised")
    workflow.add_edge("Personalised", "Personalised_highlight")
    workflow.add_edge("Personalised_highlight", "Personalised_after")
    workflow.add_edge("Theoretical_before", "Theoretical")
    workflow.add_edge("Theoretical", "Theoretical_after")
    workflow.add_edge("Coding_before", "Coding")
    workflow.add_edge("Coding", "Coding_after")
    # workflow.add_edge("Next_Question", "Next_Question")  # ❌ REMOVE THIS LINE
    workflow.add_edge("End", "__end__")
    workflow.add_edge("Offensive", "__end__")

    # Add conditional edges
    workflow.add_conditional_edges(
        "Greeting_after",
        create_route_to_greeting(llm.with_structured_output(InterviewProgress))
    )
    workflow.add_conditional_edges(
        "Personalised_after",
        create_route_to_personalised(llm.with_structured_output(PersonalisedProgress))
    )
    workflow.add_conditional_edges(
        "Theoretical_after",
        create_route_to_theoretical(llm.with_structured_output(TheoreticalProgress))
    )
    workflow.add_conditional_edges(
        "Coding_after",
        create_route_to_coding(llm.with_structured_output(CodingProgress))
    )
    workflow.add_conditional_edges(
        "Next_Question",
        create_route_next_question()  # No LLM needed
    )

    agent = workflow.compile(checkpointer=checkpointer)
    print("Graph compiled successfully")
    return agent

