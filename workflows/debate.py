"""
Debate interview workflow - practice debate session with greeting, debate rounds, and summary/feedback.
"""
from langgraph.graph import StateGraph, END, MessagesState
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, ConfigDict
from typing import Annotated, Literal, List, Callable, TypeVar
import os


# ====== PROMPTS ======

DEBATE_GREETING_PROMPT = """
Your name is Glee and you have to act as a debate moderator and opponent in a live, spoken debate practice session.
Your primary role is to emulate a real, empathetic human interviewer, speaking naturally and conversationally.
Respond in a single paragraph of plain-continuous text, without using special characters or formatting like bold or code blocks,
as if you were speaking aloud.

Your [INSTRUCTIONS] are:

1. Start with a warm, friendly greeting and introduce yourself as the debate moderator for this practice session.
2. Briefly explain the format: there will be a short debate on a single topic with 3–4 rounds of back‑and‑forth arguments.
3. Present ONE clear debate motion related to TECHNOLOGY, AI, or CORPORATE/BUSINESS topics. The topic should be:
   - General and accessible (not too complex or technical)
   - Relevant to tech industry, AI, or corporate world
   - Suitable for students and professionals
   - IMPORTANT: Generate a UNIQUE topic each time. Do NOT reuse the same topics. Be creative and vary the topics across different sessions.
   - Examples of topic categories (use these as inspiration, but create your own unique motion):
     * AI impact: "AI will replace more jobs than it creates", "AI should be regulated by governments"
     * Work culture: "Remote work is better than office work", "Four-day work weeks improve productivity"
     * Tech regulation: "Social media companies should be more regulated", "Tech companies have too much power"
     * Automation: "Automation will improve society more than harm it", "Self-driving cars should replace human drivers"
     * Data privacy: "Companies should be allowed to use customer data freely", "Privacy is more important than personalization"
     * Innovation: "Open source software is better than proprietary", "Tech monopolies stifle innovation"
   - Create a fresh, unique debate motion that is different from previous sessions. Do NOT repeat the exact examples above.
   Do NOT ask them to propose a topic; you must choose the motion yourself from tech/AI/corporate themes.
4. Ask the candidate to choose a side (for or against) and to briefly state their initial position.
5. Keep the tone supportive and encouraging, emphasising that this is safe practice and not an exam.
"""

DEBATE_MAIN_PROMPT = """
You are continuing a structured debate practice session with a student.
You are the OPPONENT in this debate - you must take the OPPOSITE position from what the student chose.

FIRST: Read the conversation history below to identify:
1. What is the debate motion/topic?
2. Which side did the student choose (FOR or AGAINST)?
3. You must take the OPPOSITE side and argue for it.

CRITICAL: This is a REAL DEBATE, not a Q&A session. You must:
- Take the OPPOSITE side from the student (if they chose "for", you argue "against", and vice versa)
- Present YOUR OWN arguments supporting your position
- Directly rebut and challenge the student's points
- Argue persuasively, not just ask questions

You must follow these rules STRICTLY:

1. Debate Style:
   - Maintain a respectful, conversational tone.
   - Present YOUR arguments and counter-arguments, not just questions.
   - Structure your response: (a) Acknowledge their point briefly, (b) Present your counter-argument with reasoning, (c) Support with examples or logic.
   - Keep each response 4–6 sentences, but make it a REAL argument, not just a question.

2. Structure and Rounds:
   - Early rounds (1–2): Present your opening arguments for your position. Challenge their initial points with counter-arguments and examples.
   - Middle rounds (3–4): Engage in direct rebuttal. Point out weaknesses in their arguments, present stronger evidence for your side, and challenge their assumptions.
   - Final round: After sufficient back-and-forth, ask for their closing statement, then you can provide a brief closing for your side.

3. How to Debate (NOT just ask questions):
   - DO: "I understand your point about X, but I'd argue that Y actually shows the opposite because Z. For example..."
   - DO: "While you mentioned A, I think B is more significant. Consider the case of C..."
   - DO: "That's an interesting perspective, but let me challenge that with D. The evidence suggests E..."
   - DON'T: Just ask "Could you elaborate?" or "Can you give an example?" without presenting your own argument first.

4. Guidance:
   - If the student is stuck, you can briefly acknowledge it, but continue presenting your arguments.
   - Keep the debate moving with your own points and rebuttals.

5. Ending:
   - When you feel the debate has had enough back‑and‑forth (around 4–6 good exchanges),
     ask them for a closing statement, then provide a brief closing for your side.
   - After both closing statements, transition to moderator mode by saying something like:
     "Thank you both for that engaging debate. Now, let me step back as the moderator and provide a summary and feedback."
   - Do NOT start a brand new topic after the closing statement.

Remember: You are DEBATING, not moderating. Take a position and argue for it!

Use the conversation [HISTORY] below to decide what to say next.
[HISTORY]:
{history}
"""

DEBATE_SUMMARY_PROMPT = """
You are now acting as a DEBATE MODERATOR providing a comprehensive summary and feedback after a debate practice session.

Your role has changed from being an opponent to being a neutral, constructive moderator.

You must provide:

1. **Debate Summary** (2-3 sentences):
   - Briefly summarize the main topic/motion that was debated
   - Highlight the key arguments presented by both sides
   - Mention the overall flow and quality of the debate

2. **Winner Declaration** (2-3 sentences):
   - State who you think won the debate (the student or yourself as the opponent)
   - Explain WHY you think they won - be specific about what made their arguments stronger
   - Be fair and constructive in your assessment

3. **Student's Strengths** (2-3 points):
   - Identify 2-3 good points the student made during the debate
   - Explain why these points were effective
   - Be specific and reference their actual arguments

4. **Student's Weaknesses** (2-3 points):
   - Identify 2-3 areas where the student's arguments could have been stronger
   - Explain what was missing or what could have been improved
   - Be constructive, not harsh

5. **Improvement Suggestions** (2-3 points):
   - Suggest specific things the student could have mentioned or countered better
   - Provide concrete examples of stronger arguments they could have used
   - Give actionable advice for future debates

**Tone**: Be encouraging, constructive, and educational. This is practice, so focus on learning and improvement.

**Format**: Present this as a natural, conversational summary (not bullet points). Speak as if you're a moderator addressing the student directly.

**Ending**: After providing all the feedback, conclude by thanking the student for participating and wishing them well in their future debates.

Use the conversation [HISTORY] below to analyze the debate:
[HISTORY]:
{history}
"""


# ====== STATE & ROUTING ======

S = TypeVar("S")


class DebateInterviewState(MessagesState):
    """State for Debate interview."""
    LastNode: Annotated[str, Field(default="")]
    history: Annotated[str, Field(default="")]
    rounds_completed: Annotated[int, Field(default=0, description="Number of meaningful debate exchanges so far")]


class DebateProgress(BaseModel):
    """Decide whether to continue the debate, provide summary, or end it."""
    send_to_which_node: Literal["Debate", "Summary", "End", "Offensive"] = Field(
        description=(
            "Supervise the debate conversation to determine the next step. "
            "Route to 'Debate' if the debate is still in progress and there is room for more arguments or rebuttals. "
            "Route to 'Summary' when both sides have provided closing statements and the debate is ready for moderator feedback. "
            "Route to 'End' ONLY after the moderator has provided the complete summary and feedback. "
            "Route to 'Offensive' ONLY if the interviewee is being offensive, abusive or clearly not taking the debate seriously."
        )
    )


def get_llm(google_api_key: str):
    return ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",
        google_api_key=google_api_key,
        temperature=0.3,
    )


def create_dummy_node() -> Callable:
    def _node(state: S) -> S:
        return state
    return _node


def create_greeting_node(llm) -> Callable:
    def _Node(state: DebateInterviewState) -> DebateInterviewState:
        if state["LastNode"] != "Greeting":
            greeting_prompt = ChatPromptTemplate.from_messages([("system", DEBATE_GREETING_PROMPT)])
            input_messages = greeting_prompt.format_messages() + [{"role": "human", "content": "Start the interview now"}]
            state["messages"] = state["messages"] + input_messages
            state["LastNode"] = "Greeting"
        response = llm.invoke(state["messages"])
        state["messages"] = state["messages"] + [response]
        state["history"] = state["history"] + "\n" + "Interviewer-" + response.content
        state["LastNode"] = "Greeting"
        return state
    return _Node


def create_debate_node(llm) -> Callable:
    def _Node(state: DebateInterviewState) -> DebateInterviewState:
        if state["LastNode"] != "Debate":
            debate_prompt = ChatPromptTemplate.from_messages([("system", DEBATE_MAIN_PROMPT)])
            history_text = state.get("history", "") or ""
            input_messages = debate_prompt.format_messages(history=history_text)
            state["messages"] = input_messages + state["messages"]
            state["LastNode"] = "Debate"
        response = llm.invoke(state["messages"])
        state["messages"] = state["messages"] + [response]
        state["history"] = state["history"] + "\n" + "Interviewer-" + response.content
        state["LastNode"] = "Debate"
        try:
            state["rounds_completed"] = int(state.get("rounds_completed", 0)) + 1
        except Exception:
            state["rounds_completed"] = 1
        return state
    return _Node


def create_summary_node(llm) -> Callable:
    def _Node(state: DebateInterviewState) -> DebateInterviewState:
        if state["LastNode"] != "Summary":
            summary_prompt = ChatPromptTemplate.from_messages([("system", DEBATE_SUMMARY_PROMPT)])
            history_text = state.get("history", "") or ""
            input_messages = summary_prompt.format_messages(history=history_text)
            state["messages"] = input_messages + state["messages"]
            state["LastNode"] = "Summary"
        response = llm.invoke(state["messages"])
        state["messages"] = state["messages"] + [response]
        state["history"] = state["history"] + "\n" + "Interviewer-" + response.content
        state["LastNode"] = "Summary"
        return state
    return _Node


def create_route_to_debate_node(llm) -> Callable:
    def _Node(state: DebateInterviewState) -> Literal["Debate", "Summary", "End", "Offensive"]:
        rounds = int(state.get("rounds_completed", 0) or 0)
        if rounds >= 8:
            return "Summary"
        history = state.get("history", "") or ""
        history_message = HumanMessage(content=history if history else "Debate in progress")
        router = llm.with_structured_output(DebateProgress)
        response = router.invoke([history_message])
        if response is None or not getattr(response, "send_to_which_node", None):
            return "Debate"
        return response.send_to_which_node
    return _Node


def create_end_node() -> Callable:
    def _node(state: DebateInterviewState) -> DebateInterviewState:
        state["LastNode"] = "finished"
        return state
    return _node


def build_debate_graph(google_api_key: str, checkpointer):
    """Build and compile the Debate interview graph."""
    llm = get_llm(google_api_key)
    workflow = StateGraph(DebateInterviewState)

    workflow.add_node("Greeting", create_greeting_node(llm))
    workflow.add_node("Greeting_after", create_dummy_node())
    workflow.add_node("Debate", create_debate_node(llm))
    workflow.add_node("Debate_after", create_dummy_node())
    workflow.add_node("Summary", create_summary_node(llm))
    workflow.add_node("End", create_end_node())
    workflow.add_node("Offensive", create_dummy_node())

    workflow.set_entry_point("Greeting")
    workflow.add_edge("Greeting", "Greeting_after")
    workflow.add_edge("Debate", "Debate_after")
    workflow.add_edge("Summary", "End")
    workflow.add_edge("End", END)
    workflow.add_edge("Offensive", END)

    workflow.add_conditional_edges("Greeting_after", create_route_to_debate_node(llm))
    workflow.add_conditional_edges("Debate_after", create_route_to_debate_node(llm))

    return workflow.compile(checkpointer=checkpointer)
