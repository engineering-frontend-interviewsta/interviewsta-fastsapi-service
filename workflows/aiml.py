"""
LangGraph workflow for AI/ML Engineering interviews.

Follows the same pattern as technical.py:
  AimlInterviewState → StateGraph → nodes → conditional edges → compile(checkpointer)

Interview structure (three phases):
  1. Conceptual  — definitions, comparisons, first-principles explanations
  2. Applied     — design decisions, debugging, practical trade-offs
  3. Deep Dive   — derivations, optimizations, expert-level reasoning

Question sourcing:
  - Curated questions are loaded from workflows/data/aiml_questions.json at session start.
  - The interviewer asks each curated question verbatim, then generates adaptive follow-ups
    based on the quality of the candidate's response (harder if strong, clarifying if weak).
  - This hybrid approach ensures consistent coverage while allowing dynamic difficulty adjustment.
"""

import json
import os
import random
from typing import Annotated, Callable, Dict, List, Literal, TypeVar

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import MessagesState, StateGraph
from pydantic import BaseModel, Field

from .interview_prompt_tone import GREETING_BREVITY
from .utils import get_llm

# ---------------------------------------------------------------------------
# Question bank loader
# ---------------------------------------------------------------------------

_QUESTIONS_PATH = os.path.join(os.path.dirname(__file__), "data", "aiml_questions.json")

# Cached at module load — file is read once per process lifetime
with open(_QUESTIONS_PATH, encoding="utf-8") as _f:
    _QUESTION_BANK: Dict = json.load(_f)

# Maps InterviewTest title (from interview_types.json) to question bank key
TOPIC_KEY_MAP: Dict[str, str] = {
    "Foundational ML": "Foundational ML",
    "Neural Architectures": "Neural Architectures",
    "Transformers & Attention": "Transformers & Attention",
    "LLMs & Modern AI": "LLMs & Modern AI",
    "RAG & Retrieval Systems": "RAG & Retrieval Systems",
    "AI Agents & LangGraph": "AI Agents & LangGraph",
    "MLOps & AI Systems": "MLOps & AI Systems",
    "NLP Fundamentals": "NLP Fundamentals",
}

# Number of questions sampled per tier per session
QUESTIONS_PER_TIER = {
    "conceptual": 2,
    "applied": 2,
    "deep_dive": 1,
}


def _load_questions_for_topic(topic: str, tier: str) -> List[Dict]:
    """
    Return a random sample of curated questions for the given topic and tier.
    Falls back to all available questions if the bank has fewer than requested.
    """
    bank_key = TOPIC_KEY_MAP.get(topic, topic)
    questions = _QUESTION_BANK.get(bank_key, {}).get(tier, [])
    n = QUESTIONS_PER_TIER.get(tier, 3)
    return random.sample(questions, min(n, len(questions)))


def _format_questions_as_research(questions: List[Dict], tier_label: str) -> str:
    """
    Serialise a list of question-bank entries into structured text the LLM
    can parse. Includes the curated follow_up hint so the LLM can use it as
    a starting point for adaptive probing.
    """
    lines = [f"=== {tier_label} Questions ===\n"]
    for i, q in enumerate(questions, 1):
        lines.append(f"[Q{i}] {q['question']}")
        lines.append(f"  Evaluation criteria: {'; '.join(q['evaluation_criteria'])}")
        lines.append(f"  Common mistakes to watch for: {'; '.join(q['common_mistakes'])}")
        lines.append(f"  Suggested follow-up probe: {q['follow_up']}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

S = TypeVar("S", bound="AimlInterviewState")


class AimlInterviewState(MessagesState):
    LastNode: Annotated[str, Field(default="default", description="The last node that was executed")]
    history: Annotated[str, Field(default="", description="Running transcript of the interview so far")]
    TopicResearch: str = ""
    # Set from the InterviewTest title (e.g. "Transformers & Attention") via session payload
    interview_topic: str = ""


# ---------------------------------------------------------------------------
# Routing models (structured LLM output drives conditional edges)
# ---------------------------------------------------------------------------

class AimlGreetingProgress(BaseModel):
    """Route after the greeting exchange."""
    send_to_which_node: Literal["Greeting", "Conceptual_before"] = Field(
        description="'Greeting' to continue the greeting exchange, 'Conceptual_before' once the candidate confirms they are ready to begin"
    )


class AimlConceptualProgress(BaseModel):
    """Route after a conceptual phase exchange."""
    send_to_which_node: Literal["Conceptual", "Applied_before"] = Field(
        description="'Conceptual' to continue asking conceptual questions or follow-ups, 'Applied_before' once all conceptual questions are exhausted"
    )


class AimlAppliedProgress(BaseModel):
    """Route after an applied phase exchange."""
    send_to_which_node: Literal["Applied", "DeepDive_before"] = Field(
        description="'Applied' to continue asking applied questions or follow-ups, 'DeepDive_before' once all applied questions are exhausted"
    )


class AimlDeepDiveProgress(BaseModel):
    """Route after a deep-dive phase exchange."""
    send_to_which_node: Literal["DeepDive", "End"] = Field(
        description="'DeepDive' to continue asking deep-dive questions or follow-ups, 'End' once all deep-dive questions are exhausted"
    )


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

AIML_GREETING_PROMPT = """
You are Glee, an AI/ML engineering interviewer conducting a live technical interview. Embody the persona of a knowledgeable, empathetic human interviewer — professional, encouraging, and precise.

Your goal is to create a warm, focused atmosphere. Introduce yourself, explain the three-part interview structure, and invite the candidate to ask any questions before you begin.

Instructions:

1. Warm Greeting: Begin with a friendly, personal greeting. No stage directions or parenthetical actions.

2. Introduce Yourself: State your name and role (e.g., "My name is Glee, and I'll be your AI/ML interviewer today").

3. State the Topic: Tell the candidate which topic area this interview covers: {topic}.

4. Explain the Format: The interview has three phases:
   - Phase 1 (Conceptual): Core ML/AI theory — definitions, comparisons, and first-principles explanations.
   - Phase 2 (Applied): Practical judgment — design decisions, debugging, and trade-off analysis.
   - Phase 3 (Deep Dive): Expert reasoning — derivations, optimizations, and system-level thinking.

5. Invite Questions: Explicitly ask if the candidate has any questions about the format before you begin. Be welcoming.

6. Listen and Respond: Address any questions concisely, then proceed to Phase 1.
""" + "\n\n" + GREETING_BREVITY

AIML_CONCEPTUAL_PROMPT = """
You are Glee, an AI/ML interviewer conducting Phase 1 (Conceptual) of a technical interview on the topic: {topic}.

Your role is to rigorously assess the candidate's theoretical depth and first-principles understanding using the curated questions below.

[CURATED QUESTIONS]
{research}

[CONVERSATION HISTORY]
{history}

Interview instructions:

1. Question Selection:
   - Work through the curated questions in order.
   - Ask each [Q] verbatim — do not paraphrase or telegraph the topic.
   - Check [CONVERSATION HISTORY] to determine which questions have already been asked.

2. After Each Response — Adaptive Follow-up:
   - Strong answer (covers evaluation criteria well): Generate a harder follow-up that probes edge cases, boundary conditions, or deeper implications. Use the "Suggested follow-up probe" as a starting point but adapt it to what the candidate actually said.
   - Adequate answer (correct intuition, missing precision): Ask a targeted clarifying question to surface the gap (e.g., "You mentioned X — can you formalise that?").
   - Weak answer (incorrect or confused): Ask a simpler sub-question to find the boundary of their knowledge. Do not reveal the answer.

3. Watch for Common Mistakes:
   - Each question lists common mistakes. If the candidate makes one, probe it directly without stating it is a mistake.

4. Transition:
   - Once all curated questions (and their follow-ups) are complete, signal that Phase 1 is done and you are moving to Phase 2.

Tone: Rigorous but encouraging. Never condescending. Acknowledge strong answers explicitly.
"""

AIML_APPLIED_PROMPT = """
You are Glee, an AI/ML interviewer conducting Phase 2 (Applied) of a technical interview on the topic: {topic}.

Your role is to assess the candidate's practical judgment — their ability to make design decisions, debug failures, and reason about trade-offs in real AI/ML systems.

[CURATED QUESTIONS]
{research}

[CONVERSATION HISTORY]
{history}

Interview instructions:

1. Question Selection:
   - Work through the curated applied questions in order.
   - Ask each [Q] verbatim. These are scenario-based: "How would you design...", "What would you do if...", "Compare approaches A and B for...".
   - Check [CONVERSATION HISTORY] to determine which questions have already been asked.

2. After Each Response — Adaptive Follow-up:
   - Strong answer (structured thinking, explicit trade-offs, constraint awareness): Introduce a constraint change to stress-test the design (e.g., "Now assume you have 10x the data but 1/10th the compute budget — does your approach change?").
   - Adequate answer (correct direction, vague on specifics): Ask "What specific metric would you optimise for?" or "What breaks at scale?".
   - Weak answer: Ask a simpler version of the same scenario to find their floor.

3. Probe for depth: Ask about failure modes, monitoring strategies, or how the candidate would validate their design.

4. Transition:
   - Once all applied questions and follow-ups are complete, signal readiness to move to Phase 3 (Deep Dive).

Tone: Collaborative and inquisitive. You are probing judgment, not testing memorisation.
"""

AIML_DEEPDIVE_PROMPT = """
You are Glee, an AI/ML interviewer conducting Phase 3 (Deep Dive) of a technical interview on the topic: {topic}.

This is the expert-level phase — you are assessing the candidate's ability to derive, optimise, and reason at the frontier of AI/ML knowledge.

[CURATED QUESTIONS]
{research}

[CONVERSATION HISTORY]
{history}

Interview instructions:

1. Question Selection:
   - Work through the curated deep-dive questions in order.
   - Ask each [Q] verbatim. These require derivation, optimisation reasoning, or expert-level system thinking.
   - Check [CONVERSATION HISTORY] to determine which questions have already been asked.

2. After Each Response — Adaptive Follow-up:
   - Strong answer (mathematically precise, acknowledges approximations, cites limitations): Ask about recent developments, open research questions, or how the concept differs across model scales.
   - Adequate answer (correct intuition, missing rigour): Probe with "Can you formalise that?" or "What is the complexity bound?".
   - Weak answer: Identify the specific gap and ask a simpler sub-question to find the boundary of their knowledge.

3. Wrap Up:
   - After all deep-dive questions and follow-ups are complete, thank the candidate warmly, give a brief summary of the topics covered, and invite any final questions they may have.

Tone: Intellectually rigorous. Treat the candidate as a peer. Acknowledge genuinely strong answers.
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_history(state: AimlInterviewState) -> str:
    """Reconstruct a readable transcript from the messages list."""
    lines: List[str] = []
    for msg in state.get("messages", []):
        if isinstance(msg, HumanMessage):
            lines.append(f"Candidate: {msg.content}")
        elif isinstance(msg, AIMessage):
            lines.append(f"Interviewer: {msg.content}")
    return "\n".join(lines)


def _fill_prompt(template: str, topic: str, research: str, history: str) -> str:
    return (
        template
        .replace("{topic}", topic)
        .replace("{research}", research)
        .replace("{history}", history)
    )


# ---------------------------------------------------------------------------
# Node factories
# ---------------------------------------------------------------------------

def create_aiml_greeting_node(llm) -> Callable:
    def _node(state: AimlInterviewState) -> Dict:
        topic = state.get("interview_topic", "AI/ML")
        existing_messages = list(state.get("messages", []))

        # On the very first turn, seed the conversation with the system prompt and a
        # synthetic human trigger. Gemini enforces strict human/AI turn alternation —
        # without a HumanMessage the model hangs waiting for one.
        # MessagesState uses an append reducer, so we return only the NEW messages (delta).
        delta: list = []
        if state.get("LastNode") != "Greeting":
            prompt = AIML_GREETING_PROMPT.replace("{topic}", topic)
            delta = [
                SystemMessage(content=prompt),
                HumanMessage(content="Start the interview now."),
            ]
            invoke_messages = existing_messages + delta
        else:
            invoke_messages = existing_messages

        response = llm.invoke(invoke_messages)
        delta.append(response)
        history = _build_history(state) + f"\nInterviewer: {response.content}"
        return {
            "messages": delta,  # only the new messages — reducer appends them
            "LastNode": "Greeting",
            "history": history,
        }
    return _node


def create_aiml_dummy_node() -> Callable:
    """Pass-through node used as a human-in-the-loop interrupt point."""
    def _node(state: AimlInterviewState) -> Dict:
        return state
    return _node


def create_aiml_before_conceptual_node() -> Callable:
    """
    Loads curated conceptual questions from the question bank and formats them
    into TopicResearch. No LLM call — purely deterministic data loading.
    """
    def _node(state: AimlInterviewState) -> Dict:
        topic = state.get("interview_topic", "AI/ML")
        questions = _load_questions_for_topic(topic, "conceptual")
        research = _format_questions_as_research(questions, "Conceptual")
        return {
            "LastNode": "Conceptual_before",
            "TopicResearch": research,
        }
    return _node


def create_aiml_conceptual_node(llm) -> Callable:
    def _node(state: AimlInterviewState) -> Dict:
        topic = state.get("interview_topic", "AI/ML")
        research = state.get("TopicResearch", "")
        history = state.get("history", "")
        prompt = _fill_prompt(AIML_CONCEPTUAL_PROMPT, topic, research, history)
        messages = list(state.get("messages", []))
        # Replace system prompt in slot 0 so conversation history stays intact,
        # then invoke with the full list. Return only the AI response as delta.
        if messages and isinstance(messages[0], SystemMessage):
            messages[0] = SystemMessage(content=prompt)
        else:
            messages = [SystemMessage(content=prompt)] + messages
        response = llm.invoke(messages)
        new_history = history + f"\nInterviewer: {response.content}"
        return {
            "messages": [response],  # delta only — reducer appends
            "LastNode": "Conceptual",
            "history": new_history,
        }
    return _node


def create_aiml_before_applied_node() -> Callable:
    """
    Loads curated applied questions from the question bank and appends them
    to TopicResearch (preserving the conceptual questions already in state).
    """
    def _node(state: AimlInterviewState) -> Dict:
        topic = state.get("interview_topic", "AI/ML")
        questions = _load_questions_for_topic(topic, "applied")
        applied_research = _format_questions_as_research(questions, "Applied")
        # Append to existing research so the full context is available if needed
        combined = state.get("TopicResearch", "") + "\n\n" + applied_research
        return {
            "LastNode": "Applied_before",
            "TopicResearch": combined,
        }
    return _node


def create_aiml_applied_node(llm) -> Callable:
    def _node(state: AimlInterviewState) -> Dict:
        topic = state.get("interview_topic", "AI/ML")
        research = state.get("TopicResearch", "")
        history = state.get("history", "")
        prompt = _fill_prompt(AIML_APPLIED_PROMPT, topic, research, history)
        messages = list(state.get("messages", []))
        if messages and isinstance(messages[0], SystemMessage):
            messages[0] = SystemMessage(content=prompt)
        else:
            messages = [SystemMessage(content=prompt)] + messages
        response = llm.invoke(messages)
        new_history = history + f"\nInterviewer: {response.content}"
        return {
            "messages": [response],  # delta only — reducer appends
            "LastNode": "Applied",
            "history": new_history,
        }
    return _node


def create_aiml_before_deepdive_node() -> Callable:
    """
    Loads curated deep-dive questions from the question bank and appends them
    to TopicResearch.
    """
    def _node(state: AimlInterviewState) -> Dict:
        topic = state.get("interview_topic", "AI/ML")
        questions = _load_questions_for_topic(topic, "deep_dive")
        deepdive_research = _format_questions_as_research(questions, "Deep Dive")
        combined = state.get("TopicResearch", "") + "\n\n" + deepdive_research
        return {
            "LastNode": "DeepDive_before",
            "TopicResearch": combined,
        }
    return _node


def create_aiml_deepdive_node(llm) -> Callable:
    def _node(state: AimlInterviewState) -> Dict:
        topic = state.get("interview_topic", "AI/ML")
        research = state.get("TopicResearch", "")
        history = state.get("history", "")
        prompt = _fill_prompt(AIML_DEEPDIVE_PROMPT, topic, research, history)
        messages = list(state.get("messages", []))
        if messages and isinstance(messages[0], SystemMessage):
            messages[0] = SystemMessage(content=prompt)
        else:
            messages = [SystemMessage(content=prompt)] + messages
        response = llm.invoke(messages)
        new_history = history + f"\nInterviewer: {response.content}"
        return {
            "messages": [response],  # delta only — reducer appends
            "LastNode": "DeepDive",
            "history": new_history,
        }
    return _node


def create_aiml_end_node() -> Callable:
    def _node(state: AimlInterviewState) -> Dict:
        return {"LastNode": "finished"}
    return _node


# ---------------------------------------------------------------------------
# Routing factories (structured LLM output → conditional edge targets)
# ---------------------------------------------------------------------------

def create_route_greeting(routing_llm) -> Callable:
    def _route(state: AimlInterviewState) -> Literal["Greeting", "Conceptual_before"]:
        response = routing_llm.invoke(state["history"])
        return response.send_to_which_node
    return _route


def create_route_conceptual(routing_llm) -> Callable:
    def _route(state: AimlInterviewState) -> Literal["Conceptual", "Applied_before"]:
        response = routing_llm.invoke(state["history"])
        return response.send_to_which_node
    return _route


def create_route_applied(routing_llm) -> Callable:
    def _route(state: AimlInterviewState) -> Literal["Applied", "DeepDive_before"]:
        response = routing_llm.invoke(state["history"])
        return response.send_to_which_node
    return _route


def create_route_deepdive(routing_llm) -> Callable:
    def _route(state: AimlInterviewState) -> Literal["DeepDive", "End"]:
        response = routing_llm.invoke(state["history"])
        return response.send_to_which_node
    return _route


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def get_aiml_graph(google_api_key: str, tavily_api_key: str, checkpointer):
    """
    Build and compile the AI/ML interview LangGraph workflow.

    Graph structure:
        Greeting → Greeting_after
            ↓ (conditional: loop or advance)
        Conceptual_before → Conceptual → Conceptual_after
            ↓ (conditional: loop or advance)
        Applied_before → Applied → Applied_after
            ↓ (conditional: loop or advance)
        DeepDive_before → DeepDive → DeepDive_after
            ↓ (conditional: loop or conclude)
        End → __end__

    The *_before nodes are deterministic (no LLM call) — they load questions
    from the static question bank. The interview nodes (Conceptual, Applied,
    DeepDive) hold the LLM that asks curated questions and generates adaptive
    follow-ups based on response quality.
    """
    llm = get_llm(google_api_key=google_api_key)

    workflow = StateGraph(AimlInterviewState)

    # --- Nodes ---
    workflow.add_node("Greeting", create_aiml_greeting_node(llm))
    workflow.add_node("Greeting_after", create_aiml_dummy_node())

    # _before nodes are pure data-loading — no LLM, no API call
    workflow.add_node("Conceptual_before", create_aiml_before_conceptual_node())
    workflow.add_node("Conceptual", create_aiml_conceptual_node(llm))
    workflow.add_node("Conceptual_after", create_aiml_dummy_node())

    workflow.add_node("Applied_before", create_aiml_before_applied_node())
    workflow.add_node("Applied", create_aiml_applied_node(llm))
    workflow.add_node("Applied_after", create_aiml_dummy_node())

    workflow.add_node("DeepDive_before", create_aiml_before_deepdive_node())
    workflow.add_node("DeepDive", create_aiml_deepdive_node(llm))
    workflow.add_node("DeepDive_after", create_aiml_dummy_node())

    workflow.add_node("End", create_aiml_end_node())

    # --- Entry point ---
    workflow.set_entry_point("Greeting")

    # --- Linear edges (within each phase) ---
    workflow.add_edge("Greeting", "Greeting_after")
    workflow.add_edge("Conceptual_before", "Conceptual")
    workflow.add_edge("Conceptual", "Conceptual_after")
    workflow.add_edge("Applied_before", "Applied")
    workflow.add_edge("Applied", "Applied_after")
    workflow.add_edge("DeepDive_before", "DeepDive")
    workflow.add_edge("DeepDive", "DeepDive_after")
    workflow.add_edge("End", "__end__")

    # --- Conditional edges (LLM-driven routing) ---
    workflow.add_conditional_edges(
        "Greeting_after",
        create_route_greeting(llm.with_structured_output(AimlGreetingProgress)),
    )
    workflow.add_conditional_edges(
        "Conceptual_after",
        create_route_conceptual(llm.with_structured_output(AimlConceptualProgress)),
    )
    workflow.add_conditional_edges(
        "Applied_after",
        create_route_applied(llm.with_structured_output(AimlAppliedProgress)),
    )
    workflow.add_conditional_edges(
        "DeepDive_after",
        create_route_deepdive(llm.with_structured_output(AimlDeepDiveProgress)),
    )

    return workflow.compile(checkpointer=checkpointer)
