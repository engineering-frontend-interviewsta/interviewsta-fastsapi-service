"""
phase_engine.py — plug-and-play interview phase engine.

Human-in-the-loop pattern
--------------------------
interrupt_before is set on every *_after node.

Flow per turn:
  {phase}_before  → {phase} (LLM fires) → {phase}_after  ← PAUSE HERE
  caller shows response, user types reply
  agent.update_state(messages=[HumanMessage(...)]) injects reply
  agent.invoke(None) resumes — {phase}_after router now sees human reply
  router → {phase}_before (self-loop) or {next}_before or End → PAUSE again
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import (
    Annotated, Any, Callable, Dict, List, Literal, Optional,
    Type, TypeVar,
)

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, MessagesState, StateGraph
from pydantic import BaseModel, Field, create_model


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

class BaseInterviewState(MessagesState):
    LastNode:           Annotated[str,                  Field(default="")]
    history:            Annotated[str,                  Field(default="")]
    background:         Annotated[Dict[str, Any],       Field(default_factory=dict)]
    phase_questions:    Annotated[Dict[str, List[Any]], Field(default_factory=dict)]
    phase_question_idx: Annotated[Dict[str, int],       Field(default_factory=dict)]
    phase_state:        Annotated[Dict[str, Any],       Field(default_factory=dict)]


S = TypeVar("S")


def _ensure_defaults(state: Dict[str, Any]) -> None:
    """Populate keys that LangGraph doesn't auto-fill from Field defaults."""
    state.setdefault("history", "")
    state.setdefault("LastNode", "")
    state.setdefault("background", {})
    state.setdefault("phase_questions", {})
    state.setdefault("phase_question_idx", {})
    state.setdefault("phase_state", {})
    if not state.get("messages"):
        state["messages"] = []
    if state["phase_state"] is None:
        state["phase_state"] = {}


# ---------------------------------------------------------------------------
# PhaseConfig dataclass
# ---------------------------------------------------------------------------

@dataclass
class PhaseConfig:
    phase_name: str
    order: int
    prompt: str
    prompt_inputs: List[str] = field(default_factory=list)
    number_of_questions_to_ask: int = 0
    setup_questions: bool = False
    setup_questions_prompt: str = ""
    question_filters: Dict[str, Any] = field(default_factory=dict)
    route_nodes: List[str] = field(default_factory=list)
    route_ahead_prompt: str = ""
    immediate_feedback_required: bool = False
    feedback_prompt: str = ""
    mcp_tools: bool = False
    tool_names: List[str] = field(default_factory=list)
    special_output_format: Optional[str] = None
    entity_schema: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Dynamic model builders
# ---------------------------------------------------------------------------

_PYTHON_TYPE_MAP: Dict[str, type] = {
    "str": str, "int": int, "float": float, "bool": bool, "list": List[str],
}


def build_pydantic_model(model_name: str, schema: Dict[str, Any]) -> Type[BaseModel]:
    field_definitions: Dict[str, Any] = {}
    for f in schema.get("fields", []):
        python_type = _PYTHON_TYPE_MAP.get(f.get("type", "str"), str)
        desc = f.get("description", "")
        if f.get("optional"):
            python_type = Optional[python_type]
            field_definitions[f["name"]] = (python_type, Field(default=None, description=desc))
        else:
            field_definitions[f["name"]] = (python_type, Field(..., description=desc))
    return create_model(model_name, **field_definitions)


def build_routing_model(
    phase_name: str,
    route_nodes: List[str],
    route_ahead_prompt: str,
) -> Type[BaseModel]:
    literal_type = Literal[tuple(route_nodes)]  # type: ignore[valid-type]
    return create_model(
        f"{phase_name}Routing",
        send_to_which_node=(
            literal_type,
            Field(..., description=route_ahead_prompt or "Route to the next node."),
        ),
    )


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------

def render_prompt(template: str, phase: PhaseConfig, state: Dict[str, Any]) -> str:
    ctx: Dict[str, Any] = {}

    # Expose individual background fields (company, subject, role, resume …)
    bg = state.get("background", {})
    if isinstance(bg, dict):
        ctx.update(bg)
        ctx.setdefault("resume", "No resume provided")

    if "background" in phase.prompt_inputs:
        ctx["background"] = json.dumps(bg, indent=2) if isinstance(bg, dict) else str(bg)

    if "questions" in phase.prompt_inputs:
        all_qs  = state.get("phase_questions", {}).get(phase.phase_name, [])
        idx     = state.get("phase_question_idx", {}).get(phase.phase_name, 0)
        total   = len(all_qs)

        # {questions}     → the single current question (index-sliced)
        # {all_questions} → the full list (for phases that want everything at once)
        current_q = all_qs[idx] if idx < total else (all_qs[-1] if all_qs else {})
        ctx["questions"]     = json.dumps(current_q, indent=2)
        ctx["all_questions"] = json.dumps(all_qs, indent=2)
        ctx["current_question_number"] = idx + 1
        ctx["total_questions"]         = total

        print(f"[render_prompt] {phase.phase_name}  idx={idx}/{total}  "
              f"question={json.dumps(current_q)[:60]}…")

    if "history" in phase.prompt_inputs:
        ctx["history"] = state.get("history", "")

    # Merge phase-specific state (e.g. case_question, case_reference for CaseStudy)
    extra = state.get("phase_state", {}).get(phase.phase_name, {})
    if isinstance(extra, dict):
        ctx.update(extra)

    try:
        return template.format(**ctx)
    except KeyError as exc:
        print(f"[render_prompt] WARNING: missing {exc} in '{phase.phase_name}' — using template as-is")
        return template


# ---------------------------------------------------------------------------
# Node factories
# ---------------------------------------------------------------------------

def make_dummy_node() -> Callable:
    def _node(state: S) -> S:
        return state
    return _node


def make_phase_node(phase: PhaseConfig, llm, deferred_setup: bool = False) -> Callable:
    """
    deferred_setup=True means this phase has setup_questions=True but its
    setup_questions_prompt depends on another phase's questions (e.g. Theoretical
    depends on Coding questions). Questions are generated on first entry here
    instead of at graph start, guaranteeing the dependency is already in state.
    """
    if phase.special_output_format == "json" and phase.entity_schema:
        entity_model = build_pydantic_model(f"{phase.phase_name}Entity", phase.entity_schema)
        invoke_llm   = llm.with_structured_output(entity_model)
        structured   = True
    else:
        invoke_llm = llm
        structured = False

    def _node(state: Dict[str, Any]) -> Dict[str, Any]:
        _ensure_defaults(state)
        name     = phase.phase_name
        messages = list(state.get("messages", []))
        n_human  = sum(1 for m in messages if isinstance(m, HumanMessage))
        print(f"[Phase:{name}] enter  LastNode={state.get('LastNode')!r}  n_messages={len(messages)}  n_human={n_human}")

        # Use phase_state._meta to track first entry (LastNode unreliable on resume)
        ps         = dict(state.get("phase_state", {}))
        phase_meta = ps.get(f"_meta_{name}", {})
        already_injected = phase_meta.get("prompt_injected", False)

        # --- Deferred question generation (runs once on first entry) ---
        if deferred_setup and not phase_meta.get("questions_generated", False):
            pq = dict(state.get("phase_questions", {}))
            if name not in pq:
                background       = state.get("background", {})
                coding_questions = pq.get("Coding", [])
                coding_str       = (
                    json.dumps(coding_questions, indent=2)
                    if coding_questions
                    else json.dumps(background, indent=2)
                )
                print(f"[Phase:{name}] generating deferred questions from "
                      f"{len(coding_questions)} coding question(s)")
                try:
                    gen_prompt = phase.setup_questions_prompt.format(
                        background=json.dumps(background, indent=2),
                        filters=json.dumps(phase.question_filters, indent=2),
                        coding_questions=coding_str,
                    )
                    response  = llm.invoke(gen_prompt)
                    content   = response.content if hasattr(response, "content") else str(response)
                    try:
                        questions = json.loads(content)
                    except json.JSONDecodeError:
                        questions = [{"question": content}]
                    pq[name] = questions
                    print(f"[Phase:{name}] generated {len(questions)} theoretical question(s)")
                except Exception as exc:
                    print(f"[Phase:{name}] question generation failed: {exc}")
                    pq[name] = []
                state["phase_questions"] = pq

            phase_meta["questions_generated"] = True
            ps[f"_meta_{name}"] = phase_meta
            state["phase_state"] = ps

        # --- System prompt injection ---
        system_content = render_prompt(phase.prompt, phase, state)
        new_system     = SystemMessage(content=system_content)

        if not already_injected:
            if messages and isinstance(messages[0], SystemMessage):
                messages = [new_system] + messages[1:]
            else:
                messages = [new_system] + messages
            phase_meta["prompt_injected"] = True
            ps[f"_meta_{name}"] = phase_meta
            state["phase_state"] = ps
            print(f"[Phase:{name}] injected system prompt  n_messages={len(messages)}")
        else:
            if messages and isinstance(messages[0], SystemMessage):
                messages = [new_system] + messages[1:]
            print(f"[Phase:{name}] refreshed system prompt  n_messages={len(messages)}")

        state["messages"] = messages
        response = invoke_llm.invoke(state["messages"])

        if structured:
            ps2 = dict(state.get("phase_state", {}))
            ps2[name] = response.model_dump() if hasattr(response, "model_dump") else response
            state["phase_state"] = ps2
            response_str = _entity_to_str(response)
            state["messages"] = state["messages"] + [AIMessage(content=response_str)]
            state["history"]  = state["history"]  + f"\nInterviewer- {response_str}"
            print(f"[Phase:{name}] structured  preview={response_str[:80]!r}")
        else:
            content = response.content if hasattr(response, "content") else str(response)
            state["messages"] = state["messages"] + [response]
            state["history"]  = state["history"]  + f"\nInterviewer- {content}"
            print(f"[Phase:{name}] responded  preview={content[:80]!r}")

        state["LastNode"] = name
        return state

    return _node


def _entity_to_str(entity: Any) -> str:
    if hasattr(entity, "model_dump"):
        d = entity.model_dump()
    elif isinstance(entity, dict):
        d = entity
    else:
        return str(entity)
    parts = []
    for k, v in d.items():
        if v is None:
            continue
        parts.append("\n".join(f"  {i+1}) {x}" for i, x in enumerate(v)) if isinstance(v, list) else str(v))
    return "\n\n".join(parts)


def make_routing_node(phase: PhaseConfig, llm) -> Callable:
    routing_model = build_routing_model(
        phase.phase_name, phase.route_nodes, phase.route_ahead_prompt
    )
    routing_llm = llm.with_structured_output(routing_model)
    fallback    = phase.route_nodes[0] if phase.route_nodes else f"{phase.phase_name}_before"
    self_loop   = f"{phase.phase_name}_before"   # normalised self-reference

    def _route(state: Dict[str, Any]) -> str:
        _ensure_defaults(state)
        name     = phase.phase_name
        history  = state.get("history", "") or ""
        messages = state.get("messages", [])
        n_human  = sum(1 for m in messages if isinstance(m, HumanMessage))

        # Current index info for logging
        idx   = state.get("phase_question_idx", {}).get(name, 0)
        total = len(state.get("phase_questions", {}).get(name, []))
        print(f"[Router:{name}] enter  n_human={n_human}  "
              f"q_idx={idx}/{total}  history_tail={history[-120:]!r}")

        response = routing_llm.invoke(history)
        if response is None:
            print(f"[Router:{name}] None → {fallback}")
            return fallback

        decision = response.send_to_which_node
        print(f"[Router:{name}] → {decision}")

        # If staying in the same phase (self-loop), advance to the next question.
        # This means the current question is done and the LLM will receive the
        # next one on the following phase node invocation.
        if decision == self_loop and total > 0:
            pqi  = dict(state.get("phase_question_idx", {}))
            next_idx = idx + 1
            if next_idx < total:
                pqi[name] = next_idx
                state["phase_question_idx"] = pqi
                print(f"[Router:{name}] ✅ advanced question index {idx} → {next_idx}")
            else:
                print(f"[Router:{name}] ⚠️  already at last question (idx={idx}/{total-1}), not advancing")

        return decision

    return _route


def make_feedback_node(phase: PhaseConfig, llm) -> Callable:
    def _node(state: Dict[str, Any]) -> Dict[str, Any]:
        _ensure_defaults(state)
        name = phase.phase_name
        last_human = ""
        for msg in reversed(state.get("messages", [])):
            if isinstance(msg, HumanMessage):
                last_human = msg.content
                break
        entity = state.get("phase_state", {}).get(name, {})
        ctx = {"answer": last_human, **({} if not isinstance(entity, dict) else entity)}
        try:
            prompt_content = phase.feedback_prompt.format(**ctx)
        except KeyError:
            prompt_content = phase.feedback_prompt
        fb = llm.invoke(prompt_content)
        content = fb.content if hasattr(fb, "content") else str(fb)
        state["messages"] = state["messages"] + [AIMessage(content=content)]
        state["history"]  = state["history"]  + f"\nInterviewer- {content}"
        state["LastNode"] = f"{name}_feedback"
        print(f"[Feedback:{name}] preview={content[:80]!r}")
        return state
    return _node


def make_end_node() -> Callable:
    def _node(state: Dict[str, Any]) -> Dict[str, Any]:
        _ensure_defaults(state)
        state["LastNode"] = "finished"
        print("[End] interview finished")
        return state
    return _node


def make_offensive_node(llm) -> Callable:
    def _node(state: Dict[str, Any]) -> Dict[str, Any]:
        _ensure_defaults(state)
        history = state.get("history", "")
        msg = llm.invoke(
            f"Generate a polite but firm message ending the interview because the "
            f"candidate has been offensive or unserious.\n\nHistory:\n{history}"
        )
        content = msg.content if hasattr(msg, "content") else str(msg)
        state["messages"] = state["messages"] + [AIMessage(content=content)]
        state["LastNode"] = "Offensive"
        return state
    return _node


# ---------------------------------------------------------------------------
# Setup questions node
# ---------------------------------------------------------------------------

def make_setup_questions_node(phases_needing_setup: List[PhaseConfig], llm) -> Callable:
    """
    Generates questions for phases that have setup_questions=True.

    Runs once at graph start ONLY for phases whose questions don't depend on
    other phases' questions being available first.

    For Theoretical questions (which must be derived from Coding questions),
    generation is deferred to the first entry of the Theoretical phase node
    via make_phase_node's lazy-setup logic, not here.
    """
    # Separate phases: those that can be set up immediately vs those that
    # depend on other phase questions (identified by {coding_questions} in prompt)
    immediate_phases = [
        p for p in phases_needing_setup
        if "{coding_questions}" not in p.setup_questions_prompt
    ]
    deferred_phases = [
        p for p in phases_needing_setup
        if "{coding_questions}" in p.setup_questions_prompt
    ]

    if deferred_phases:
        phase_names = [p.phase_name for p in deferred_phases]
        print(f"[setup_questions] Deferring setup for {phase_names} "
              f"(depend on coding questions — will generate on first phase entry)")

    def _node(state: Dict[str, Any]) -> Dict[str, Any]:
        _ensure_defaults(state)
        print(f"[setup_questions] enter  existing={list(state.get('phase_questions', {}).keys())}")
        pq = dict(state.get("phase_questions", {}))

        for phase in immediate_phases:
            if phase.phase_name in pq:
                continue
            background = state.get("background", {})
            try:
                prompt = phase.setup_questions_prompt.format(
                    background=json.dumps(background, indent=2),
                    filters=json.dumps(phase.question_filters, indent=2),
                )
            except KeyError as exc:
                print(f"[setup_questions] WARNING: {exc} — skipping {phase.phase_name}")
                continue
            try:
                response  = llm.invoke(prompt)
                content   = response.content if hasattr(response, "content") else str(response)
                try:
                    questions = json.loads(content)
                except json.JSONDecodeError:
                    questions = [{"question": content}]
                pq[phase.phase_name] = questions
                print(f"[Setup] Generated {len(questions)} questions for {phase.phase_name}")
            except Exception as exc:
                print(f"[Setup] Failed for {phase.phase_name}: {exc}")
                pq[phase.phase_name] = []

        state["phase_questions"] = pq
        return state

    return _node, deferred_phases  # return deferred list so build_graph can wire it


# ---------------------------------------------------------------------------
# Greeting node + router
# ---------------------------------------------------------------------------

# Per-interview-type format hints, matched against background["name"].
_INTERVIEW_FORMAT_HINTS: List[tuple] = [
    ("debate",
     "a structured DEBATE PRACTICE session. Present a debate motion on a "
     "tech/AI/business topic, ask the candidate to choose a side (for or against), "
     "then engage in 3-4 rounds of argumentation before providing a summary and feedback."),
    ("case",
     "a CASE STUDY interview. You will present a real business scenario and guide "
     "the candidate through structured problem-solving and discussion."),
    ("communication",
     "a COMMUNICATION SKILLS assessment covering a speaking exercise, a writing "
     "comprehension task, and vocabulary MCQ questions."),
    ("coding",
     "a TECHNICAL CODING interview starting with a personalised conversation, "
     "then conceptual questions, and finally live coding problems."),
    ("frontend",  "a FRONTEND DEVELOPMENT technical interview."),
    ("backend",   "a BACKEND DEVELOPMENT technical interview."),
    ("ui/ux",     "a UI/UX DESIGN interview focused on design thinking and process."),
    ("ai/ml",     "an AI/ML technical interview."),
    ("data",      "a DATA SCIENCE technical interview."),
    ("role",
     "a ROLE-BASED technical interview covering personalisation, technical "
     "questions, a coding challenge, and a project discussion."),
]

_DEFAULT_FORMAT_HINT = "a live interview session."


def _resolve_format_hint(background: dict) -> str:
    name = (background.get("name") or "").lower()
    for keyword, hint in _INTERVIEW_FORMAT_HINTS:
        if keyword in name:
            return hint
    return _DEFAULT_FORMAT_HINT


GREETING_PROMPT = """Your name is Glee. You are conducting {format_hint}

{context_lines}

Instructions:
- Greet the candidate warmly and introduce yourself by name.
- Clearly explain the format of this specific interview type so the candidate knows what to expect.
- Invite any questions they may have about the process before you begin.
- Respond in plain, conversational prose — no markdown, no bullet points.
"""


def make_greeting_node(llm) -> Callable:
    def _node(state: Dict[str, Any]) -> Dict[str, Any]:
        _ensure_defaults(state)
        messages  = list(state.get("messages", []))
        last_node = state.get("LastNode", "")
        print(f"[Greeting] enter  LastNode={last_node!r}  n_messages={len(messages)}")

        if last_node != "Greeting":
            bg = state.get("background", {})

            # Allow fully custom greeting via background["greeting_prompt"]
            custom = bg.get("greeting_prompt") if isinstance(bg, dict) else None
            if custom:
                prompt_content = custom
                print("[Greeting] using custom greeting_prompt from background")
            else:
                format_hint = _resolve_format_hint(bg)

                # Build a concise context block (skip empty/placeholder values)
                skip_keys = {"name", "tags", "greeting_prompt"}
                skip_vals = {None, "", "-", "No resume provided"}
                parts = [
                    f"- {k.replace('_', ' ').capitalize()}: {v}"
                    for k, v in bg.items()
                    if k not in skip_keys and v not in skip_vals
                ]
                context_lines = (
                    "Additional session context:\n" + "\n".join(parts)
                    if parts else ""
                )

                prompt_content = GREETING_PROMPT.format(
                    format_hint=format_hint,
                    context_lines=context_lines,
                )
                print(f"[Greeting] format_hint={format_hint[:70]!r}")

            system  = SystemMessage(content=prompt_content)
            kickoff = HumanMessage(content="Start the interview now.")
            if messages and isinstance(messages[0], SystemMessage):
                messages = [system] + messages[1:] + [kickoff]
            else:
                messages = [system, kickoff]
            print(f"[Greeting] injected  n_messages={len(messages)}")

        state["messages"] = messages
        response = llm.invoke(state["messages"])
        content  = response.content if hasattr(response, "content") else str(response)
        state["messages"] = state["messages"] + [response]
        state["history"]  = state["history"]  + f"\nInterviewer- {content}"
        state["LastNode"] = "Greeting"
        print(f"[Greeting] responded  preview={content[:80]!r}")
        return state
    return _node


def make_greeting_router(first_phase_before: str, llm) -> Callable:
    """
    Runs from Greeting_after AFTER the human has replied.
    (interrupt_before on Greeting_after guarantees the human message is in state.)
    Hidden kickoff 'Start the interview now.' = 1 human msg always present.
    Real reply = n_human >= 2.
    """
    ROUTING_PROMPT = (
        "You are supervising an interview greeting phase.\n"
        "Conversation history:\n{history}\n\n"
        "The candidate has just replied to the interviewer's greeting.\n"
        "Decide the next step:\n"
        "- Route to 'Greeting' ONLY if the candidate asked a specific question "
        "about the interview process that still needs answering.\n"
        f"- Route to '{first_phase_before}' if the candidate acknowledged the "
        "greeting, said they have no questions, or is ready to begin "
        "(even short replies like 'hi', 'ok', 'ready', 'sure', 'let's go').\n"
        "- Route to 'Offensive' if rude or unserious.\n"
        f"Default: route to '{first_phase_before}'.\n"
    )
    GreetingRouting = build_routing_model(
        "Greeting",
        ["Greeting", first_phase_before, "Offensive"],
        ROUTING_PROMPT,
    )
    routing_llm = llm.with_structured_output(GreetingRouting)

    def _route(state: Dict[str, Any]) -> str:
        _ensure_defaults(state)
        messages = state.get("messages", [])
        history  = state.get("history", "") or ""
        n_human  = sum(1 for m in messages if isinstance(m, HumanMessage))
        print(f"[Router:Greeting] n_human={n_human}  history_tail={history[-120:]!r}")
        if n_human < 2:
            # No real reply yet — advance anyway; phase router will pause correctly
            print(f"[Router:Greeting] no real reply → {first_phase_before}")
            return first_phase_before
        response = routing_llm.invoke(ROUTING_PROMPT.format(history=history))
        decision = response.send_to_which_node if response else first_phase_before
        print(f"[Router:Greeting] → {decision}")
        return decision

    return _route


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def get_llm(google_api_key: str):
    return ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",
        google_api_key=google_api_key,
        temperature=0.3,
    )


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph(
    phases: List[PhaseConfig],
    state_class,
    google_api_key: str,
    checkpointer,
):
    """
    Compiles with interrupt_before on every *_after node.

    Turn cycle:
      {phase}_before (pass-through)
        → {phase} (LLM fires, Glee speaks)
          → {phase}_after  ← PAUSE (interrupt_before)
    caller: show AI response → collect human reply →
      agent.update_state(config=THREAD, values={"messages": [HumanMessage(...)]})
      agent.invoke(None, config=THREAD)  ← resumes at {phase}_after
        → router sees human reply → routes to next _before or End → PAUSE again
    """
    llm = get_llm(google_api_key)
    workflow = StateGraph(state_class)

    # Every *_after node is an interrupt point
    interrupt_nodes: List[str] = ["Greeting_after"]

    # Fixed nodes
    workflow.add_node("End",            make_end_node())
    workflow.add_node("Offensive",      make_offensive_node(llm))
    workflow.add_node("Greeting",       make_greeting_node(llm))
    workflow.add_node("Greeting_after", make_dummy_node())

    # Optional setup — returns (node_fn, deferred_phases_list)
    phases_with_setup = [p for p in phases if p.setup_questions]
    deferred_phase_names: set = set()
    if phases_with_setup:
        setup_node_fn, deferred_phases = make_setup_questions_node(phases_with_setup, llm)
        deferred_phase_names = {p.phase_name for p in deferred_phases}
        workflow.add_node("setup_questions", setup_node_fn)
        workflow.set_entry_point("setup_questions")
        workflow.add_edge("setup_questions", "Greeting")
    else:
        workflow.set_entry_point("Greeting")

    workflow.add_edge("Greeting", "Greeting_after")

    first_phase_before = f"{phases[0].phase_name}_before" if phases else "End"
    workflow.add_conditional_edges(
        "Greeting_after",
        make_greeting_router(first_phase_before, llm),
        {dest: dest for dest in ["Greeting", first_phase_before, "Offensive"]},
    )

    # Dynamic phases
    for i, phase in enumerate(phases):
        name        = phase.phase_name
        next_phase  = phases[i + 1] if i + 1 < len(phases) else None
        next_before = f"{next_phase.phase_name}_before" if next_phase else "End"

        # Normalise self-references: "Rapport" → "Rapport_before"
        if not phase.route_nodes:
            phase.route_nodes = [f"{name}_before", next_before, "Offensive"]
        else:
            phase.route_nodes = [
                f"{r}_before" if r == name else r
                for r in phase.route_nodes
            ]

        # Pass deferred_setup=True for phases whose questions are generated
        # on first entry (after their dependencies are already in state)
        is_deferred = name in deferred_phase_names

        workflow.add_node(f"{name}_before", make_dummy_node())
        workflow.add_node(name,             make_phase_node(phase, llm, deferred_setup=is_deferred))
        workflow.add_node(f"{name}_after",  make_dummy_node())
        interrupt_nodes.append(f"{name}_after")

        if phase.immediate_feedback_required:
            workflow.add_node(f"{name}_feedback", make_feedback_node(phase, llm))
            workflow.add_edge(f"{name}_before",   name)
            workflow.add_edge(name,               f"{name}_feedback")
            workflow.add_edge(f"{name}_feedback", f"{name}_after")
        else:
            workflow.add_edge(f"{name}_before", name)
            workflow.add_edge(name,             f"{name}_after")

        # Single-destination phases (e.g. Summary → End) use a direct edge —
        # no LLM router needed and no interrupt (fires once and advances automatically).
        if len(phase.route_nodes) == 1:
            sole_dest = phase.route_nodes[0]
            workflow.add_edge(f"{name}_after", sole_dest)
            # Remove this _after from interrupt_nodes since we don't need to pause here
            interrupt_nodes.remove(f"{name}_after")
            print(f"[build_graph] {name}_after → {sole_dest} (direct edge, no interrupt)")
        else:
            workflow.add_conditional_edges(
                f"{name}_after",
                make_routing_node(phase, llm),
                {dest: dest for dest in phase.route_nodes},
            )

    workflow.add_edge("End",       END)
    workflow.add_edge("Offensive", END)

    agent = workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=interrupt_nodes,
    )
    print(f"Graph compiled  (interrupt_before={interrupt_nodes})")
    return agent, interrupt_nodes