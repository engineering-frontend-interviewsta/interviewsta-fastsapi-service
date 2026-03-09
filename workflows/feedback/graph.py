# graph.py
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from typing import Any, Dict
from .schemas import FeedbackGraphState, StrengthsAndImprovements, InteractionFeedbackItem
from .prompts import SLEEVE_SCORING_SYSTEM, STRENGTHS_SYSTEM, INTERACTION_FEEDBACK_SYSTEM


# ── Node factories ────────────────────────────────────────────────────────────

def make_sleeve_node(sleeve_name: str, sleeve_model: type, llm):
    structured_llm = llm.with_structured_output(sleeve_model)

    # Build metric list string once at node creation time, not inside _node
    metric_names = list(sleeve_model.model_fields.keys())
    metric_list_str = "\n".join(f"  - {m.replace('_', ' ')}" for m in metric_names)

    def _node(state: FeedbackGraphState) -> FeedbackGraphState:
        prompt = SLEEVE_SCORING_SYSTEM.format(
            interview_title=state["interview_title"],
            sleeve_name=sleeve_name,
            metric_list=metric_list_str,   # ← key must match {metric_list} in template
            history_log=state["history_log"],
        )
        response = structured_llm.invoke([HumanMessage(content=prompt)])
        state["sleeve_scores"][sleeve_name] = response
        return state

    _node.__name__ = f"sleeve__{sleeve_name[:20].replace(' ', '_')}"
    return _node

def make_strengths_node(llm):
    structured_llm = llm.with_structured_output(StrengthsAndImprovements)

    def _node(state: FeedbackGraphState) -> FeedbackGraphState:
        prompt = STRENGTHS_SYSTEM.format(
            interview_title=state["interview_title"],
            history_log=state["history_log"],
        )
        state["strengths_and_improvements"] = structured_llm.invoke(
            [HumanMessage(content=prompt)]
        )
        return state

    return _node


def make_interaction_feedback_node(llm):
    from pydantic import BaseModel
    from typing import List

    class FeedbackList(BaseModel):
        items: List[InteractionFeedbackItem]

    structured_llm = llm.with_structured_output(FeedbackList)

    def _node(state: FeedbackGraphState) -> FeedbackGraphState:
        prompt = INTERACTION_FEEDBACK_SYSTEM.format(
            interview_title=state["interview_title"],
            history_log=state["history_log"],
        )
        result = structured_llm.invoke([HumanMessage(content=prompt)])
        state["interaction_feedback"] = result.items
        return state

    return _node


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_feedback_graph(sleeve_models: Dict[str, type], llm):
    """
    Builds a linear feedback graph for any interview type.

    sleeve_models: { sleeve_name -> Pydantic model class }
    Graph topology: sleeve_1 → sleeve_2 → ... → strengths → interaction_feedback → END
    """
    graph = StateGraph(FeedbackGraphState)
    sleeve_names = list(sleeve_models.keys())
    node_names = []

    # Add one node per sleeve
    for sleeve_name, sleeve_model in sleeve_models.items():
        node_id = f"sleeve__{sleeve_name}"
        graph.add_node(node_id, make_sleeve_node(sleeve_name, sleeve_model, llm))
        node_names.append(node_id)

    # Fixed downstream nodes
    graph.add_node("strengths", make_strengths_node(llm))
    graph.add_node("interaction_feedback", make_interaction_feedback_node(llm))
    node_names += ["strengths", "interaction_feedback"]

    # Wire linearly: sleeve_1 → sleeve_2 → ... → strengths → interaction_feedback → END
    graph.set_entry_point(node_names[0])
    for i in range(len(node_names) - 1):
        graph.add_edge(node_names[i], node_names[i + 1])
    graph.add_edge("interaction_feedback", END)

    return graph.compile()
