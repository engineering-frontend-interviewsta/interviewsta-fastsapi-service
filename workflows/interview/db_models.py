"""
db_models.py — SQLAlchemy models + build_graph_for_interview()

Question fetching resolution order (per phase)
------------------------------------------------
1. question_filters = {"use_db_questions": true}
       → fetch from DB using interview.subject + interview.difficulty
2. question_filters = {"subject": "...", "difficulty": "..."}
       → fetch from DB using explicit override values in the filter
3. setup_questions = True
       → LLM generates questions at runtime; nothing to pre-fetch
4. number_of_questions_to_ask > 0  (no other flag set)
       → fall back to interview.subject + interview.difficulty (derived)
5. Everything else → phase needs no questions
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    Boolean, Column, ForeignKey, Integer, JSON, String, Table, Text,
)
from sqlalchemy.orm import DeclarativeBase, relationship
import enum

from workflows.interview.phase_engine import BaseInterviewState, PhaseConfig, build_graph


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# M2M join tables
# ---------------------------------------------------------------------------

question_subjects_table = Table(
    "question_subjects",
    Base.metadata,
    Column("question_id", String(36), ForeignKey("questions.id"),  primary_key=True),
    Column("subject_id",  String(36), ForeignKey("subjects.id"),   primary_key=True),
)

question_companies_table = Table(
    "question_companies",
    Base.metadata,
    Column("question_id", String(36), ForeignKey("questions.id"),  primary_key=True),
    Column("company_id",  String(36), ForeignKey("companies.id"),  primary_key=True),
)


# ---------------------------------------------------------------------------
# Subject
# ---------------------------------------------------------------------------

class Subject(Base):
    __tablename__ = "subjects"

    id   = Column(String(36), primary_key=True)
    name = Column(String(255), nullable=False, unique=True)

    questions = relationship(
        "Question",
        secondary=question_subjects_table,
        back_populates="subjects",
    )


# ---------------------------------------------------------------------------
# Company
# ---------------------------------------------------------------------------

class Company(Base):
    __tablename__ = "companies"

    id   = Column(String(36), primary_key=True)
    name = Column(String(255), nullable=False, unique=True)

    questions = relationship(
        "Question",
        secondary=question_companies_table,
        back_populates="companies",
    )


# ---------------------------------------------------------------------------
# Question
# ---------------------------------------------------------------------------

class Question(Base):
    __tablename__ = "questions"

    id          = Column(String(36), primary_key=True)
    title       = Column(String(255), nullable=False, index=True)
    source      = Column(String(100), nullable=True)
    url         = Column(String(200), nullable=True)
    raw_content = Column("raw_content", Text, nullable=True)   # raw HTML
    description = Column(Text, nullable=False)
    difficulty  = Column(String(10), nullable=False, index=True)
    example     = Column(Text, nullable=True)

    subjects  = relationship("Subject",  secondary=question_subjects_table,  back_populates="questions")
    companies = relationship("Company",  secondary=question_companies_table, back_populates="questions")

    def to_phase_dict(self) -> Dict[str, Any]:
        return {
            "id":          self.id,
            "title":       self.title,
            "description": self.description,
            "difficulty":  self.difficulty,
            "raw_content": self.raw_content,
            "example":     self.example,
            "source":      self.source,
            "url":         self.url,
            "subjects":    [s.name for s in self.subjects],
            "companies":   [c.name for c in self.companies],
        }


# ---------------------------------------------------------------------------
# Interview engine models
# ---------------------------------------------------------------------------

class Interview(Base):
    __tablename__ = "interviews"

    id              = Column(Integer, primary_key=True)
    name            = Column(String(255), nullable=False)
    difficulty      = Column(String(50), default="Medium")
    company         = Column(String(255), nullable=True)
    subject         = Column(String(255), nullable=True)
    tags            = Column(String(500), default="")
    # Custom greeting prompt — if set, used instead of the engine's generic greeting.
    # Store the full system prompt string here (e.g. DEBATE_GREETING_PROMPT).
    greeting_prompt = Column(Text, nullable=True)

    phases = relationship(
        "InterviewPhase",
        back_populates="interview",
        order_by="InterviewPhase.order",
    )


class InterviewPhase(Base):
    __tablename__ = "interview_phases"

    id           = Column(Integer, primary_key=True)
    interview_id = Column(Integer, ForeignKey("interviews.id"), nullable=False)

    phase_name    = Column(String(100), nullable=False)
    order         = Column(Integer, nullable=False)
    prompt        = Column(Text, nullable=False)
    prompt_inputs = Column(JSON, default=list)

    number_of_questions_to_ask = Column(Integer, default=0)
    setup_questions            = Column(Boolean, default=False)
    setup_questions_prompt     = Column(Text, default="")
    question_filters           = Column(JSON, default=dict)

    route_nodes        = Column(JSON, default=list)
    route_ahead_prompt = Column(Text, default="")

    immediate_feedback_required = Column(Boolean, default=False)
    feedback_prompt             = Column(Text, default="")

    mcp_tools  = Column(Boolean, default=False)
    tool_names = Column(JSON, default=list)

    special_output_format = Column(String(20), nullable=True)
    entity_schema         = Column(JSON, nullable=True)

    interview = relationship("Interview", back_populates="phases")


# ---------------------------------------------------------------------------
# Hydration
# ---------------------------------------------------------------------------

def hydrate_phase(row: InterviewPhase) -> PhaseConfig:
    return PhaseConfig(
        phase_name=row.phase_name,
        order=row.order,
        prompt=row.prompt,
        prompt_inputs=row.prompt_inputs or [],
        number_of_questions_to_ask=row.number_of_questions_to_ask or 0,
        setup_questions=bool(row.setup_questions),
        setup_questions_prompt=row.setup_questions_prompt or "",
        question_filters=row.question_filters or {},
        route_nodes=row.route_nodes or [],
        route_ahead_prompt=row.route_ahead_prompt or "",
        immediate_feedback_required=bool(row.immediate_feedback_required),
        feedback_prompt=row.feedback_prompt or "",
        mcp_tools=bool(row.mcp_tools),
        tool_names=row.tool_names or [],
        special_output_format=row.special_output_format,
        entity_schema=row.entity_schema,
    )


# ---------------------------------------------------------------------------
# Question fetching
# ---------------------------------------------------------------------------

def _run_question_query(
    db_session,
    subject_name: Optional[str],
    difficulty: Optional[str],
    limit: int,
) -> List[Question]:
    query = db_session.query(Question)

    if subject_name:
        query = (
            query
            .join(Question.subjects)
            .filter(Subject.name.ilike(f"%{subject_name}%"))
        )

    if difficulty:
        query = query.filter(
            Question.difficulty == difficulty.strip().capitalize()
        )

    if limit > 0:
        query = query.limit(limit)

    return query.all()


def fetch_questions_for_phase(
    phase_row: InterviewPhase,
    interview: Interview,
    db_session,
) -> Optional[List[Dict[str, Any]]]:
    filters   = phase_row.question_filters or {}
    has_setup = bool(phase_row.setup_questions)
    limit     = phase_row.number_of_questions_to_ask or 0

    # Case 3 — LLM will generate at runtime
    if has_setup:
        return None

    # Determine source and filter values
    if filters.get("use_db_questions") or filters.get("subject"):
        # Cases 1 & 2 — explicitly requested from DB
        subject_name   = filters.get("subject")   or interview.subject
        difficulty_val = filters.get("difficulty") or interview.difficulty
        source_label   = "explicit_filter"

    elif limit > 0:
        # Case 4 — derive from interview info
        subject_name   = interview.subject
        difficulty_val = interview.difficulty
        source_label   = "derived_from_interview"

    else:
        # Case 5 — phase doesn't need questions
        return None

    print(
        f"[fetch_questions] phase={phase_row.phase_name}  "
        f"source={source_label}  "
        f"subject={subject_name!r}  difficulty={difficulty_val!r}  limit={limit}"
    )

    if not subject_name and not difficulty_val:
        # No usable filter — return everything (safety valve)
        rows = db_session.query(Question).limit(limit or 10).all()
    else:
        rows = _run_question_query(db_session, subject_name, difficulty_val, limit)

    print(f"[fetch_questions] → found {len(rows)} question(s)")
    return [q.to_phase_dict() for q in rows]


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------

def build_graph_for_interview(
    interview_id: int,
    db_session,
    google_api_key: str,
    checkpointer,
):
    interview: Interview = db_session.get(Interview, interview_id)
    if not interview:
        raise ValueError(f"Interview {interview_id} not found")

    background = {
        "name":            interview.name,
        "difficulty":      interview.difficulty,
        "company":         interview.company,
        "subject":         interview.subject,
        "tags":            interview.tags,
        # Only included if set — engine falls back to generic greeting when absent
        "greeting_prompt": interview.greeting_prompt or None,
    }

    phases: List[PhaseConfig] = [hydrate_phase(p) for p in interview.phases]
    if not phases:
        raise ValueError(f"Interview {interview_id} has no phases configured")

    pre_fetched: Dict[str, List[Dict[str, Any]]] = {}
    for phase_row in interview.phases:
        result = fetch_questions_for_phase(phase_row, interview, db_session)
        if result is not None:
            pre_fetched[phase_row.phase_name] = result
            print(
                f"[build_graph_for_interview] pre-fetched {len(result)} questions "
                f"for phase '{phase_row.phase_name}'"
            )

    agent, _interrupt_nodes = build_graph(
        phases=phases,
        state_class=BaseInterviewState,
        google_api_key=google_api_key,
        checkpointer=checkpointer,
    )

    initial_state_extras = {
        "background":      background,
        "phase_questions": pre_fetched,
    }
    return agent, initial_state_extras