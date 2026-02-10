"""
Singleton service for Interview agents (LangGraph).
Graphs are compiled with a shared checkpointer (Redis by default); state is isolated per
session via thread_id in config. One compiled graph per interview type is cached and reused.
"""
import os
import logging
from typing import Optional, Any, Dict, List

from langgraph.checkpoint.redis import RedisSaver
from workflows.technical import get_technical_graph
from workflows.hr import get_hr_graph
from workflows.coding import get_graph as get_coding_graph
from workflows.case_study import build_case_study_graph
from workflows.communication import build_communication_graph

logger = logging.getLogger(__name__)

# Interview types that use a checkpointer (session state)
INTERVIEW_TYPES = ("Technical", "HR", "Company", "Subject", "CaseStudy", "Communication")

# Interrupt nodes per type (for human-in-the-loop)
INTERRUPT_NODES: Dict[str, List[str]] = {
    "Technical": ["Greeting_after", "Technical_after", "Coding_after", "Project_after"],
    "HR": ["Greeting_after", "HR_after"],
    "Company": ["Greeting_after", "Coding_after"],
    "Subject": ["Greeting_after", "Coding_after"],
    "CaseStudy": ["Greeting_after", "CaseStudy_after"],
    "Communication": ["Greeting_after", "Rapport_after", "Dictation_after", "Comprehension_after", "MCQ_after"],
}


class InterviewAgentService:
    """
    Singleton service for interview workflow agents.
    Uses a single checkpointer (Redis by default); each invoke passes thread_id=session_id
    so state is persisted per session. Compiled graphs are cached per interview type.
    """

    _instance: Optional["InterviewAgentService"] = None
    _checkpointer: Any = None
    _redis_cm: Any = None
    _setup_done: bool = False
    _graphs: Dict[str, Any] = {}
    _google_key: Optional[str] = None
    _tavily_key: Optional[str] = None

    def __new__(cls, checkpointer: Any = None) -> "InterviewAgentService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            if checkpointer is not None:
                cls._instance._checkpointer = checkpointer
        return cls._instance

    def get_checkpointer(self):
        """
        Return the shared checkpointer. Uses Redis by default; call once to initialize.
        Pass checkpointer=... in constructor for tests (e.g. InMemorySaver()).
        """
        if self._checkpointer is not None:
            if not self._setup_done:
                try:
                    self._checkpointer.setup()
                except Exception as e:
                    if "already exists" not in str(e).lower():
                        raise
                self._setup_done = True
            return self._checkpointer

        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        if self._redis_cm is None:
            self._redis_cm = RedisSaver.from_conn_string(redis_url)
            self._checkpointer = self._redis_cm.__enter__()
        if not self._setup_done:
            try:
                self._checkpointer.setup()
            except Exception as e:
                if "already exists" not in str(e).lower():
                    raise
            self._setup_done = True
        return self._checkpointer

    def _ensure_keys(self) -> tuple:
        google_key = os.getenv("GOOGLE_API_KEY", "")
        tavily_key = os.getenv("TAVILY_API_KEY", "")
        if not google_key:
            raise ValueError("Interview agent requires GOOGLE_API_KEY")
        return google_key, tavily_key

    def get_graph(
        self,
        interview_type: str,
        *,
        google_api_key: Optional[str] = None,
        tavily_api_key: Optional[str] = None,
    ):
        """
        Return the compiled workflow for the given interview type. Builds once per type
        and caches; all graphs share the same checkpointer. Use config_for_session(session_id)
        when invoking so state is stored per session (thread_id).
        """
        if interview_type not in INTERVIEW_TYPES:
            raise ValueError(
                f"Invalid interview_type: {interview_type}. Must be one of {INTERVIEW_TYPES}"
            )

        if interview_type in self._graphs:
            return self._graphs[interview_type]

        google_key = google_api_key or os.getenv("GOOGLE_API_KEY", "")
        tavily_key = tavily_api_key or os.getenv("TAVILY_API_KEY", "")
        if not google_key:
            raise ValueError("Interview agent requires GOOGLE_API_KEY (env or argument)")

        checkpointer = self.get_checkpointer()

        logger.info("Building interview graph (singleton): %s", interview_type)
        if interview_type == "Technical":
            graph = get_technical_graph(google_key, tavily_key, checkpointer)
        elif interview_type == "HR":
            graph = get_hr_graph(google_key, tavily_key, checkpointer)
        elif interview_type in ("Company", "Subject"):
            graph = get_coding_graph(interview_type, google_key, tavily_key, checkpointer)
        elif interview_type == "CaseStudy":
            graph = build_case_study_graph(google_key, checkpointer)
        elif interview_type == "Communication":
            graph = build_communication_graph(google_key, checkpointer)
        else:
            raise ValueError(f"Unknown interview type: {interview_type}")

        self._graphs[interview_type] = graph
        self._google_key = google_key
        self._tavily_key = tavily_key
        return graph

    def config_for_session(self, session_id: str, recursion_limit: int = 150) -> dict:
        """Return LangGraph config for this session (thread_id = session_id)."""
        return {
            "configurable": {"thread_id": session_id},
            "recursion_limit": recursion_limit,
        }

    def get_interrupt_nodes(self, interview_type: str) -> List[str]:
        """Return the interrupt_before node list for this interview type."""
        return INTERRUPT_NODES.get(interview_type, [])


def get_interview_agent(checkpointer: Any = None) -> InterviewAgentService:
    """Return the singleton InterviewAgentService. Pass checkpointer for tests (e.g. InMemorySaver())."""
    return InterviewAgentService(checkpointer=checkpointer)
