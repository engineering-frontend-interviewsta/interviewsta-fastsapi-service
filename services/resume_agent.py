"""
Singleton service for the Resume analysis agent (LangGraph).
The resume graph is compiled without a checkpointer; one compiled graph is reused for all analyses.
"""
import os
import logging
from typing import Optional, Any

from workflows.feedback.resume_analysis import build_resume_analysis_graph

logger = logging.getLogger(__name__)


class ResumeAgentService:
    """
    Singleton service for the resume analysis agent.
    Builds the graph once (no checkpointer) and caches the compiled graph.
    """

    _instance: Optional["ResumeAgentService"] = None
    _graph: Optional[Any] = None
    _google_key_used: Optional[str] = None

    def __new__(cls) -> "ResumeAgentService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_graph(self, google_api_key: Optional[str] = None):
        """
        Return the compiled resume analysis graph. Builds once and caches.
        Uses GOOGLE_API_KEY from env if google_api_key is not provided.
        """
        key = google_api_key or os.getenv("GOOGLE_API_KEY", "")
        if self._graph is None or self._google_key_used != key:
            if not key:
                raise ValueError("Resume agent requires GOOGLE_API_KEY (env or argument)")
            logger.info("Building resume analysis graph (singleton, no checkpointer)")
            self._graph = build_resume_analysis_graph(key)
            self._google_key_used = key
        return self._graph

    def invoke(self, inputs: dict, *, google_api_key: Optional[str] = None) -> dict:
        """Run the resume analysis graph with the given inputs."""
        graph = self.get_graph(google_api_key=google_api_key)
        return graph.invoke(inputs)


def get_resume_agent() -> ResumeAgentService:
    """Return the singleton ResumeAgentService instance."""
    return ResumeAgentService()
