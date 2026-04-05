"""
Legacy path: interview telemetry scoring lives in ``services.telemetry_scoring``.

Import from there, e.g.::

    from services.telemetry_scoring import ScoringEngine, log_telemetry_scoring_at_session_end
"""

from services.telemetry_scoring import *  # noqa: F403
