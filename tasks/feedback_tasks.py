"""
Celery tasks for feedback generation
"""
from celery import Task
from tasks.celery_app import celery_app
from typing import Dict, Any
import logging
import os
from redis import Redis

from workflows.feedback.technical_feedback import build_tech_skills_feedback_graph, TechIntState
from workflows.feedback.hr_feedback import build_hr_skills_feedback_graph, HRIntState
from workflows.feedback.case_study_feedback import build_case_study_feedback_graph, CaseStudyIntState

logger = logging.getLogger(__name__)


class FeedbackTask(Task):
    """Base task with shared resources"""
    _redis_client = None
    
    @property
    def redis_client(self):
        if self._redis_client is None:
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
            self._redis_client = Redis.from_url(redis_url, decode_responses=True)
        return self._redis_client


@celery_app.task(bind=True, base=FeedbackTask, name="tasks.feedback_tasks.generate_technical_feedback")
def generate_technical_feedback(self, session_id: str, history: str, user_id: str) -> Dict[str, Any]:
    """
    Generate feedback for technical interview
    
    Args:
        session_id: Interview session ID
        history: Conversation history
        user_id: User ID
        
    Returns:
        dict: Feedback results
    """
    try:
        logger.info(f"Generating technical feedback for session {session_id}")
        
        # Get API key
        google_key = os.getenv("GOOGLE_API_KEY", "")
        
        # Build feedback graph
        graph = build_tech_skills_feedback_graph(google_key)
        
        # Run feedback generation
        result = graph.invoke({"history_log": history})
        
        # Extract results
        feedback = {
            "language_score": result["technical"].programming_language,
            "framework_score": result["technical"].framework,
            "algorithms_score": result["technical"].algorithms,
            "data_structures_score": result["technical"].data_structures,
            "approach_score": result["problem_solving"].approach,
            "optimization_score": result["problem_solving"].optimization,
            "debugging_score": result["problem_solving"].debugging,
            "syntax_score": result["problem_solving"].syntax,
            "strengths": [
                result["strengths_and_areas_of_improvements"].strength1,
                result["strengths_and_areas_of_improvements"].strength2,
                result["strengths_and_areas_of_improvements"].strength3,
            ],
            "areas_of_improvements": [
                result["strengths_and_areas_of_improvements"].areas_of_improvements1,
                result["strengths_and_areas_of_improvements"].areas_of_improvements2,
                result["strengths_and_areas_of_improvements"].areas_of_improvements3,
            ],
            "interaction_log_feedback": {
                "answer_status": result["interaction_log_feedback"].answer_status,
                "comment": result["interaction_log_feedback"].comment,
            }
        }
        
        # Store in Redis
        redis_key = f"feedback:{session_id}"
        self.redis_client.setex(redis_key, 3600, str(feedback))
        
        logger.info(f"Technical feedback generated for session {session_id}")
        
        return {
            "status": "completed",
            "feedback": feedback
        }
        
    except Exception as e:
        logger.error(f"Error generating technical feedback: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "feedback": None
        }


@celery_app.task(bind=True, base=FeedbackTask, name="tasks.feedback_tasks.generate_hr_feedback")
def generate_hr_feedback(self, session_id: str, history: str, user_id: str) -> Dict[str, Any]:
    """
    Generate feedback for HR interview
    
    Args:
        session_id: Interview session ID
        history: Conversation history
        user_id: User ID
        
    Returns:
        dict: Feedback results
    """
    try:
        logger.info(f"Generating HR feedback for session {session_id}")
        
        # Get API key
        google_key = os.getenv("GOOGLE_API_KEY", "")
        
        # Build feedback graph
        graph = build_hr_skills_feedback_graph(google_key)
        
        # Run feedback generation
        result = graph.invoke({"history_log": history})
        
        # Extract results
        feedback = {
            "clarity_score": result["communication_skills"].clarity,
            "confidence_score": result["communication_skills"].confidence,
            "structure_score": result["communication_skills"].structure,
            "engagement_score": result["communication_skills"].engagement,
            "values_score": result["cultural_fit"].values,
            "teamwork_score": result["cultural_fit"].teamwork,
            "growth_score": result["cultural_fit"].growth,
            "initiative_score": result["cultural_fit"].initiative,
            "strengths": [
                result["strengths_and_areas_of_improvements"].strength1,
                result["strengths_and_areas_of_improvements"].strength2,
                result["strengths_and_areas_of_improvements"].strength3,
            ],
            "areas_of_improvements": [
                result["strengths_and_areas_of_improvements"].areas_of_improvements1,
                result["strengths_and_areas_of_improvements"].areas_of_improvements2,
                result["strengths_and_areas_of_improvements"].areas_of_improvements3,
            ]
        }
        
        # Store in Redis
        redis_key = f"feedback:{session_id}"
        self.redis_client.setex(redis_key, 3600, str(feedback))
        
        logger.info(f"HR feedback generated for session {session_id}")
        
        return {
            "status": "completed",
            "feedback": feedback
        }
        
    except Exception as e:
        logger.error(f"Error generating HR feedback: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "feedback": None
        }


@celery_app.task(bind=True, base=FeedbackTask, name="tasks.feedback_tasks.generate_case_study_feedback")
def generate_case_study_feedback(self, session_id: str, history: str, user_id: str) -> Dict[str, Any]:
    """
    Generate feedback for case study interview
    
    Args:
        session_id: Interview session ID
        history: Conversation history
        user_id: User ID
        
    Returns:
        dict: Feedback results
    """
    try:
        logger.info(f"Generating case study feedback for session {session_id}")
        
        # Get API key
        google_key = os.getenv("GOOGLE_API_KEY", "")
        
        # Build feedback graph
        graph = build_case_study_feedback_graph(google_key)
        
        # Run feedback generation
        result = graph.invoke({"history_log": history})
        
        # Extract results
        feedback = {
            "problem_understanding_score": result["analytical_skills"].problem_understanding,
            "hypothesis_score": result["analytical_skills"].hypothesis,
            "analysis_score": result["analytical_skills"].analysis,
            "synthesis_score": result["analytical_skills"].synthesis,
            "business_judgment_score": result["business_impact"].business_judgment,
            "creativity_score": result["business_impact"].creativity,
            "decision_making_score": result["business_impact"].decision_making,
            "impact_orientation_score": result["business_impact"].impact_orientation,
            "strengths": [
                result["strengths_and_areas_of_improvements"].strength1,
                result["strengths_and_areas_of_improvements"].strength2,
                result["strengths_and_areas_of_improvements"].strength3,
            ],
            "areas_of_improvements": [
                result["strengths_and_areas_of_improvements"].areas_of_improvements1,
                result["strengths_and_areas_of_improvements"].areas_of_improvements2,
                result["strengths_and_areas_of_improvements"].areas_of_improvements3,
            ]
        }
        
        # Store in Redis
        redis_key = f"feedback:{session_id}"
        self.redis_client.setex(redis_key, 3600, str(feedback))
        
        logger.info(f"Case study feedback generated for session {session_id}")
        
        return {
            "status": "completed",
            "feedback": feedback
        }
        
    except Exception as e:
        logger.error(f"Error generating case study feedback: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "feedback": None
        }
