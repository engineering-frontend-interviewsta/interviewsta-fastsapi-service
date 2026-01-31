"""
Celery tasks for feedback generation
"""
from celery import Task
from tasks.celery_app import celery_app
from typing import Dict, Any
import logging
import os
import json
from redis import Redis

from workflows.feedback.technical_feedback import build_tech_skills_feedback_graph, TechIntState
from workflows.feedback.hr_feedback import build_hr_skills_feedback_graph, HRIntState
from workflows.feedback.case_study_feedback import build_case_study_feedback_graph, CaseStudyIntState
from services.interview_session import InterviewSessionManager

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
        
        # Get session data for saving to Django DB
        session_manager = InterviewSessionManager(self.redis_client)
        session = session_manager.get_session(session_id)
        
        # Get soft skills and big5 from Redis
        soft_skill_summary = None
        big5_profile = None
        try:
            soft_skills_key = f"session:{session_id}:soft_skills_summary"
            soft_skills_json = self.redis_client.get(soft_skills_key)
            if soft_skills_json:
                soft_skill_summary = json.loads(soft_skills_json)
            
            big5_key = f"big5_profile:{session_id}"
            big5_json = self.redis_client.get(big5_key)
            if big5_json:
                big5_profile = json.loads(big5_json)
        except Exception as e:
            logger.warning(f"Could not retrieve soft skills/Big-5 for session {session_id}: {e}")
        
        # Save to Django database
        try:
            from services.django_db import save_feedback_to_db
            
            interview_type = session.get("interview_type", "Technical") if session else "Technical"
            interview_test_id = session.get("interview_test_id") if session else None
            duration = session.get("duration", 0) if session else 0
            
            # Parse history to extract Q&A pairs (simple parsing for now)
            interaction_log = []  # Will be populated from history if needed
            
            db_saved = save_feedback_to_db(
                user_id=user_id,
                session_id=session_id,
                interview_type=interview_type,
                interview_test_id=interview_test_id,
                duration_seconds=int(duration) if duration else 0,
                feedback_data=feedback,
                interaction_log=interaction_log,
                soft_skill_summary=soft_skill_summary,
                big5_profile=big5_profile
            )
            
            if db_saved:
                logger.info(f"Technical feedback saved to Django database for session {session_id}")
            else:
                logger.warning(f"Failed to save technical feedback to Django database for session {session_id}")
        except Exception as e:
            logger.error(f"Error saving technical feedback to Django database: {e}", exc_info=True)
            # Don't fail the task if DB save fails, Redis storage is sufficient for immediate access
        
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
        
        # Get session data for saving to Django DB
        session_manager = InterviewSessionManager(self.redis_client)
        session = session_manager.get_session(session_id)
        
        # Get soft skills and big5 from Redis
        soft_skill_summary = None
        big5_profile = None
        try:
            soft_skills_key = f"session:{session_id}:soft_skills_summary"
            soft_skills_json = self.redis_client.get(soft_skills_key)
            if soft_skills_json:
                soft_skill_summary = json.loads(soft_skills_json)
            
            big5_key = f"big5_profile:{session_id}"
            big5_json = self.redis_client.get(big5_key)
            if big5_json:
                big5_profile = json.loads(big5_json)
        except Exception as e:
            logger.warning(f"Could not retrieve soft skills/Big-5 for session {session_id}: {e}")
        
        # Save to Django database
        try:
            from services.django_db import save_feedback_to_db
            
            interview_type = session.get("interview_type", "HR") if session else "HR"
            interview_test_id = session.get("interview_test_id") if session else None
            duration = session.get("duration", 0) if session else 0
            
            # Parse history to extract Q&A pairs (simple parsing for now)
            interaction_log = []  # Will be populated from history if needed
            
            db_saved = save_feedback_to_db(
                user_id=user_id,
                session_id=session_id,
                interview_type=interview_type,
                interview_test_id=interview_test_id,
                duration_seconds=int(duration) if duration else 0,
                feedback_data=feedback,
                interaction_log=interaction_log,
                soft_skill_summary=soft_skill_summary,
                big5_profile=big5_profile
            )
            
            if db_saved:
                logger.info(f"HR feedback saved to Django database for session {session_id}")
            else:
                logger.warning(f"Failed to save HR feedback to Django database for session {session_id}")
        except Exception as e:
            logger.error(f"Error saving HR feedback to Django database: {e}", exc_info=True)
            # Don't fail the task if DB save fails, Redis storage is sufficient for immediate access
        
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
        
        # Get session data for saving to Django DB
        session_manager = InterviewSessionManager(self.redis_client)
        session = session_manager.get_session(session_id)
        
        # Get soft skills and big5 from Redis
        soft_skill_summary = None
        big5_profile = None
        try:
            soft_skills_key = f"session:{session_id}:soft_skills_summary"
            soft_skills_json = self.redis_client.get(soft_skills_key)
            if soft_skills_json:
                soft_skill_summary = json.loads(soft_skills_json)
            
            big5_key = f"big5_profile:{session_id}"
            big5_json = self.redis_client.get(big5_key)
            if big5_json:
                big5_profile = json.loads(big5_json)
        except Exception as e:
            logger.warning(f"Could not retrieve soft skills/Big-5 for session {session_id}: {e}")
        
        # Save to Django database
        try:
            from services.django_db import save_feedback_to_db
            
            interview_type = session.get("interview_type", "CaseStudy") if session else "CaseStudy"
            interview_test_id = session.get("interview_test_id") if session else None
            duration = session.get("duration", 0) if session else 0
            
            # Parse history to extract Q&A pairs (simple parsing for now)
            interaction_log = []  # Will be populated from history if needed
            
            db_saved = save_feedback_to_db(
                user_id=user_id,
                session_id=session_id,
                interview_type=interview_type,
                interview_test_id=interview_test_id,
                duration_seconds=int(duration) if duration else 0,
                feedback_data=feedback,
                interaction_log=interaction_log,
                soft_skill_summary=soft_skill_summary,
                big5_profile=big5_profile
            )
            
            if db_saved:
                logger.info(f"Case study feedback saved to Django database for session {session_id}")
            else:
                logger.warning(f"Failed to save case study feedback to Django database for session {session_id}")
        except Exception as e:
            logger.error(f"Error saving case study feedback to Django database: {e}", exc_info=True)
            # Don't fail the task if DB save fails, Redis storage is sufficient for immediate access
        
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
