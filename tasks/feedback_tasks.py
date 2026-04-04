"""
Celery tasks for feedback generation
"""
from celery import Task
from tasks.celery_app import celery_app
from typing import Dict, Any, Optional
import logging
import os
import json
from redis import Redis
from datetime import datetime

from workflows.feedback.legacy.technical_feedback import build_tech_skills_feedback_graph, TechIntState
from workflows.feedback.legacy.hr_feedback import build_hr_skills_feedback_graph, HRIntState
from workflows.feedback.legacy.case_study_feedback import build_case_study_feedback_graph, CaseStudyIntState
from workflows.feedback.legacy.communication_feedback import build_communication_feedback_graph
from workflows.feedback.legacy.debate_feedback import build_debate_feedback_graph
from services.interview_session import InterviewSessionManager
from langgraph.checkpoint.redis import RedisSaver

from services.llm_metrics import (
    get_big5_from_transcript_llm,
    get_speech_summary_from_transcript_llm,
    get_language_quality_scores_from_transcript_llm,
    get_candidate_transcript_from_messages,
)
from services.dynamic_metrics import get_stored_video_telemetry

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

class FeedbackTask(Task):
    """Base task with shared resources"""
    _redis_client = None
    
    @property
    def redis_client(self):
        if self._redis_client is None:
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
            self._redis_client = Redis.from_url(redis_url, decode_responses=True)
        return self._redis_client

def get_latest_checkpoint(thread_id: str, checkpoint_ns: Optional[str] = None):
    cfg = {"configurable": {"thread_id": thread_id}}
    if checkpoint_ns:
        cfg["configurable"]["checkpoint_ns"] = checkpoint_ns
    
    logger.info(f"Fetching checkpoint for thread_id: {thread_id}")
    
    with RedisSaver.from_conn_string(REDIS_URL) as cp:
        logger.info(f"Using Redis URL : {REDIS_URL}")
        checkpoint = cp.get(cfg)
        
        if checkpoint is None:
            logger.info(f"No checkpoint found for session {thread_id}")
            return None
        
        logger.info(f"Checkpoint found for session {thread_id}")
        return checkpoint

def get_interaction_history_from_redis(session_id):
    latest_checkp = get_latest_checkpoint(session_id)
    logger.info(f"Latest checkpoint retrieved for session_id {session_id}: {latest_checkp}")
    # print("This is channel value",latest_checkp["channel_values"]["history"])
    return latest_checkp["channel_values"]["history"], latest_checkp["channel_values"]["messages"]

def extract_qa_pairs(messages):
        """
        Convert raw Langchain messages (dicts with 'type' and 'data') into Q/A pairs.
        Assumes messages alternate between AI (question) and human (answer).
        Discards any initial AI messages if there is no preceding human answer.
        """
        # print("*********************")
        qa_pairs = []
        question = None
        for ind, msg in enumerate(messages):

            msg_type = getattr(msg,"type")
            # msg = json.loads(msg)
            # _content = getattr(msg,"data", {})
            content = getattr(msg,"content", "")
            duration = getattr(msg,"additional_kwargs",{}).get("timestamp",1232324432)

            # print(f"Message {ind}: type={msg_type}, content={content}")  # Debug print

            if msg_type == "ai":
                # question = content
                qa_pairs.append({"question":content,"timestamp":datetime.fromtimestamp(duration).strftime("%H:%M:%S")})
                # print(f"Set question: {question}")
            elif msg_type == "human":
                qa_pairs.append({"answer": content,"timestamp":datetime.fromtimestamp(duration).strftime("%H:%M:%S")})
                # print(f"Appended Q/A pair: Q: {question} | A: {content}")
                # question = None
            else:
                pass
                # Skip or log messages outside QA pattern
                # print(f"Ignored message type {msg_type} or no question set yet")

        return qa_pairs


def _safe_score(value: Any) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric < 0:
        return 0.0
    if numeric > 100:
        return 100.0
    return round(numeric, 2)


def _get_universal_language_scores(messages, google_key: str) -> Dict[str, Optional[float]]:
    transcript = get_candidate_transcript_from_messages(messages)
    if not transcript.strip() or len(transcript.strip()) <= 30 or not google_key:
        return {"communicationScore": None, "grammarScore": None}

    speech_summary = get_speech_summary_from_transcript_llm(transcript, google_key) or {}
    return {
        "communicationScore": _safe_score(speech_summary.get("clarity")),
        "grammarScore": _safe_score(speech_summary.get("grammar")),
    }


def _get_detailed_language_metrics(messages, google_key: str) -> Dict[str, Any]:
    transcript = get_candidate_transcript_from_messages(messages)
    if not transcript.strip() or len(transcript.strip()) <= 30 or not google_key:
        return {
            "communicationMetrics": None,
            "grammarMetrics": None,
            "communicationScore": None,
            "grammarScore": None,
        }

    detailed = get_language_quality_scores_from_transcript_llm(transcript, google_key)
    if detailed:
        communication_metrics = detailed.get("communicationMetrics")
        grammar_metrics = detailed.get("grammarMetrics")
        return {
            "communicationMetrics": communication_metrics,
            "grammarMetrics": grammar_metrics,
            "communicationScore": _safe_score((communication_metrics or {}).get("overall")),
            "grammarScore": _safe_score((grammar_metrics or {}).get("overall")),
        }

    universal_scores = _get_universal_language_scores(messages, google_key)
    if universal_scores["communicationScore"] is None and universal_scores["grammarScore"] is None:
        return {
            "communicationMetrics": None,
            "grammarMetrics": None,
            "communicationScore": None,
            "grammarScore": None,
        }

    communication_score = universal_scores["communicationScore"]
    grammar_score = universal_scores["grammarScore"]
    return {
        "communicationMetrics": (
            {
                "overall": communication_score,
                "clarity": communication_score,
                "fluency": communication_score,
                "responseRelevance": communication_score,
                "structure": communication_score,
            }
            if communication_score is not None
            else None
        ),
        "grammarMetrics": (
            {
                "overall": grammar_score,
                "grammarCorrectness": grammar_score,
                "sentenceConstruction": grammar_score,
                "vocabularyControl": grammar_score,
                "conciseness": grammar_score,
            }
            if grammar_score is not None
            else None
        ),
        "communicationScore": communication_score,
        "grammarScore": grammar_score,
    }


def _run_unified_feedback_pipeline(
    session_id: str,
    history: str,
    user_email: str,
    session: Dict[str, Any],
    redis_client: Redis,
    feedback_item_id: str,
    google_key: str,
) -> Dict[str, Any]:
    """Run the unified feedback pipeline (feedback_items.json + service.py) and save to DB + Redis."""
    from workflows.feedback.service import run_feedback_pipeline, get_feedback_item
    from services.drf_client import save_feedback_items_to_db

    if not get_feedback_item(feedback_item_id):
        raise ValueError(f"Unknown feedback_item_id: {feedback_item_id}")
    pipeline_result = run_feedback_pipeline(feedback_item_id, history, google_key)
    strengths_obj = pipeline_result.get("strengths_and_improvements") or {}
    strengths_list = [
        strengths_obj.get("strength1", ""),
        strengths_obj.get("strength2", ""),
        strengths_obj.get("strength3", ""),
    ]
    improvements_list = [
        strengths_obj.get("improvement1", ""),
        strengths_obj.get("improvement2", ""),
        strengths_obj.get("improvement3", ""),
    ]
    interaction_feedback = pipeline_result.get("interaction_feedback") or []
    interaction_status_logs = [{"status": x.get("status"), "comment": x.get("comment")} for x in interaction_feedback]
    history_ckpt, messages = get_interaction_history_from_redis(session_id)
    interaction_logs = extract_qa_pairs(messages)[1:] if messages else []

    interview_test_id = (session or {}).get("interview_test_id")
    if interview_test_id is None and session:
        payload = session.get("payload") or {}
        interview_test_id = payload.get("interview_type_id") or payload.get("interview_test_id")
    interview_test_id_str = str(interview_test_id) if interview_test_id is not None else ""
    duration_seconds = session.get("duration", 0) if session else 0
    items = pipeline_result.get("sleeve_scores") or {}

    language_metrics = _get_detailed_language_metrics(messages, google_key)

    save_feedback_items_to_db(
        user_email=user_email,
        session_id=session_id,
        interview_test_id=interview_test_id_str,
        items=items,
        strengths=strengths_list,
        duration_seconds=int(duration_seconds) if duration_seconds else 0,
        areas_of_improvements=improvements_list,
        interaction_logs=interaction_logs,
        interaction_status_logs=interaction_status_logs,
        user_id=(session or {}).get("user_id"),
        communication_score=language_metrics["communicationScore"],
        grammar_score=language_metrics["grammarScore"],
        communication_metrics=language_metrics["communicationMetrics"],
        grammar_metrics=language_metrics["grammarMetrics"],
    )
    feedback = {
        "items": items,
        "strengths": strengths_list,
        "areas_of_improvements": improvements_list,
        "interaction_feedback": interaction_feedback,
        "interaction_status_logs": interaction_status_logs,
        "interview_type_id": feedback_item_id,
        "interview_title": pipeline_result.get("interview_title", ""),
        "communicationScore": language_metrics["communicationScore"],
        "grammarScore": language_metrics["grammarScore"],
        "communicationMetrics": language_metrics["communicationMetrics"],
        "grammarMetrics": language_metrics["grammarMetrics"],
    }
    redis_key = f"feedback:{session_id}"
    redis_client.setex(redis_key, 3600, json.dumps(feedback))
    return {"status": "completed", "feedback": feedback}


@celery_app.task(bind=True, base=FeedbackTask, name="tasks.feedback_tasks.generate_feedback")
def generate_feedback(self, session_id: str, history: str, user_email: str) -> Dict[str, Any]:
    """
    Single entry point for feedback generation. Requires feedback_item_id from session
    (from X-Interview-Access-Token). Uses unified pipeline only: feedback_items.json,
    service.py, graph.py, prompts.py, schemas.py. No legacy per-type feedback.
    """
    logger.info(f"Generating feedback for session {session_id}")
    try:
        google_key = os.getenv("GOOGLE_API_KEY", "")
        if not google_key:
            return {"status": "error", "error": "GOOGLE_API_KEY not set", "feedback": None}
        session_manager = InterviewSessionManager(self.redis_client)
        session = session_manager.get_session(session_id)
        feedback_item_id = (session or {}).get("feedback_item_id") or ((session or {}).get("payload") or {}).get("feedback_item_id")

        if not feedback_item_id:
            return {"status": "error", "error": "feedback_item_id missing (required from X-Interview-Access-Token)", "feedback": None}
        from workflows.feedback.service import get_feedback_item
        if not get_feedback_item(feedback_item_id):
            return {"status": "error", "error": f"Unknown feedback_item_id: {feedback_item_id}", "feedback": None}

        return _run_unified_feedback_pipeline(
            session_id=session_id,
            history=history,
            user_email=user_email,
            session=session or {},
            redis_client=self.redis_client,
            feedback_item_id=feedback_item_id,
            google_key=google_key,
        )
    except Exception as e:
        logger.error(f"Error generating feedback for session {session_id}: {e}", exc_info=True)
        return {"status": "error", "error": str(e), "feedback": None}


@celery_app.task(bind=True, base=FeedbackTask, name="tasks.feedback_tasks.generate_technical_feedback")
def generate_technical_feedback(self, session_id: str, history: str, user_email: str) -> Dict[str, Any]:
    """
    Generate feedback for technical interview
    
    Args:
        session_id: Interview session ID
        history: Conversation history
        user_email: User ID
        
    Returns:
        dict: Feedback results
    """

    
        
    logger.info(f"API Key: {os.getenv('GOOGLE_API_KEY')}")
    try:
        google_key = os.getenv("GOOGLE_API_KEY", "")
        session_manager = InterviewSessionManager(self.redis_client)
        session = session_manager.get_session(session_id)

        # Legacy path only (unified path is in generate_feedback)
        graph = build_tech_skills_feedback_graph(google_key)
        result = graph.invoke({"history_log": history})
        logger.info(f"Technical feedback result: {result}")

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
            },
            "interaction_log_feedback_corrected": {
                "answer_status": result["interaction_log_feedback"].answer_status,
                "comment": result["interaction_log_feedback"].comment,
            }
        }

        redis_key = f"feedback:{session_id}"
        self.redis_client.setex(redis_key, 3600, str(feedback))

        session = session_manager.get_session(session_id)
        
        history, messages = get_interaction_history_from_redis(session_id)
        interaction_log = extract_qa_pairs(messages)[1:]

        # 1) Telemetry: only from stored data (set_video_telemetry) — same key as set endpoint
        video_telemetry = get_stored_video_telemetry(self.redis_client, session_id)

        # 2) Candidate transcript = human messages only (for LLM)
        candidate_transcript = get_candidate_transcript_from_messages(messages)
        google_key = os.getenv("GOOGLE_API_KEY", "")

        # 3) Big5: entirely from LLM (no Redis, no generate_dynamic_metrics)
        personality_scores = None
        if candidate_transcript.strip() and len(candidate_transcript.strip()) > 30 and google_key:
            personality_scores = get_big5_from_transcript_llm(candidate_transcript, google_key)
        big5_profile = personality_scores if personality_scores else {}

        # 4) Communication/speech scores: entirely from LLM (no Redis, no generate_dynamic_metrics)
        communication_scores = None
        if candidate_transcript.strip() and len(candidate_transcript.strip()) > 30 and google_key:
            communication_scores = get_speech_summary_from_transcript_llm(candidate_transcript, google_key)

        # 5) soft_skill_summary = stored video_telemetry + LLM communication_scores only
        soft_skill_summary = dict(video_telemetry) if video_telemetry else {}
        if communication_scores:
            soft_skill_summary["communication_scores"] = communication_scores
            soft_skill_summary["speech_summary"] = communication_scores

        logger.info(f"[IMPORTANT][DEBUG:generate_technical_feedback] Telemetry: {video_telemetry}")
        logger.info(f"[IMPORTANT][DEBUG:generate_technical_feedback] Soft skill summary: {soft_skill_summary}")
        logger.info(f"[IMPORTANT][DEBUG:generate_technical_feedback] Big5 profile: {big5_profile}")

        # Save to Django database
        try:
            from services.drf_client import save_feedback_to_db
            
            interview_type = session.get("interview_type", "Technical") if session else "Technical"
            interview_test_id = (session or {}).get("interview_test_id")
            if interview_test_id is None and session:
                payload = session.get("payload") or {}
                interview_test_id = payload.get("interview_type_id") or payload.get("interview_test_id")
            if interview_test_id is not None:
                try:
                    interview_test_id = int(interview_test_id)
                except (TypeError, ValueError):
                    interview_test_id = None
            duration_seconds = session.get("duration", 0) if session else 0
            
            # Parse history to extract Q&A pairs (simple parsing for now)
            # interaction_log = []  # Will be populated from history if needed

            # hours, remainder = divmod(duration, 3600)
            # minutes, seconds = divmod(remainder, 60)
            # duration = datetime.time(hours, minutes, seconds)
            
            db_saved = save_feedback_to_db(
                user_email=user_email,
                session_id=session_id,
                interview_type=interview_type,
                interview_test_id=interview_test_id,
                duration_seconds=int(duration_seconds) if duration_seconds else 0,
                feedback_data=feedback,
                interaction_log=interaction_log,
                soft_skill_summary=soft_skill_summary,
                big5_profile=big5_profile
            )

            logger.info(f"Technical feedback saved to Django database for session {session_id}: {db_saved}")
            
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
def generate_hr_feedback(self, session_id: str, history: str, user_email: str) -> Dict[str, Any]:
    """
    Generate feedback for HR interview
    
    Args:
        session_id: Interview session ID
        history: Conversation history
        user_email: User ID
        
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

        history, messages = get_interaction_history_from_redis(session_id)
        interaction_log = extract_qa_pairs(messages)[1:]
        # Extract results
        feedback = {
            "clarity_score": result["communication_skills"].clarity,
            "confidence_score": result["communication_skills"].confidence,
            "structure_score": result["communication_skills"].structure,
            "engagement_score": result["communication_skills"].engagement,
            "values_score": result["cultural_skills"].values,
            "teamwork_score": result["cultural_skills"].teamwork,
            "growth_score": result["cultural_skills"].growth,
            "initiative_score": result["cultural_skills"].initiative,
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
            },
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
            from services.drf_client import save_feedback_to_db
            
            interview_type = session.get("interview_type", "HR") if session else "HR"
            interview_test_id = (session or {}).get("interview_test_id")
            if interview_test_id is None and session:
                payload = session.get("payload") or {}
                interview_test_id = payload.get("interview_type_id") or payload.get("interview_test_id")
            if interview_test_id is not None:
                try:
                    interview_test_id = int(interview_test_id)
                except (TypeError, ValueError):
                    interview_test_id = None
            
            duration_seconds = session.get("duration", 0) if session else 0
            
            # Parse history to extract Q&A pairs (simple parsing for now)
            # interaction_log = []  # Will be populated from history if needed
            
            db_saved = save_feedback_to_db(
                user_email=user_email,
                session_id=session_id,
                interview_type=interview_type,
                interview_test_id=interview_test_id,
                duration_seconds=int(duration_seconds) if duration_seconds else 0,
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
def generate_case_study_feedback(self, session_id: str, history: str, user_email: str) -> Dict[str, Any]:
    """
    Generate feedback for case study interview
    
    Args:
        session_id: Interview session ID
        history: Conversation history
        user_email: User ID
        
    Returns:
        dict: Feedback results
    """
    try:
        logger.info(f"Generating case study feedback for session {session_id}")
        # logger.info(f"[INITIAL DEBUG:generate_case_study_feedback] interaction_log: \n{interaction_log}")
        
        # Get API key
        google_key = os.getenv("GOOGLE_API_KEY", "")
        
        # Build feedback graph
        graph = build_case_study_feedback_graph(google_key)
        
        # Run feedback generation
        result = graph.invoke({"history_log": history})

        history, messages = get_interaction_history_from_redis(session_id)
        interaction_log = extract_qa_pairs(messages)[1:]
        # Extract results
        feedback = {
            "problem_understanding_score": result["analytical"].problem_understanding,
            "hypothesis_score": result["analytical"].hypothesis,
            "analysis_score": result["analytical"].analysis,
            "synthesis_score": result["analytical"].synthesis,
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
            ],
            "interaction_log_feedback": {
                "answer_status": result["interaction_log_feedback"].answer_status,
                "comment": result["interaction_log_feedback"].comment,
            },
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
            from services.drf_client import save_feedback_to_db
            
            interview_type = session.get("interview_type", "CaseStudy") if session else "CaseStudy"
            interview_test_id = (session or {}).get("interview_test_id")
            if interview_test_id is None and session:
                payload = session.get("payload") or {}
                interview_test_id = payload.get("interview_type_id") or payload.get("interview_test_id")
            if interview_test_id is not None:
                try:
                    interview_test_id = int(interview_test_id)
                except (TypeError, ValueError):
                    interview_test_id = None
            duration = session.get("duration", 0) if session else 0
            
            # Parse history to extract Q&A pairs (simple parsing for now)
            # interaction_log = []  # Will be populated from history if needed
            
            print("[DEBUG:generate_case_study_feedback] interaction_log: \n", interaction_log)

            db_saved = save_feedback_to_db(
                user_email=user_email,
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


@celery_app.task(bind=True, base=FeedbackTask, name="tasks.feedback_tasks.generate_communication_feedback")
def generate_communication_feedback(self, session_id: str, history: str, user_email: str) -> Dict[str, Any]:
    """
    Generate feedback for Communication interview (Speaking + Comprehension).
    """
    try:
        logger.info(f"Generating communication feedback for session {session_id}")
        google_key = os.getenv("GOOGLE_API_KEY", "")
        graph = build_communication_feedback_graph(google_key)
        result = graph.invoke({"history_log": history})

        history, messages = get_interaction_history_from_redis(session_id)
        interaction_log = extract_qa_pairs(messages)[1:]

        s = result["speaking"]
        c = result["comprehension"]
        il = result["interaction_log_feedback"]
        st = result["strengths_and_areas_of_improvements"]

        feedback = {
            "fluency_score": s.fluency,
            "pronunciation_score": s.pronunciation,
            "vocabulary_range_score": s.vocabulary_range,
            "sentence_construction_score": s.sentence_construction,
            "listening_comprehension_score": c.listening_comprehension,
            "reading_comprehension_score": c.reading_comprehension,
            "contextual_understanding_score": c.contextual_understanding,
            "response_relevance_score": c.response_relevance,
            "strengths": [st.strength1, st.strength2, st.strength3],
            "areas_of_improvements": [st.areas_of_improvements1, st.areas_of_improvements2, st.areas_of_improvements3],
            "interaction_log_feedback": {"answer_status": il.answer_status, "comment": il.comment},
        }

        redis_key = f"feedback:{session_id}"
        self.redis_client.setex(redis_key, 3600, json.dumps(feedback))

        session_manager = InterviewSessionManager(self.redis_client)
        session = session_manager.get_session(session_id)
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

        try:
            from services.drf_client import save_feedback_to_db
            interview_type = session.get("interview_type", "Communication Interview") if session else "Communication Interview"
            interview_test_id = (session or {}).get("interview_test_id")
            if interview_test_id is None and session:
                payload = session.get("payload") or {}
                interview_test_id = payload.get("interview_type_id") or payload.get("interview_test_id")
            if interview_test_id is not None:
                try:
                    interview_test_id = int(interview_test_id)
                except (TypeError, ValueError):
                    interview_test_id = None
            duration = session.get("duration", 0) if session else 0
            # Backend expects interaction_status_log in feedback_data (list of answer_status)
            feedback_data = {**feedback, "interaction_status_log": il.answer_status}
            db_saved = save_feedback_to_db(
                user_email=user_email,
                session_id=session_id,
                interview_type=interview_type,
                interview_test_id=interview_test_id,
                duration_seconds=int(duration) if duration else 0,
                feedback_data=feedback_data,
                interaction_log=interaction_log,
                soft_skill_summary=soft_skill_summary,
                big5_profile=big5_profile,
            )
            if db_saved:
                logger.info(f"Communication feedback saved to Django for session {session_id}")
            else:
                logger.warning(f"Failed to save communication feedback to Django for session {session_id}")
        except Exception as e:
            logger.error(f"Error saving communication feedback to Django: {e}", exc_info=True)

        logger.info(f"Communication feedback generated for session {session_id}")
        return {"status": "completed", "feedback": feedback}
    except Exception as e:
        logger.error(f"Error generating communication feedback: {e}", exc_info=True)
        return {"status": "error", "error": str(e), "feedback": None}


@celery_app.task(bind=True, base=FeedbackTask, name="tasks.feedback_tasks.generate_debate_feedback")
def generate_debate_feedback(self, session_id: str, history: str, user_email: str) -> Dict[str, Any]:
    """
    Generate feedback for Debate interview (Argumentation + Persuasion).
    """
    try:
        logger.info(f"Generating debate feedback for session {session_id}")
        google_key = os.getenv("GOOGLE_API_KEY", "")
        graph = build_debate_feedback_graph(google_key)
        result = graph.invoke({"history_log": history})

        history, messages = get_interaction_history_from_redis(session_id)
        interaction_log = extract_qa_pairs(messages)[1:]

        arg = result["argumentation"]
        pers = result["persuasion"]
        il = result["interaction_log_feedback"]
        st = result["strengths_and_areas_of_improvements"]

        feedback = {
            "argument_structure_score": arg.argument_structure,
            "evidence_usage_score": arg.evidence_usage,
            "logical_reasoning_score": arg.logical_reasoning,
            "counterargument_handling_score": arg.counterargument_handling,
            "persuasiveness_score": pers.persuasiveness,
            "rhetorical_skills_score": pers.rhetorical_skills,
            "audience_awareness_score": pers.audience_awareness,
            "conclusion_strength_score": pers.conclusion_strength,
            "strengths": [st.strength1, st.strength2, st.strength3],
            "areas_of_improvements": [st.areas_of_improvements1, st.areas_of_improvements2, st.areas_of_improvements3],
            "interaction_log_feedback": {"answer_status": il.answer_status, "comment": il.comment},
        }

        redis_key = f"feedback:{session_id}"
        self.redis_client.setex(redis_key, 3600, json.dumps(feedback))

        session_manager = InterviewSessionManager(self.redis_client)
        session = session_manager.get_session(session_id)
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

        try:
            from services.drf_client import save_feedback_to_db
            interview_type = session.get("interview_type", "Debate Interview") if session else "Debate Interview"
            interview_test_id = (session or {}).get("interview_test_id")
            if interview_test_id is None and session:
                payload = session.get("payload") or {}
                interview_test_id = payload.get("interview_type_id") or payload.get("interview_test_id")
            if interview_test_id is not None:
                try:
                    interview_test_id = int(interview_test_id)
                except (TypeError, ValueError):
                    interview_test_id = None
            duration = session.get("duration", 0) if session else 0
            feedback_data = {**feedback, "interaction_status_log": il.answer_status}
            db_saved = save_feedback_to_db(
                user_email=user_email,
                session_id=session_id,
                interview_type=interview_type,
                interview_test_id=interview_test_id,
                duration_seconds=int(duration) if duration else 0,
                feedback_data=feedback_data,
                interaction_log=interaction_log,
                soft_skill_summary=soft_skill_summary,
                big5_profile=big5_profile,
            )
            if db_saved:
                logger.info(f"Debate feedback saved to Django for session {session_id}")
            else:
                logger.warning(f"Failed to save debate feedback to Django for session {session_id}")
        except Exception as e:
            logger.error(f"Error saving debate feedback to Django: {e}", exc_info=True)

        logger.info(f"Debate feedback generated for session {session_id}")
        return {"status": "completed", "feedback": feedback}
    except Exception as e:
        logger.error(f"Error generating debate feedback: {e}", exc_info=True)
        return {"status": "error", "error": str(e), "feedback": None}
