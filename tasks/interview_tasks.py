"""
Celery tasks for interview processing
"""
from celery import Task
from tasks.celery_app import celery_app
from typing import Dict, Any
import logging
import json
from redis import Redis
import os
from datetime import datetime, timedelta

from services.interview_session import InterviewSessionManager
from workflows.technical import get_technical_graph, TechnicalInterviewState
from workflows.hr import get_hr_graph, HRInterviewState  
from workflows.coding import get_graph, CompanyInterviewState, SubjectInterviewState
from workflows.case_study import build_case_study_graph, CaseStudyInterviewState
from langgraph.checkpoint.redis import RedisSaver
from langchain_core.messages import HumanMessage
from services.audio_processor import AudioProcessor

logger = logging.getLogger(__name__)


class InterviewTask(Task):
    """Base task with shared resources"""
    _redis_client = None
    _session_manager = None
    _audio_processor = None
    
    @property
    def redis_client(self):
        if self._redis_client is None:
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
            self._redis_client = Redis.from_url(redis_url, decode_responses=True)
        return self._redis_client
    
    @property
    def session_manager(self):
        if self._session_manager is None:
            self._session_manager = InterviewSessionManager(self.redis_client)
        return self._session_manager
    
    @property
    def audio_processor(self):
        if self._audio_processor is None:
            # Initialize audio processor for Cartesia STT and AWS Polly TTS
            cartesia_api_key = os.getenv("CARTESIA_API_KEY", "")
            cartesia_model = os.getenv("CARTESIA_MODEL", "ink-whisper")
            aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID", "")
            aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY", "")
            aws_region = os.getenv("AWS_REGION", "ap-south-1")
            polly_voice_id = os.getenv("AWS_POLLY_VOICE_ID", "Joanna")
            polly_engine = os.getenv("AWS_POLLY_ENGINE", "neural")
            polly_speech_rate = os.getenv("AWS_POLLY_SPEECH_RATE", "85%")
            
            self._audio_processor = AudioProcessor(
                cartesia_api_key=cartesia_api_key,
                aws_access_key_id=aws_access_key_id or None,
                aws_secret_access_key=aws_secret_access_key or None,
                aws_region=aws_region,
                polly_voice_id=polly_voice_id,
                polly_engine=polly_engine,
                polly_speech_rate=polly_speech_rate,
                cartesia_model=cartesia_model
            )
        return self._audio_processor


@celery_app.task(bind=True, base=InterviewTask, name="tasks.interview_tasks.process_interview_start")
def process_interview_start(self, session_id: str, interview_type: str, user_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Initialize interview workflow and generate greeting
    
    Args:
        session_id: Session identifier
        interview_type: Type of interview (Technical, HR, Company, Subject, CaseStudy)
        user_id: Firebase user ID
        payload: Interview initialization data
        
    Returns:
        dict: Initial response with greeting
    """
    try:
        logger.info(f"Starting {interview_type} interview for session {session_id}")
        
        # Create session
        logger.info(f"Creating session {session_id} in Redis for user {user_id}")
        self.session_manager.create_session(session_id, interview_type, user_id, payload)
        self.session_manager.set_status(session_id, "processing")
        logger.info(f"Session {session_id} created and marked as processing")
        
        # Get API keys
        google_key = os.getenv("GOOGLE_API_KEY", "")
        tavily_key = os.getenv("TAVILY_API_KEY", "")
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        
        # Setup checkpointer
        redis_cm = RedisSaver.from_conn_string(redis_url)
        checkpointer = redis_cm.__enter__()
        
        # Setup indexes (ignore if already exist)
        try:
            checkpointer.setup()
        except Exception as e:
            # Index already exists, which is fine
            if "already exists" not in str(e).lower():
                raise
        
        # Config for LangGraph
        config = {"configurable": {"thread_id": session_id}}
        
        # Build workflow based on interview type
        workflow = None
        initial_state = None
        interrupt_nodes = []
        
        if interview_type == "Technical":
            workflow = get_technical_graph(google_key, tavily_key, checkpointer)
            initial_state = TechnicalInterviewState(
                LastNode="default",
                resume=payload.get("resume", ""),
                history="",
                TechnicalResearch=payload.get("TechnicalResearch", ""),
                CodingResearch=payload.get("CodingResearch", "")
            )
            interrupt_nodes = ["Greeting_after", "Technical_after", "Coding_after", "Project_after"]
            
        elif interview_type == "HR":
            workflow = get_hr_graph(google_key, tavily_key, checkpointer)
            initial_state = HRInterviewState(
                LastNode="default",
                resume=payload.get("resume", ""),
                history=""
            )
            interrupt_nodes = ["Greeting_after", "HR_after"]
            
        elif interview_type in ["Company", "Subject"]:
            workflow = get_graph(interview_type, google_key, tavily_key, checkpointer)
            
            if interview_type == "Company":
                initial_state = CompanyInterviewState(
                    LastNode="default",
                    company=payload.get("company", ""),
                    QuestionResearch=payload.get("QuestionResearch", ""),
                    history="",
                    Difficulty=payload.get("Difficulty", "Medium"),
                    Tags=payload.get("Tags", [])
                )
            else:
                initial_state = SubjectInterviewState(
                    LastNode="default",
                    subject=payload.get("subject", ""),
                    QuestionResearch=payload.get("QuestionResearch", ""),
                    history="",
                    Difficulty=payload.get("Difficulty", "Medium"),
                    Tags=payload.get("Tags", [])
                )
            interrupt_nodes = ["Greeting_after", "Coding_after"]
            
        elif interview_type == "CaseStudy":
            workflow = build_case_study_graph(google_key, checkpointer)
            initial_state = CaseStudyInterviewState(
                LastNode="",
                messages=[],
                history="",
                current_query="",
                current_case_question="",
                current_case_reference="",
                case_completed=False
            )
            interrupt_nodes = ["Greeting_after", "CaseStudy_after"]
        
        if not workflow:
            raise ValueError(f"Invalid interview type: {interview_type}")
        
        # Invoke workflow
        response = workflow.invoke(initial_state, config=config, interrupt_before=interrupt_nodes)
        
        # Extract greeting message
        message = response['messages'][-1].content if response.get('messages') else ""
        last_node = response.get('LastNode', '')
        
        # Synthesize audio with AWS Polly (call directly, don't use Celery task from within task)
        audio_base64 = None
        if message:
            try:
                logger.info(f"Synthesizing audio for interview {session_id}")
                audio_base64 = self.audio_processor.synthesize_speech_base64(message)
                logger.info(f"Audio synthesis successful for {session_id}")
            except Exception as e:
                logger.error(f"Error synthesizing audio for {session_id}: {e}")
                # Continue without audio if synthesis fails
        
        # Update session with response (don't store full workflow_state, LangGraph handles that)
        self.session_manager.update_session(session_id, {
            "message_count": len(response.get('messages', [])),
            "history": response.get('history', ''),
            "last_node": last_node
        })
        
        # Store response for retrieval with audio
        self.session_manager.set_response(session_id, message, audio_base64, last_node)
        self.session_manager.set_status(session_id, "ai_responded")
        
        logger.info(f"Interview {session_id} initialized successfully")
        
        return {
            "session_id": session_id,
            "status": "ai_responded",
            "message": message,
            "last_node": last_node
        }
        
    except Exception as e:
        logger.error(f"Error starting interview {session_id}: {e}", exc_info=True)
        self.session_manager.set_status(session_id, "error")
        raise


@celery_app.task(bind=True, base=InterviewTask, name="tasks.interview_tasks.process_user_response")
def process_user_response(self, session_id: str, human_input: str) -> Dict[str, Any]:
    """
    Process user response and generate AI reply
    
    Args:
        session_id: Session identifier
        human_input: User's transcribed response
        
    Returns:
        dict: AI response
    """
    try:
        logger.info(f"Processing user response for session {session_id}")
        
        # Get session
        session = self.session_manager.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        self.session_manager.set_status(session_id, "processing")
        
        # Get workflow configuration
        interview_type = session["interview_type"]
        google_key = os.getenv("GOOGLE_API_KEY", "")
        tavily_key = os.getenv("TAVILY_API_KEY", "")
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        
        # Setup checkpointer
        redis_cm = RedisSaver.from_conn_string(redis_url)
        checkpointer = redis_cm.__enter__()
        config = {"configurable": {"thread_id": session_id}}
        
        # Rebuild workflow
        workflow = None
        interrupt_nodes = []
        
        if interview_type == "Technical":
            workflow = get_technical_graph(google_key, tavily_key, checkpointer)
            interrupt_nodes = ["Greeting_after", "Technical_after", "Coding_after", "Project_after"]
        elif interview_type == "HR":
            workflow = get_hr_graph(google_key, tavily_key, checkpointer)
            interrupt_nodes = ["Greeting_after", "HR_after"]
        elif interview_type in ["Company", "Subject"]:
            workflow = get_graph(interview_type, google_key, tavily_key, checkpointer)
            interrupt_nodes = ["Greeting_after", "Coding_after"]
        elif interview_type == "CaseStudy":
            workflow = build_case_study_graph(google_key, checkpointer)
            interrupt_nodes = ["Greeting_after", "CaseStudy_after"]
        
        # Get current state and update with user input
        current_state = workflow.get_state(config)
        
        # Check if interview is finished
        if not len(current_state.next):
            # Compute and store soft skills summary before completion
            try:
                soft_skills_summary = self.session_manager.get_soft_skills_summary(session_id)
                if soft_skills_summary:
                    soft_skills_key = f"session:{session_id}:soft_skills_summary"
                    self.redis_client.setex(soft_skills_key, 3600, json.dumps(soft_skills_summary))
                    logger.info(f"Stored soft skills summary for session {session_id}")
            except Exception as e:
                logger.warning(f"Failed to compute soft skills summary for session {session_id}: {e}")
            
            # Clear processing flag
            processing_key = f"session:{session_id}:processing"
            self.redis_client.delete(processing_key)
            
            self.session_manager.set_status(session_id, "completed")
            return {
                "session_id": session_id,
                "status": "completed",
                "message": "",
                "last_node": "finished"
            }
        
        # Update state with user input
        if interview_type == "CaseStudy":
            # For CaseStudy, append HumanMessage
            human_message = HumanMessage(content=human_input)
            current_messages = current_state.values.get("messages", [])
            updated_messages = current_messages + [human_message]
            
            workflow.update_state(config, {
                "messages": updated_messages,
                "history": current_state.values.get("history", "") + "\nInterviewee-" + human_input
            })
        else:
            # For other types
            messages = current_state.values.get("messages", [])
            messages.append(human_input)
            
            workflow.update_state(config, {
                "messages": messages,
                "history": current_state.values.get("history", "") + "\nInterviewee-" + human_input
            })
        
        # Invoke workflow
        response = workflow.invoke(None, config=config, interrupt_before=interrupt_nodes)
        
        # Extract AI message
        message = response['messages'][-1].content if response.get('messages') else ""
        last_node = response.get('LastNode', '')
        
        # Synthesize audio with AWS Polly (call directly, don't use Celery task from within task)
        audio_base64 = None
        if message:
            try:
                logger.info(f"Synthesizing audio for response in session {session_id}")
                audio_base64 = self.audio_processor.synthesize_speech_base64(message)
                logger.info(f"Audio synthesis successful for {session_id}")
            except Exception as e:
                logger.error(f"Error synthesizing audio for {session_id}: {e}")
                # Continue without audio if synthesis fails
        
        # Update session (don't store full messages, LangGraph handles state)
        self.session_manager.update_session(session_id, {
            "message_count": len(response.get('messages', [])),
            "history": response.get('history', ''),
            "last_node": last_node
        })
        
        # Store response with audio
        self.session_manager.set_response(session_id, message, audio_base64, last_node)
        self.session_manager.set_status(session_id, "ai_responded")
        
        # Clear processing flag
        processing_key = f"session:{session_id}:processing"
        self.redis_client.delete(processing_key)
        
        logger.info(f"User response processed for session {session_id}")
        
        return {
            "session_id": session_id,
            "status": "ai_responded",
            "message": message,
            "last_node": last_node
        }
        
    except Exception as e:
        logger.error(f"Error processing response for session {session_id}: {e}", exc_info=True)
        self.session_manager.set_status(session_id, "error")
        # Clear processing flag on error
        processing_key = f"session:{session_id}:processing"
        self.redis_client.delete(processing_key)
        raise


@celery_app.task(bind=True, base=InterviewTask, name="tasks.interview_tasks.cleanup_expired_sessions")
def cleanup_expired_sessions(self):
    """Periodic task to cleanup expired sessions"""
    try:
        logger.info("Running session cleanup")
        # This would need to scan Redis for expired sessions
        # For now, Redis TTL handles expiration automatically
        return {"status": "completed", "message": "Cleanup completed"}
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        raise
