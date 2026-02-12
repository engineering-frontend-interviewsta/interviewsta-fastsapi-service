"""
Celery tasks for interview processing - OPTIMIZED VERSION (FIXED)
"""
from celery import Task
from tasks.celery_app import celery_app
from typing import Dict, Any, Optional
import logging
import json
from redis import Redis
import os
from datetime import datetime, timedelta

from services.interview_session import InterviewSessionManager
from services.interview_agent import get_interview_agent
from workflows.technical import TechnicalInterviewState
from workflows.hr import HRInterviewState
from workflows.coding import CompanyInterviewState, SubjectInterviewState
from workflows.case_study import CaseStudyInterviewState
from langchain_core.messages import HumanMessage
from services.audio_processor import AudioProcessor

logger = logging.getLogger(__name__)


# ============================================================================
# BASE TASK CLASS
# ============================================================================

class InterviewTask(Task):
    """Base task with shared resources (singleton pattern)"""
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
            cartesia_api_version = os.getenv("CARTESIA_API_VERSION", "2025-04-16")
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
                cartesia_model=cartesia_model,
                cartesia_api_version=cartesia_api_version
            )
        return self._audio_processor


# ============================================================================
# HELPER FUNCTIONS (Module-level, not class methods)
# ============================================================================

def create_initial_state(interview_type: str, payload: Dict[str, Any]):
    """Create initial state for interview workflow"""
    if interview_type == "Technical":
        return TechnicalInterviewState(
            LastNode="default",
            resume=payload.get("resume", ""),
            history="",
            TechnicalResearch=payload.get("TechnicalResearch", ""),
            CodingResearch=payload.get("CodingResearch", "")
        )
    elif interview_type == "HR":
        return HRInterviewState(
            LastNode="default",
            resume=payload.get("resume", ""),
            history=""
        )
    elif interview_type == "Company":
        return CompanyInterviewState(
            LastNode="default",
            company=payload.get("company", ""),
            QuestionResearch=payload.get("QuestionResearch", ""),
            history="",
            Difficulty=payload.get("Difficulty", "Medium"),
            Tags=payload.get("Tags", [])
        )
    elif interview_type == "Subject":
        return SubjectInterviewState(
            LastNode="default",
            subject=payload.get("subject", ""),
            QuestionResearch=payload.get("QuestionResearch", ""),
            history="",
            Difficulty=payload.get("Difficulty", "Medium"),
            Tags=payload.get("Tags", [])
        )
    elif interview_type == "CaseStudy":
        return CaseStudyInterviewState(
            LastNode="",
            messages=[],
            history="",
            current_query="",
            current_case_question="",
            current_case_reference="",
            case_completed=False
        )
    else:
        raise ValueError(f"Invalid interview type: {interview_type}")


def update_workflow_state(workflow, config, interview_type: str, current_state, human_input: str):
    """Update workflow state based on interview type"""
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


def clear_processing_flag(redis_client: Redis, session_id: str):
    """Clear processing flag for session"""
    try:
        processing_key = f"session:{session_id}:processing"
        redis_client.delete(processing_key)
        logger.debug(f"Cleared processing flag for {session_id}")
    except Exception as e:
        logger.warning(f"Failed to clear processing flag for {session_id}: {e}")


def handle_interview_completion(redis_client: Redis, session_manager, session_id: str):
    """Handle interview completion - compute soft skills, update status"""
    try:
        # Compute and store soft skills summary
        soft_skills_summary = session_manager.get_soft_skills_summary(session_id)
        if soft_skills_summary:
            soft_skills_key = f"session:{session_id}:soft_skills_summary"
            redis_client.setex(soft_skills_key, 3600, json.dumps(soft_skills_summary))
            logger.info(f"Stored soft skills summary for session {session_id}")
    except Exception as e:
        logger.warning(f"Failed to compute soft skills summary for {session_id}: {e}")
    
    # Update status
    clear_processing_flag(redis_client, session_id)
    session_manager.set_status(session_id, "completed")


# ============================================================================
# TASK: Start Interview
# ============================================================================

@celery_app.task(
    bind=True,
    base=InterviewTask,
    name="tasks.interview_tasks.process_interview_start",
    max_retries=3,
    default_retry_delay=5
)
def process_interview_start(
    self,
    session_id: str,
    interview_type: str,
    user_id: str,
    payload: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Initialize interview workflow and generate greeting.
    
    Pipeline:
    1. Create session in Redis
    2. Initialize workflow state
    3. Invoke workflow to get greeting
    4. Generate TTS audio
    5. Store response
    
    Args:
        session_id: Session identifier
        interview_type: Type of interview (Technical, HR, Company, Subject, CaseStudy)
        user_id: Firebase user ID
        payload: Interview initialization data
        
    Returns:
        dict: Initial response with greeting and audio
    """
    processing_key = f"session:{session_id}:processing"
    
    try:
        logger.info(f"Starting {interview_type} interview for session {session_id}")
        
        # Step 1: Create session (20% progress)
        self.update_state(
            state="PROGRESS",
            meta={"progress": 20, "message": "Creating session..."}
        )
        
        logger.info(f"Creating session {session_id} in Redis for user {user_id}")
        self.session_manager.create_session(session_id, interview_type, user_id, payload)
        
        # Set processing flag (expires in 60s as safety)
        self.redis_client.setex(processing_key, 60, "true")
        self.session_manager.set_status(session_id, "processing")
        
        logger.info(f"Session {session_id} created and marked as processing")
        
        # Step 2: Initialize workflow (40% progress)
        self.update_state(
            state="PROGRESS",
            meta={"progress": 40, "message": "Initializing workflow..."}
        )
        
        # Get singleton interview agent
        agent = get_interview_agent()
        workflow = agent.get_graph(interview_type)
        config = agent.config_for_session(session_id)
        interrupt_nodes = agent.get_interrupt_nodes(interview_type)
        
        # Create initial state using module-level function
        initial_state = create_initial_state(interview_type, payload)
        
        logger.info(f"Workflow initialized for {session_id}, invoking...")
        
        # Step 3: Invoke workflow (60% progress)
        self.update_state(
            state="PROGRESS",
            meta={"progress": 60, "message": "Generating greeting..."}
        )
        
        response = workflow.invoke(
            initial_state,
            config=config,
            interrupt_before=interrupt_nodes
        )
        
        # Extract greeting message
        message = response['messages'][-1].content if response.get('messages') else ""
        last_node = response.get('LastNode', '')
        
        if not message:
            raise ValueError("No greeting message generated by workflow")
        
        logger.info(f"Greeting generated for {session_id}: '{message[:100]}...'")
        
        # Step 4: Synthesize audio (80% progress)
        self.update_state(
            state="PROGRESS",
            meta={"progress": 80, "message": "Generating audio..."}
        )
        
        logger.info(f"Synthesizing audio for interview {session_id}")
        
        audio_base64 = None
        try:
            audio_base64 = self.audio_processor.synthesize_speech_base64(message)
            logger.info(f"Audio synthesis successful for {session_id}")
        except Exception as e:
            logger.error(f"Error synthesizing audio for {session_id}: {e}")
            # Continue without audio - better than failing completely
        
        # Check if audio is required (can make this configurable)
        if not audio_base64:
            logger.warning(f"No audio generated for {session_id}, continuing without audio")
        
        # Step 5: Store response (95% progress)
        self.update_state(
            state="PROGRESS",
            meta={"progress": 95, "message": "Finalizing..."}
        )
        
        # Update session with response
        self.session_manager.update_session(session_id, {
            "message_count": len(response.get('messages', [])),
            "history": response.get('history', ''),
            "last_node": last_node
        })
        
        # Store response for retrieval
        self.session_manager.set_response(session_id, message, audio_base64, last_node)
        self.session_manager.set_status(session_id, "ai_responded")
        
        # Clear processing flag
        clear_processing_flag(self.redis_client, session_id)
        
        logger.info(f"Interview {session_id} initialized successfully")
        
        return {
            "session_id": session_id,
            "status": "ai_responded",
            "message": message,
            "last_node": last_node,
            "audio_available": audio_base64 is not None
        }
        
    except Exception as e:
        logger.error(f"Error starting interview {session_id}: {e}", exc_info=True)
        
        # Clean up on error
        try:
            self.session_manager.set_status(session_id, "error")
            clear_processing_flag(self.redis_client, session_id)
        except:
            pass
        
        raise


# ============================================================================
# TASK: Process User Response (Complete Pipeline with Transcription)
# ============================================================================

@celery_app.task(
    bind=True,
    base=InterviewTask,
    name="tasks.interview_tasks.process_user_response_with_transcription",
    max_retries=3,
    default_retry_delay=5
)
def process_user_response_with_transcription(
    self,
    session_id: str,
    audio_data: Optional[str],
    text_response: Optional[str],
    code_input: Optional[str]
) -> Dict[str, Any]:
    """
    Complete response pipeline: transcribe → process → generate audio.
    
    This is the NEW OPTIMIZED task that combines everything:
    - Transcription (if audio provided)
    - Workflow processing
    - TTS audio generation
    
    Args:
        session_id: Session identifier
        audio_data: Base64 encoded audio (optional)
        text_response: Text response (optional)
        code_input: Code submission (optional)
        
    Returns:
        dict: AI response with audio
    """
    try:
        logger.info(f"Processing complete response pipeline for session {session_id}")
        
        # Get session
        session = self.session_manager.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        interview_type = session["interview_type"]
        
        # Step 1: Get human input (transcribe if needed) - 25% progress
        self.update_state(
            state="PROGRESS",
            meta={"progress": 10, "message": "Processing input..."}
        )
        
        human_input = text_response
        
        if audio_data and not text_response:
            # Transcribe audio
            self.update_state(
                state="PROGRESS",
                meta={"progress": 25, "message": "Transcribing audio..."}
            )
            
            logger.info(f"Transcribing audio for session {session_id}")
            
            try:
                transcription = self.audio_processor.transcribe_audio(audio_data)
                human_input = transcription.strip()
                
                # Store transcript in session
                self.session_manager.set_transcript(session_id, human_input)
                
                logger.info(f"Transcribed ({len(human_input)} chars): '{human_input[:100]}...'")
                
            except Exception as e:
                logger.error(f"Transcription failed for {session_id}: {e}")
                clear_processing_flag(self.redis_client, session_id)
                raise Exception(f"Transcription failed: {str(e)}")
        
        if not human_input:
            clear_processing_flag(self.redis_client, session_id)
            raise ValueError("No input received - empty audio or text")
        
        # Add code input if provided
        if code_input:
            human_input += f"\n\n[CODE INPUT]\n{code_input}"
        
        # Step 2: Process through workflow - 50% progress
        self.update_state(
            state="PROGRESS",
            meta={"progress": 50, "message": "Processing response..."}
        )
        
        logger.info(f"Processing workflow for session {session_id}")
        
        # Get workflow and current state
        agent = get_interview_agent()
        workflow = agent.get_graph(interview_type)
        config = agent.config_for_session(session_id)
        interrupt_nodes = agent.get_interrupt_nodes(interview_type)
        
        current_state = workflow.get_state(config)
        
        # Check if interview is finished
        if not len(current_state.next):
            logger.info(f"Interview {session_id} is finished")
            handle_interview_completion(self.redis_client, self.session_manager, session_id)
            
            return {
                "status": "completed",
                "session_id": session_id,
                "message": "",
                "last_node": "finished"
            }
        
        # Update state with user input using module-level function
        update_workflow_state(workflow, config, interview_type, current_state, human_input)
        
        # Invoke workflow
        self.update_state(
            state="PROGRESS",
            meta={"progress": 60, "message": "Generating AI response..."}
        )
        
        response = workflow.invoke(None, config=config, interrupt_before=interrupt_nodes)
        
        # Extract AI message
        message = response['messages'][-1].content if response.get('messages') else ""
        last_node = response.get('LastNode', '')
        
        if not message:
            clear_processing_flag(self.redis_client, session_id)
            raise ValueError("No AI message generated")
        
        logger.info(f"AI response generated for {session_id}: '{message[:100]}...'")
        
        # Step 3: Generate TTS audio (80% progress)
        self.update_state(
            state="PROGRESS",
            meta={"progress": 80, "message": "Generating audio..."}
        )
        
        audio_base64 = None
        try:
            logger.info(f"Synthesizing audio for session {session_id}")
            audio_base64 = self.audio_processor.synthesize_speech_base64(message)
            logger.info(f"Audio synthesis successful")
        except Exception as e:
            logger.error(f"Audio synthesis failed: {e}")
            # Continue without audio
        
        # Step 4: Store response in Redis (95% progress)
        self.update_state(
            state="PROGRESS",
            meta={"progress": 95, "message": "Storing response..."}
        )
        
        self.session_manager.update_session(session_id, {
            "message_count": len(response.get('messages', [])),
            "history": response.get('history', ''),
            "last_node": last_node
        })
        
        self.session_manager.set_response(
            session_id,
            message,
            audio_base64,
            last_node
        )
        
        self.session_manager.set_status(session_id, "ai_responded")
        
        # Clear processing flag
        clear_processing_flag(self.redis_client, session_id)
        
        logger.info(f"Response processed successfully for session {session_id}")
        
        return {
            "status": "completed",
            "session_id": session_id,
            "message": message,
            "last_node": last_node,
            "audio_available": audio_base64 is not None
        }
        
    except Exception as e:
        logger.error(f"Error processing response: {e}", exc_info=True)
        
        # Clear processing flag on error
        clear_processing_flag(self.redis_client, session_id)
        
        try:
            self.session_manager.set_status(session_id, "error")
        except:
            pass
        
        raise


# ============================================================================
# TASK: Legacy Process User Response (Keep for backward compatibility)
# ============================================================================

@celery_app.task(
    bind=True,
    base=InterviewTask,
    name="tasks.interview_tasks.process_user_response"
)
def process_user_response(self, session_id: str, human_input: str) -> Dict[str, Any]:
    """
    Legacy task: Process user response with pre-transcribed input.
    Kept for backward compatibility.
    
    For new implementations, use process_user_response_with_transcription instead.
    """
    return process_user_response_with_transcription(
        self,
        session_id=session_id,
        audio_data=None,
        text_response=human_input,
        code_input=None
    )


# ============================================================================
# TASK: Cleanup Expired Sessions
# ============================================================================

@celery_app.task(bind=True, base=InterviewTask, name="tasks.interview_tasks.cleanup_expired_sessions")
def cleanup_expired_sessions(self):
    """Periodic task to cleanup expired sessions"""
    try:
        logger.info("Running session cleanup")
        # Redis TTL handles expiration automatically
        # Additional cleanup logic can be added here if needed
        return {"status": "completed", "message": "Cleanup completed"}
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        raise
