

"""
Celery tasks for interview processing - OPTIMIZED VERSION (FIXED)
"""
from celery import Task
from tasks.celery_app import celery_app
from typing import Dict, Any, List, Optional
import logging
import json
from redis import Redis
import os
from datetime import datetime, timedelta

from services.interview_session import InterviewSessionManager
from services.interview_agent import get_interview_agent
from workflows.technical import TechnicalInterviewState
from workflows.hr import HRInterviewState
from workflows.coding import SubjectInterviewState
from workflows.companybuilder import CompanyInterviewState as CompanyInterviewStateBuilder
from workflows.coding import CompanyInterviewState, SubjectInterviewState, Questions as CodingQuestions
from workflows.case_study import CaseStudyInterviewState
from workflows.communication import CommunicationInterviewState
from workflows.rolebased import RoleBasedInterviewState
from workflows.debate import DebateInterviewState
from workflows.aiml import AimlInterviewState
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

def _normalize_interview_test_id(value: Any) -> Optional[Any]:
    """
    Return interview_test_id as int only when it is numeric; otherwise keep as str (e.g. UUID).
    Prevents ValueError when the client sends a UUID string and we were previously calling int() on it.
    """
    if value is None:
        return None
    if isinstance(value, int):
        return value
    s = str(value).strip()
    if not s:
        return None
    if s.isdigit():
        return int(s)
    return value  # UUID or other non-numeric id: pass through as-is


def _map_interview_questions_from_drf(data: Dict[str, Any]) -> list:
    """
    Map DRF interview-questions response to list of question dicts for workflow state.
    Expects theoretical_questions (3) and coding_questions (2). Returns 5 items in order: theoretical first, then coding.
    """
    out = []
    for t in (data.get("theoretical_questions") or [])[:3]:
        if isinstance(t, dict):
            out.append({
                "question_title": t.get("subject") or "Theory",
                "question_description": t.get("question") or "",
                "question_raw_content": "",
                "question_difficulty": "Medium",
                "question_type": "theoretical",
            })
        else:
            out.append({
                "question_title": getattr(t, "subject", None) or "Theory",
                "question_description": getattr(t, "question", None) or "",
                "question_raw_content": "",
                "question_difficulty": "Medium",
                "question_type": "theoretical",
            })
    for c in (data.get("coding_questions") or [])[:2]:
        if isinstance(c, dict):
            out.append({
                "question_title": c.get("title") or "Coding",
                "question_description": c.get("description") or "",
                "question_raw_content": c.get("raw_content") or "",
                "question_difficulty": (c.get("difficulty") or "Medium").strip(),
                "question_type": "coding",
            })
        else:
            out.append({
                "question_title": getattr(c, "title", None) or "Coding",
                "question_description": getattr(c, "description", None) or "",
                "question_raw_content": getattr(c, "raw_content", None) or "",
                "question_difficulty": (getattr(c, "difficulty", None) or "Medium").strip(),
                "question_type": "coding",
            })
    return out


def _serialize_questions_for_payload(questions: List[Any]) -> List[Dict[str, Any]]:
    """
    Convert workflow state 'Questions' (Pydantic models or dicts) to a list of
    plain dicts for storing in session payload and sending to frontend.
    Includes both DRF-loaded questions and any generated by Initial_research
    (e.g. theoretical questions from create_questions_node in coding.py).
    """
    if not questions:
        return []
    out = []
    for q in questions:
        if isinstance(q, dict):
            out.append({
                "question_title": q.get("question_title", ""),
                "question_description": q.get("question_description", ""),
                "question_difficulty": q.get("question_difficulty", "Medium"),
                "question_type": q.get("question_type", "theoretical"),
                "question_raw_content": q.get("question_raw_content") or q.get("raw_content") or "",
            })
        elif hasattr(q, "model_dump"):
            out.append(q.model_dump())
        elif hasattr(q, "dict"):
            out.append(q.dict())
        else:
            out.append({
                "question_title": getattr(q, "question_title", "") or "",
                "question_description": getattr(q, "question_description", "") or "",
                "question_difficulty": getattr(q, "question_difficulty", "Medium") or "Medium",
                "question_type": getattr(q, "question_type", "theoretical") or "theoretical",
                "question_raw_content": getattr(q, "question_raw_content", None) or getattr(q, "raw_content", None) or "",
            })
    return out


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
        tags = payload.get("Tags", [])
        tags_str = ", ".join(str(t) for t in tags) if isinstance(tags, list) else (tags or " ")
        company = (payload.get("company") or payload.get("Company") or "").strip()
        raw_questions = payload.get("Questions") or []
        questions_list = [
            CodingQuestions(**q) if isinstance(q, dict) else q
            for q in raw_questions
        ]
        return CompanyInterviewState(
            LastNode="default",
            company=company or "the company",
            QuestionResearch=payload.get("QuestionResearch", ""),
            history="",
            Questions=questions_list,
            CurrentQuestionIdx=0,
            Difficulty=payload.get("Difficulty", "Medium"),
            Tags=tags_str,
            resume=payload.get("resume", "No resume provided"),
        )
    elif interview_type == "Subject":
        subject = (payload.get("subject") or payload.get("Subject") or "").strip()
        tags = payload.get("Tags", [])
        tags_str = ", ".join(str(t) for t in tags) if isinstance(tags, list) else (tags or " ")
        raw_questions = payload.get("Questions") or []
        questions_list = [
            CodingQuestions(**q) if isinstance(q, dict) else q
            for q in raw_questions
        ]
        return SubjectInterviewState(
            LastNode="default",
            subject=subject or "Arrays",
            QuestionResearch=payload.get("QuestionResearch", ""),
            history="",
            Questions=questions_list,
            CurrentQuestionIdx=0,
            Difficulty=payload.get("Difficulty", "Medium"),
            Tags=tags_str,
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
    elif interview_type == "Communication":  # ADD THIS BLOCK
        return CommunicationInterviewState(
            messages=[],
            LastNode="",
            history="",
            current_query="",
            mcq_questions_asked=[],
            pending_mcq_answer="",
            set_timer_on=False
        )
    elif interview_type == "Role-Based Interview":
        return RoleBasedInterviewState(
            LastNode="",
            history="",
            resume=payload.get("resume", "No resume provided"),
            role=payload.get("role", "Frontend Development"),
            messages=[],
        )
    elif interview_type == "Debate":
        return DebateInterviewState(
            LastNode="",
            history="",
            messages=[],
            rounds_completed=0,
        )
    elif interview_type == "AIML":
        # interview_topic maps to the InterviewTest title (e.g. "Transformers & Attention"),
        # which is embedded in the access token JWT as `title` and forwarded in payload.
        return AimlInterviewState(
            LastNode="",
            history="",
            interview_topic=payload.get("title", "AI/ML"),
            TopicResearch="",
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
    elif interview_type == "Communication":  # ADD THIS BLOCK
        # For Communication, handle phase-specific logic
        human_message = HumanMessage(content=human_input)
        current_messages = current_state.values.get("messages", [])
        updated_messages = current_messages + [human_message]
        
        last_node = current_state.values.get("LastNode", "")
        
        # Store MCQ answer if in MCQ phase
        if last_node in ("MCQ", "MCQ_after"):
            workflow.update_state(config, {
                "messages": updated_messages,
                "current_query": human_input,
                "pending_mcq_answer": human_input,
                "history": current_state.values.get("history", "") + "\nInterviewee-" + human_input
            })
        else:
            workflow.update_state(config, {
                "messages": updated_messages,
                "current_query": human_input,
                "history": current_state.values.get("history", "") + "\nInterviewee-" + human_input
            })
    else:
        # For Technical, HR, Company, Subject, Role-Based: append HumanMessage so state matches graph expectations
        human_message = HumanMessage(content=human_input)
        current_messages = current_state.values.get("messages", [])
        updated_messages = current_messages + [human_message]
        workflow.update_state(config, {
            "messages": updated_messages,
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
        
        # Store interview_test_id at top level (int or str/UUID) so feedback task always has it
        interview_test_id_from_payload = payload.get("interview_test_id") or payload.get("interview_type_id")
        if interview_test_id_from_payload is not None:
            tid = _normalize_interview_test_id(interview_test_id_from_payload)
            if tid is not None:
                self.session_manager.update_session(session_id, {"interview_test_id": tid})
                logger.info(f"Session {session_id}: stored interview_test_id={tid} from payload")
        
        # Set processing flag (expires in 60s as safety)
        self.redis_client.setex(processing_key, 60, "true")
        self.session_manager.set_status(session_id, "processing")
        
        logger.info(f"Session {session_id} created and marked as processing")
        
        # Subject interviews: fetch research questions and subject/topic name from DRF by interview_test_id
        if interview_type == "Subject":
            interview_test_id = payload.get("interview_test_id") or payload.get("interview_type_id")
            if interview_test_id is not None:
                try:
                    from services.drf_client import get_research_questions_for_subject
                    tid = _normalize_interview_test_id(interview_test_id)
                    research = get_research_questions_for_subject(interview_test_id=tid) if tid is not None else None
                    if research is not None:
                        qr = None
                        if isinstance(research, str):
                            qr = research
                        elif isinstance(research, dict):
                            raw = (
                                research.get("QuestionResearch")
                                or research.get("research_questions")
                                or research.get("research")
                            )
                            if isinstance(raw, list):
                                qr = "\n".join(str(x) for x in raw)
                            else:
                                qr = str(raw) if raw is not None else None
                            if qr is None:
                                qr = str(research)
                            if qr:
                                payload = {**payload, "QuestionResearch": qr}
                            # DRF returns "topic" (e.g. "Arrays") - set subject so greeting says the actual topic
                            topic = research.get("topic") or research.get("subject")
                            if topic:
                                payload = {**payload, "subject": str(topic).strip()}
                                logger.info(f"Subject interview: set subject from DRF topic={topic}")
                        elif isinstance(research, list):
                            qr = "\n".join(str(x) for x in research)
                            if qr:
                                payload = {**payload, "QuestionResearch": qr}
                        else:
                            qr = str(research)
                            if qr:
                                payload = {**payload, "QuestionResearch": qr}
                        if payload.get("QuestionResearch"):
                            logger.info(f"Subject interview: loaded research for interview_test_id={interview_test_id}")
                except Exception as e:
                    logger.warning(f"Could not fetch research questions for Subject interview: {e}")
            if not (payload.get("subject") or "").strip():
                payload = {**payload, "subject": (payload.get("Subject") or "").strip() or "Arrays"}
                logger.info(f"Subject interview: using payload.Subject fallback={payload.get('subject')}")
        
        # Company interviews: prefer frontend-sent Company (e.g. Flipkart) so greeting always says the selected name
        if interview_type == "Company":
            frontend_company = (payload.get("Company") or payload.get("company") or "").strip()
            interview_test_id = payload.get("interview_test_id") or payload.get("interview_type_id")
            drf_company = None
            if interview_test_id is not None:
                try:
                    from services.drf_client import get_company_for_interview
                    tid = _normalize_interview_test_id(interview_test_id)
                    result = get_company_for_interview(interview_type_id=tid) if tid is not None else None
                    logger.info(f"Company interview: result={result} type={type(result)}")
                    if result is not None and isinstance(result, dict) and result.get("company"):
                        drf_company = (result["company"] or "").strip()
                        logger.info(f"Company interview: DRF company for id={interview_test_id}: {drf_company}")
                except Exception as e:
                    logger.warning(f"Could not fetch company for Company interview: {e}")
            # Use frontend name first (user selected e.g. Flipkart), then DRF, then fallback
            company = frontend_company or drf_company or "the company"
            payload = {**payload, "company": company}
            logger.info(f"Company interview: using company={company} (frontend={frontend_company!r}, drf={drf_company!r})")

        # Company and Subject: fetch 5 questions (3 theoretical + 2 coding) from DRF
        if interview_type in ("Company", "Subject"):
            interview_test_id = payload.get("interview_test_id") or payload.get("interview_type_id")
            if interview_test_id is not None:
                try:
                    from services.drf_client import get_interview_questions
                    tid = _normalize_interview_test_id(interview_test_id)
                    questions_response = get_interview_questions(interview_type_id=tid) if tid is not None else None
                    if questions_response and isinstance(questions_response, dict):
                        questions_list = _map_interview_questions_from_drf(questions_response)
                        if questions_list:
                            payload = {**payload, "Questions": questions_list}
                            self.session_manager.update_session(session_id, {"payload": payload})
                            logger.info(f"{interview_type} interview: loaded {len(questions_list)} questions for interview_type_id={interview_test_id}")
                except Exception as e:
                    logger.warning(f"Could not fetch interview questions for {interview_type}: {e}")
        
        # Step 2: Initialize workflow (40% progress)
        self.update_state(
            state="PROGRESS",
            meta={"progress": 40, "message": "Initializing workflow..."}
        )
        
        # Get singleton interview agent
        agent = get_interview_agent()
        if interview_type == "Role-Based Interview":
            role = payload.get("role", "Frontend Development")
            workflow = agent.get_graph(interview_type, role=role)
        else:
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

        # Company/Subject (and Coding): store question progress and full Questions list for frontend
        # (includes Initial_research-generated theoretical questions from coding.py)
        question_number, total_questions = None, None
        question_raw_content = None
        questions = response.get("Questions") or []
        current_idx = response.get("CurrentQuestionIdx", 0)
        if current_idx < len(questions):
            q = questions[current_idx]
            if isinstance(q, dict):
                question_raw_content = q.get("question_raw_content") or q.get("raw_content") or ""
            else:
                question_raw_content = getattr(q, "question_raw_content", None) or getattr(q, "raw_content", None) or ""
        if interview_type in ("Company", "Subject"):
            total_questions = len(questions) if questions else None
            question_number = (current_idx + 1) if total_questions else None
            # Persist enriched Questions (DRF + generated theoretical) to session payload
            # so API can return interview_questions to frontend
            if questions:
                serialized = _serialize_questions_for_payload(questions)
                session = self.session_manager.get_session(session_id)
                payload = (session or {}).get("payload") or {}
                payload = {**payload, "Questions": serialized}
                self.session_manager.update_session(session_id, {"payload": payload})
                logger.info(f"Updated session payload with {len(serialized)} questions (incl. initial research)")

        # Store response for retrieval (include question_raw_content for coding/company/subject frontend)
        self.session_manager.set_response(
            session_id, message, audio_base64, last_node,
            question_number=question_number,
            total_questions=total_questions,
            question_raw_content=question_raw_content or None,
        )
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
    code_input: Optional[str],
    skip_audio: bool = False
) -> Dict[str, Any]:
    """
    Complete response pipeline: transcribe → process → generate audio.
    
    This is the NEW OPTIMIZED task that combines everything:
    - Transcription (if audio provided)
    - Workflow processing
    - TTS audio generation (skipped if skip_audio=True for dev mode)
    
    Args:
        session_id: Session identifier
        audio_data: Base64 encoded audio (optional)
        text_response: Text response (optional)
        code_input: Code submission (optional)
        skip_audio: Skip TTS generation for dev mode (default: False)
        
    Returns:
        dict: AI response with audio (audio_base64 is None if skip_audio=True)
    """
    try:
        logger.info(f"Processing complete response pipeline for session {session_id}")
        
        # Get session
        session = self.session_manager.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        interview_type = session["interview_type"]
        payload_from_session = session.get("payload") or {}

        # Step 1: Get human input (transcribe if needed) - 25% progress
        self.update_state(
            state="PROGRESS",
            meta={"progress": 10, "message": "Processing input..."}
        )
        
        human_input = (text_response or "").strip() or None
        
        if audio_data and not human_input:
            # Transcribe audio (speaking phase)
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
        elif human_input:
            # Text-only (e.g. Communication comprehension phase): use as-is and store as transcript
            self.session_manager.set_transcript(session_id, human_input)
            logger.info(f"Text-only response for session {session_id} ({len(human_input)} chars)")

        code_text = (code_input or "").strip() or None
        if not human_input and code_text:
            # Code-only (coding tab): must be valid without separate text/audio
            human_input = f"[CODE INPUT]\n{code_text}"
            self.session_manager.set_transcript(session_id, human_input)
            logger.info(f"Code-only response for session {session_id} ({len(code_text)} chars)")
        elif human_input and code_text:
            human_input += f"\n\n[CODE INPUT]\n{code_text}"

        if not human_input:
            clear_processing_flag(self.redis_client, session_id)
            raise ValueError("No input received - empty audio, text, or code")
        
        # Step 2: Process through workflow - 50% progress
        self.update_state(
            state="PROGRESS",
            meta={"progress": 50, "message": "Processing response..."}
        )
        
        logger.info(f"Processing workflow for session {session_id}")
        
        # Get workflow and current state
        agent = get_interview_agent()
        if interview_type == "Role-Based Interview":
            role = payload_from_session.get("role", "Frontend Development")
            workflow = agent.get_graph(interview_type, role=role)
        else:
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

        current_speaking = None
        speaking_feedback = None
        current_comprehension = None
        comprehension_feedback = None
        current_mcq = None
        mcq_feedback = None

        logger.info(f"Interview type: {interview_type}")
        if interview_type == "Communication":
             
            # Extract speaking data
            if last_node == "Speaking" and response.get('current_speaking'):
                logger.info(f"Speaking interview phase and custom response")
                speaking_data = response['current_speaking']
                current_speaking = {
                    "instruction": speaking_data.instruction if hasattr(speaking_data, 'instruction') else None,
                    "paragraph": speaking_data.paragraph if hasattr(speaking_data, 'paragraph') else None,
                }
                logger.info(f"Extracted speaking data for session {session_id}")
            
            # Extract speaking feedback
            if last_node in ("Speaking_feedback", "Speaking_feedback_after"):
                speaking_feedback = message  # The feedback is in the message
                logger.info(f"Extracted speaking feedback for session {session_id}")

            # Extract comprehension instruction and question (so frontend can show writing prompt)
            if last_node in ("Comprehension", "Comprehension_after") and response.get('current_writing_comprehension'):
                comp_data = response['current_writing_comprehension']
                if isinstance(comp_data, dict):
                    current_comprehension = {
                        "instruction": comp_data.get("instruction"),
                        "question": comp_data.get("question"),
                    }
                else:
                    current_comprehension = {
                        "instruction": getattr(comp_data, 'instruction', None),
                        "question": getattr(comp_data, 'question', None),
                    }
                logger.info(f"Extracted comprehension data for session {session_id}")

            # Extract comprehension feedback
            if last_node in ("Comprehension_feedback", "Comprehension_feedback_after"):
                comprehension_feedback = message  # The feedback is in the message
                logger.info(f"Extracted comprehension feedback for session {session_id}")

            # Extract MCQ entity (question, options, answer)
            if last_node in ("MCQ", "MCQ_after") and response.get('current_mcq_entity'):
                mcq_data = response['current_mcq_entity']
                if isinstance(mcq_data, dict):
                    current_mcq = {
                        "instruction": mcq_data.get("instruction"),
                        "question": mcq_data.get("question"),
                        "options": mcq_data.get("options"),
                        "answer": mcq_data.get("answer"),
                    }
                else:
                    current_mcq = {
                        "instruction": getattr(mcq_data, 'instruction', None),
                        "question": getattr(mcq_data, 'question', None),
                        "options": getattr(mcq_data, 'options', None),
                        "answer": getattr(mcq_data, 'answer', None),
                    }
                logger.info(f"Extracted MCQ data for session {session_id}")

            # Extract MCQ feedback (at End node, the final summary/feedback is in the message)
            if last_node == "End":
                mcq_feedback = message  # Final feedback/wrap-up message
                logger.info(f"Extracted MCQ/final feedback for session {session_id}")
        
        if not message:
            clear_processing_flag(self.redis_client, session_id)
            raise ValueError("No AI message generated")
        
        logger.info(f"AI response generated for {session_id}: '{message[:100]}...'")
        
        # Step 3: Generate TTS audio (80% progress) - Skip if dev mode
        audio_base64 = None
        if not skip_audio:
            self.update_state(
                state="PROGRESS",
                meta={"progress": 80, "message": "Generating audio..."}
            )
            
            try:
                logger.info(f"Synthesizing audio for session {session_id}")
                audio_base64 = self.audio_processor.synthesize_speech_base64(message)
                logger.info(f"Audio synthesis successful")
            except Exception as e:
                logger.error(f"Audio synthesis failed: {e}")
                # Continue without audio
        else:
            logger.info(f"[DEV MODE] Skipping audio generation for session {session_id}")
        
        # Step 4: Store response in Redis (95% progress)
        self.update_state(
            state="PROGRESS",
            meta={"progress": 95, "message": "Storing response..."}
        )

        if interview_type == "Communication":
            phase_data = {}
            
            if current_speaking:
                phase_data['current_speaking'] = current_speaking
            
            if speaking_feedback:
                phase_data['speaking_feedback'] = speaking_feedback

            if current_comprehension:
                phase_data['current_comprehension'] = current_comprehension

            if comprehension_feedback:
                phase_data['comprehension_feedback'] = comprehension_feedback

            if current_mcq:
                phase_data['current_mcq'] = current_mcq

            if mcq_feedback:
                phase_data['mcq_feedback'] = mcq_feedback
            
            if phase_data:
                self.session_manager.update_session(session_id, phase_data)  # ← This stores it
                logger.info(f"Stored Communication phase data: {list(phase_data.keys())}")
        
        self.session_manager.update_session(session_id, {
            "message_count": len(response.get('messages', [])),
            "history": response.get('history', ''),
            "last_node": last_node
        })

        # Company/Subject (and Coding): store question progress and keep Questions in sync for frontend
        question_number, total_questions = None, None
        question_raw_content = None
        questions = response.get("Questions") or []
        current_idx = response.get("CurrentQuestionIdx", 0)
        if current_idx < len(questions):
            q = questions[current_idx]
            if isinstance(q, dict):
                question_raw_content = q.get("question_raw_content") or q.get("raw_content") or ""
            else:
                question_raw_content = getattr(q, "question_raw_content", None) or getattr(q, "raw_content", None) or ""
        if interview_type in ("Company", "Subject"):
            total_questions = len(questions) if questions else None
            question_number = (current_idx + 1) if total_questions else None
            # Keep session payload Questions in sync with workflow state (incl. initial research)
            if questions:
                serialized = _serialize_questions_for_payload(questions)
                session = self.session_manager.get_session(session_id)
                payload = (session or {}).get("payload") or {}
                payload = {**payload, "Questions": serialized}
                self.session_manager.update_session(session_id, {"payload": payload})

        self.session_manager.set_response(
            session_id,
            message,
            audio_base64,
            last_node,
            question_number=question_number,
            total_questions=total_questions,
            question_raw_content=question_raw_content or None,
        )
        self.session_manager.set_status(session_id, "ai_responded")
        
        # Must clear so the next POST /respond is not rejected with 429 while TTL is still active.
        clear_processing_flag(self.redis_client, session_id)
        
        logger.info(f"Response processed successfully for session {session_id}")
        
        return {
            "status": "completed",
            "session_id": session_id,
            "message": message,
            "last_node": last_node,
            "audio_available": audio_base64 is not None,
            "interview_ai_response": current_speaking or speaking_feedback or current_comprehension or comprehension_feedback or current_mcq or mcq_feedback
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

