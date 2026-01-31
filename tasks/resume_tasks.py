"""
Celery tasks for resume analysis
"""
from celery import Task
from tasks.celery_app import celery_app
from typing import Dict, Any
import logging
import os
from redis import Redis
import base64
import tempfile

from workflows.feedback.resume_analysis import build_resume_analysis_graph, State
from langchain_core.messages import HumanMessage
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
from pathlib import Path

logger = logging.getLogger(__name__)


class ResumeTask(Task):
    """Base task with shared resources"""
    _redis_client = None
    
    @property
    def redis_client(self):
        if self._redis_client is None:
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
            self._redis_client = Redis.from_url(redis_url, decode_responses=True)
        return self._redis_client


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Extract text from PDF using OCR
    
    Args:
        pdf_bytes: PDF file bytes
        
    Returns:
        str: Extracted text
    """
    try:
        text = ""
        pages = convert_from_bytes(pdf_bytes)
        
        for page_number, page in enumerate(pages, start=1):
            logger.info(f"Processing page {page_number}")
            page_text = pytesseract.image_to_string(page)
            text += f"Page {page_number}:\n{page_text}\n"
        
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        raise


def extract_text_from_image(image_bytes: bytes) -> str:
    """
    Extract text from image using OCR
    
    Args:
        image_bytes: Image file bytes
        
    Returns:
        str: Extracted text
    """
    try:
        import io
        image = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        logger.error(f"Error extracting text from image: {e}")
        raise


@celery_app.task(bind=True, base=ResumeTask, name="tasks.resume_tasks.extract_text_from_file")
def extract_text_from_file(self, file_bytes_b64: str, filename: str) -> Dict[str, Any]:
    """
    Extract text from uploaded file (PDF or image)
    
    Args:
        file_bytes_b64: Base64 encoded file bytes
        filename: Original filename
        
    Returns:
        dict: Extracted text
    """
    try:
        logger.info(f"Extracting text from file: {filename}")
        
        # Decode file
        file_bytes = base64.b64decode(file_bytes_b64)
        
        # Determine file type
        file_ext = Path(filename).suffix.lower()
        
        if file_ext == ".pdf":
            text = extract_text_from_pdf(file_bytes)
        elif file_ext in [".png", ".jpg", ".jpeg", ".webp"]:
            text = extract_text_from_image(file_bytes)
        elif file_ext == ".txt":
            # Plain text file - just decode as UTF-8
            text = file_bytes.decode('utf-8')
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
        
        logger.info(f"Text extraction completed (length: {len(text)})")
        
        return {
            "status": "success",
            "text": text,
            "length": len(text)
        }
        
    except Exception as e:
        logger.error(f"Error in extract_text_from_file: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "text": ""
        }


def analyze_resume(
    parent_task,
    session_id: str,
    resume_text: str,
    job_description: str,
    user_id: str,
    resume_filename: str = "resume.pdf"
) -> Dict[str, Any]:
    """
    Analyze resume against job description using LangGraph
    
    Args:
        task_id: Unique task identifier
        resume_text: Extracted resume text
        job_description: Job description text
        user_id: User ID
        
    Returns:
        dict: Analysis results
    """
    try:
        logger.info(f"Starting resume analysis for session {session_id}")
        
        # Update progress
        parent_task.update_state(
            state='PROGRESS',
            meta={'progress': 30, 'status': 'Analyzing resume structure...'}
        )
        
        # Build analysis graph
        google_key = os.getenv("GOOGLE_API_KEY", "")
        graph = build_resume_analysis_graph(google_key)
        
        # Prepare input
        input_message = HumanMessage(content=f"""
Resume:
{resume_text}

Job Description:
{job_description}

Please analyze this resume against the job description.
""")
        
        # Update progress
        parent_task.update_state(
            state='PROGRESS',
            meta={'progress': 50, 'status': 'Running analysis...'}
        )
        
        # Run analysis
        result = graph.invoke({
            "input_message": [input_message],
            "job_description": job_description
        })
        
        # Update progress
        parent_task.update_state(
            state='PROGRESS',
            meta={'progress': 80, 'status': 'Compiling results...'}
        )
        
        # Extract results - match old Django serializer format exactly
        job_match_score = result["section_analysis"].job_match_score
        format_and_structure = result["section_analysis"].format_and_structure
        content_quality = result["section_analysis"].content_quality
        length_and_conciseness = result["section_analysis"].length_and_conciseness
        keywords_optimization = result["section_analysis"].keywords_optimization
        
        # Calculate overall_score as average of all section scores (matching old behavior)
        overall_score = round(
            (job_match_score + format_and_structure + content_quality + 
             length_and_conciseness + keywords_optimization) / 5.0, 
            2
        )
        
        analysis_result = {
            # Basic info
            "session_id": session_id,
            "resume_name": resume_filename,
            "company": result.get("company", ""),
            "role": result.get("role", ""),
            
            # Scores
            "overall_score": overall_score,
            "job_match_score": job_match_score,
            
            # Section analysis - keep nested for internal use
            "section_analysis": {
                "job_match_score": job_match_score,
                "format_and_structure": format_and_structure,
                "content_quality": content_quality,
                "length_and_conciseness": length_and_conciseness,
                "keywords_optimization": keywords_optimization,
            },
            
            # Sections array - matching old serializer format exactly
            "sections": [
                {"name": "job_match_score", "score": job_match_score},
                {"name": "format_and_structure", "score": format_and_structure},
                {"name": "content_quality", "score": content_quality},
                {"name": "length_and_conciseness", "score": length_and_conciseness},
                {"name": "keywords_optimization", "score": keywords_optimization}
            ],
            
            # Keyword analysis
            "keyword_analysis": {
                "found_keywords": result["keyword_analysis"].found_keywords,
                "not_found_keywords": result["keyword_analysis"].not_found_keywords,
                "top_3_keywords": result["keyword_analysis"].top_3_keywords,
            },
            
            # Keywords object - matching old serializer format
            "keywords": {
                "found": result["keyword_analysis"].found_keywords,
                "missing": result["keyword_analysis"].not_found_keywords,
                "jobSpecific": result["keyword_analysis"].top_3_keywords,
                "score": keywords_optimization
            },
            
            # Job alignment - matching old serializer format
            "job_alignment": {
                "required_skills": result["job_alignment_analysis"].required_skills,
                "preferred_skills": result["job_alignment_analysis"].preferred_skills,
                "experience": result["job_alignment_analysis"].experience,
                "education": result["job_alignment_analysis"].education,
                "overall": job_match_score,  # Add overall field matching old format
                "insights": result["job_alignment_analysis"].insights,
            },
            
            # Job alignment alias for compatibility
            "jobalignment": {
                "requiredSkills": result["job_alignment_analysis"].required_skills,
                "preferredSkills": result["job_alignment_analysis"].preferred_skills,
                "experience": result["job_alignment_analysis"].experience,
                "education": result["job_alignment_analysis"].education,
                "overall": job_match_score,
                "insights": result["job_alignment_analysis"].insights,
            },
            
            # Strengths and improvements
            "strengths_and_improvements": {
                "candidate_strengths": result["strengths_and_improvements"].candidate_strengths,
                "candidates_areas_of_improvements": result["strengths_and_improvements"].candidates_areas_of_improvements,
            },
            
            # Direct fields for frontend compatibility
            "candidate_strengths": result["strengths_and_improvements"].candidate_strengths,
            "candidates_areas_of_improvements": result["strengths_and_improvements"].candidates_areas_of_improvements,
            "insights": result["job_alignment_analysis"].insights,
        }
        
        # Store in Redis (use parent task's redis client)
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        redis_client = Redis.from_url(redis_url, decode_responses=True)
        redis_key = f"resume_analysis:{session_id}"
        import json
        redis_client.setex(redis_key, 3600, json.dumps(analysis_result))
        
        # Save to Django database
        try:
            from services.django_db import save_resume_analysis_to_db
            db_saved = save_resume_analysis_to_db(
                user_id=user_id,
                session_id=session_id,
                analysis_result=analysis_result
            )
            if db_saved:
                logger.info(f"Resume analysis saved to Django database for session {session_id}")
            else:
                logger.warning(f"Failed to save resume analysis to Django database for session {session_id}")
        except Exception as e:
            logger.error(f"Error saving to Django database: {e}", exc_info=True)
            # Don't fail the task if DB save fails, Redis storage is sufficient for immediate access
        
        # Update progress
        parent_task.update_state(
            state='PROGRESS',
            meta={'progress': 95, 'status': 'Analysis completed'}
        )
        
        logger.info(f"Resume analysis completed for session {session_id}")
        
        return {
            "status": "completed",
            "result": analysis_result
        }
        
    except Exception as e:
        logger.error(f"Error in analyze_resume: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "result": None
        }


@celery_app.task(bind=True, base=ResumeTask, name="tasks.resume_tasks.process_resume_upload")
def process_resume_upload(
    self,
    task_id: str,
    resume_bytes_b64: str,
    resume_filename: str,
    job_desc_bytes_b64: str,
    job_desc_filename: str,
    user_id: str,
    session_id: str = None
) -> Dict[str, Any]:
    """
    Process resume upload: extract text and analyze
    
    Args:
    
        task_id: Task identifier
        resume_bytes_b64: Base64 encoded resume file
        resume_filename: Resume filename
        job_desc_bytes_b64: Base64 encoded job description
        job_desc_filename: Job description filename
        user_id: User ID
        
    Returns:
        dict: Analysis results
    """
    try:
        # Use the actual Celery task ID
        actual_task_id = self.request.id
        logger.info(f"Processing resume upload for task {actual_task_id}")
        
        # Extract resume text
        self.update_state(
            state='PROGRESS',
            meta={'progress': 10, 'status': 'Extracting resume text...'}
        )
        
        resume_result = extract_text_from_file(resume_bytes_b64, resume_filename)
        if resume_result["status"] == "error":
            return resume_result
        
        resume_text = resume_result["text"]
        
        # Extract job description text
        self.update_state(
            state='PROGRESS',
            meta={'progress': 20, 'status': 'Extracting job description...'}
        )
        
        job_desc_result = extract_text_from_file(job_desc_bytes_b64, job_desc_filename)
        if job_desc_result["status"] == "error":
            return job_desc_result
        
        job_description = job_desc_result["text"]
        
        # Analyze resume (pass self as parent_task for progress updates)
        # Pass session_id if provided, otherwise use task_id
        session_id_to_use = session_id if session_id else actual_task_id
        analysis_result = analyze_resume(
            self, 
            session_id_to_use, 
            resume_text, 
            job_description, 
            user_id,
            resume_filename
        )
        
        return analysis_result
        
    except Exception as e:
        logger.error(f"Error processing resume upload: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "result": None
        }
