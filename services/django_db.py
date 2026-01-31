"""
Service to interact with Django database models from FastAPI
This module is designed to work when Celery workers run from Django backend venv
"""
import os
import sys
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Try to import Django - it should be available in Celery worker environment
_django_initialized = False
ResumeAnalysis = None
User = None
UserProfile = None
TechnicalFeedback = None
HRFeedback = None
CaseStudyFeedback = None
InterviewTest = None

def _init_django():
    """Initialize Django - call this lazily when needed"""
    global _django_initialized, ResumeAnalysis, User, UserProfile, TechnicalFeedback, HRFeedback, CaseStudyFeedback, InterviewTest
    
    if _django_initialized:
        return True
    
    try:
        import django
        from pathlib import Path
        
        # Add Django project to path
        DJANGO_PROJECT_PATH = Path(__file__).resolve().parent.parent.parent / "interviewsta-app-backend" / "myproject"
        if str(DJANGO_PROJECT_PATH) not in sys.path:
            sys.path.insert(0, str(DJANGO_PROJECT_PATH))
        
        # Set Django settings module
        os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'myproject.settings')
        
        # Initialize Django - django.setup() is safe to call multiple times
        # It checks internally if already configured
        django.setup()
        
        # Import Django models
        from myapp.models import (
            ResumeAnalysis, User, UserProfile,
            TechnicalFeedback, HRFeedback, CaseStudyFeedback, InterviewTest
        )
        _django_initialized = True
        logger.info("Django initialized successfully for database access")
        return True
        
    except ImportError as e:
        logger.warning(f"Django not available in this environment: {e}. Database saving will be skipped.")
        return False
    except Exception as e:
        logger.error(f"Django initialization error: {e}", exc_info=True)
        return False

# Try to initialize Django on module load
_init_django()


def save_resume_analysis_to_db(
    user_id: str,
    session_id: str,
    analysis_result: dict
) -> bool:
    """
    Save resume analysis results to Django database
    
    Args:
        user_id: Firebase user ID (UID)
        session_id: Session ID for the analysis
        analysis_result: Dictionary containing analysis results
        
    Returns:
        bool: True if saved successfully, False otherwise
    """
    global ResumeAnalysis, User, UserProfile
    
    # Check if Django is available, try to initialize if not
    if not _django_initialized:
        if not _init_django():
            logger.warning("Django not initialized, skipping database save")
            return False
    
    # Re-import models if they're None (in case initialization happened but import failed)
    if ResumeAnalysis is None:
        try:
            from myapp.models import ResumeAnalysis, User, UserProfile
        except Exception as e:
            logger.error(f"Failed to import Django models: {e}", exc_info=True)
            return False
    
    try:
        # Get Django User by Firebase UID from UserProfile
        try:
            profile = UserProfile.objects.filter(firebase_uid=user_id).first()
            if not profile:
                logger.warning(f"User profile not found for Firebase UID: {user_id}")
                return False
            user = profile.user
        except Exception as e:
            logger.error(f"Error finding user by Firebase UID: {e}")
            return False
        
        # Extract data from analysis_result to match ResumeAnalysis model
        resume_data = {
            "user": user,
            "session_id": session_id,
            "resume_name": analysis_result.get("resume_name", "Your_Resume.pdf"),
            "company": analysis_result.get("company", ""),
            "role": analysis_result.get("role", ""),
            "job_match_score": analysis_result.get("job_match_score", 0),
            "format_and_structure": analysis_result.get("format_and_structure", 0),
            "content_quality": analysis_result.get("content_quality", 0),
            "length_and_conciseness": analysis_result.get("length_and_conciseness", 0),
            "keywords_optimization": analysis_result.get("keywords_optimization", 0),
            "found_keywords": analysis_result.get("found_keywords", []),
            "not_found_keywords": analysis_result.get("not_found_keywords", []),
            "top_3_keywords": analysis_result.get("top_3_keywords", []),
            "required_skills": analysis_result.get("required_skills", 0),
            "preferred_skills": analysis_result.get("preferred_skills", 0),
            "experience": analysis_result.get("experience", 0),
            "education": analysis_result.get("education", 0),
            "insights": analysis_result.get("insights", []),
            "candidate_strengths": analysis_result.get("candidate_strengths", []),
            "candidates_areas_of_improvements": analysis_result.get("candidates_areas_of_improvements", []),
        }
        
        # Create or update ResumeAnalysis
        resume_analysis, created = ResumeAnalysis.objects.update_or_create(
            session_id=session_id,
            user=user,
            defaults=resume_data
        )
        
        logger.info(f"Resume analysis {'created' if created else 'updated'} in database: session_id={session_id}, user={user_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving resume analysis to database: {e}", exc_info=True)
        return False


def save_feedback_to_db(
    user_id: str,
    session_id: str,
    interview_type: str,
    interview_test_id: int,
    duration_seconds: int,
    feedback_data: dict,
    interaction_log: list,
    soft_skill_summary: dict = None,
    big5_profile: dict = None
) -> bool:
    """
    Save interview feedback to Django database
    
    Args:
        user_id: Firebase user ID (UID)
        session_id: Session ID for the interview
        interview_type: Type of interview (Technical, HR, CaseStudy)
        interview_test_id: InterviewTest ID (can be None)
        duration_seconds: Interview duration in seconds
        feedback_data: Dictionary containing feedback scores and results
        interaction_log: List of interaction history (Q&A pairs)
        soft_skill_summary: Optional soft skills summary dict
        big5_profile: Optional Big-5 profile dict
        
    Returns:
        bool: True if saved successfully, False otherwise
    """
    global TechnicalFeedback, HRFeedback, CaseStudyFeedback, InterviewTest, User, UserProfile
    
    # Check if Django is available
    if not _django_initialized:
        if not _init_django():
            logger.warning("Django not initialized, skipping database save")
            return False
    
    # Re-import models if needed
    if TechnicalFeedback is None:
        try:
            from myapp.models import (
                TechnicalFeedback, HRFeedback, CaseStudyFeedback, InterviewTest, User, UserProfile
            )
        except Exception as e:
            logger.error(f"Failed to import Django models: {e}", exc_info=True)
            return False
    
    try:
        # Get Django User by Firebase UID
        profile = UserProfile.objects.filter(firebase_uid=user_id).first()
        if not profile:
            logger.warning(f"User profile not found for Firebase UID: {user_id}")
            return False
        user = profile.user
        
        # Convert duration from seconds to TimeField
        import datetime
        hours, remainder = divmod(duration_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        duration = datetime.time(hours, minutes, seconds)
        
        # Map interview types to Django format
        django_interview_type = None
        if interview_type in ["Technical", "Coding"]:
            django_interview_type = "Technical Interview"
        elif interview_type == "HR":
            django_interview_type = "HR Interview"
        elif interview_type == "CaseStudy":
            django_interview_type = "Case Study Interview"
        else:
            logger.warning(f"Unknown interview type: {interview_type}")
            return False
        
        # Get or create InterviewTest object
        interview_test = None
        if interview_test_id:
            try:
                interview_test = InterviewTest.objects.get(id=interview_test_id)
            except InterviewTest.DoesNotExist:
                logger.warning(f"InterviewTest {interview_test_id} not found, will use default")
        
        # If no interview_test_id or not found, get default for interview type
        if not interview_test:
            interview_test = InterviewTest.objects.filter(
                interview_mode=django_interview_type
            ).first()
            if not interview_test:
                logger.error(f"No InterviewTest found for {django_interview_type}")
                return False
        
        # Prepare common feedback data
        common_data = {
            "user": user,
            "session_id": session_id,
            "interview_type": interview_test,
            "duration": duration,
            "strengths": feedback_data.get("strengths", []),
            "areas_of_improvements": feedback_data.get("areas_of_improvements", []),
            "interaction_log": interaction_log,
            "interaction_status_log": feedback_data.get("interaction_log_feedback", []),
            "soft_skill_summary": soft_skill_summary or {},
            "big5_profile": big5_profile or {},
        }
        
        # Save based on interview type
        if django_interview_type == "Technical Interview":
            technical_data = {
                **common_data,
                "language_score": feedback_data.get("language_score", 0),
                "framework_score": feedback_data.get("framework_score", 0),
                "algorithms_score": feedback_data.get("algorithms_score", 0),
                "data_structures_score": feedback_data.get("data_structures_score", 0),
                "approach_score": feedback_data.get("approach_score", 0),
                "optimization_score": feedback_data.get("optimization_score", 0),
                "debugging_score": feedback_data.get("debugging_score", 0),
                "syntax_score": feedback_data.get("syntax_score", 0),
            }
            feedback, created = TechnicalFeedback.objects.update_or_create(
                session_id=session_id,
                user=user,
                defaults=technical_data
            )
            logger.info(f"Technical feedback {'created' if created else 'updated'} in database: session_id={session_id}")
            
        elif django_interview_type == "HR Interview":
            hr_data = {
                **common_data,
                "clarity_score": feedback_data.get("clarity_score", 0),
                "confidence_score": feedback_data.get("confidence_score", 0),
                "structure_score": feedback_data.get("structure_score", 0),
                "engagement_score": feedback_data.get("engagement_score", 0),
                "values_score": feedback_data.get("values_score", 0),
                "teamwork_score": feedback_data.get("teamwork_score", 0),
                "growth_score": feedback_data.get("growth_score", 0),
                "initiative_score": feedback_data.get("initiative_score", 0),
            }
            feedback, created = HRFeedback.objects.update_or_create(
                session_id=session_id,
                user=user,
                defaults=hr_data
            )
            logger.info(f"HR feedback {'created' if created else 'updated'} in database: session_id={session_id}")
            
        elif django_interview_type == "Case Study Interview":
            case_study_data = {
                **common_data,
                "problem_understanding_score": feedback_data.get("problem_understanding_score", 0),
                "hypothesis_score": feedback_data.get("hypothesis_score", 0),
                "analysis_score": feedback_data.get("analysis_score", 0),
                "synthesis_score": feedback_data.get("synthesis_score", 0),
                "business_judgment_score": feedback_data.get("business_judgment_score", 0),
                "creativity_score": feedback_data.get("creativity_score", 0),
                "decision_making_score": feedback_data.get("decision_making_score", 0),
                "impact_orientation_score": feedback_data.get("impact_orientation_score", 0),
            }
            feedback, created = CaseStudyFeedback.objects.update_or_create(
                session_id=session_id,
                user=user,
                defaults=case_study_data
            )
            logger.info(f"Case study feedback {'created' if created else 'updated'} in database: session_id={session_id}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error saving feedback to database: {e}", exc_info=True)
        return False
