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

def _init_django():
    """Initialize Django - call this lazily when needed"""
    global _django_initialized, ResumeAnalysis, User, UserProfile
    
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
        from myapp.models import ResumeAnalysis, User, UserProfile
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
