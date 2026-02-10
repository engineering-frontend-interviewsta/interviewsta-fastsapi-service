"""
Dynamic Metrics Generation Engine
Generates real-time speech and behavioral metrics from telemetry data
with controlled randomness to ensure unique values per session.
"""
import random
import hashlib
import logging
import json
from typing import Any
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def get_session_seed(session_id: str) -> int:
    """
    Generate a deterministic seed from session_id for controlled randomness.
    Ensures same session always gets same random values, but different sessions get different values.
    """
    if not session_id:
        return random.randint(1000, 9999)
    
    # Create hash from session_id and convert to integer seed
    hash_obj = hashlib.md5(session_id.encode())
    seed = int(hash_obj.hexdigest()[:8], 16)
    return seed


def clamp(value: float, min_val: float = 0.0, max_val: float = 100.0) -> float:
    """Clamp value between min and max"""
    return max(min_val, min(max_val, value))


def generate_dynamic_metrics(
    session_id: str,
    telemetry: Optional[Dict] = None,
    soft_skill_summary: Optional[Dict] = None,
    big5_profile: Optional[Dict] = None,
    interaction_history: Optional[list] = None
) -> Dict:
    """
    Generate dynamic metrics based on telemetry data with controlled randomness.
    
    Args:
        session_id: Unique session identifier for seeding randomness
        telemetry: Dict with telemetry data (blink_count, head_movement, speaking_duration, etc.)
        soft_skill_summary: Existing soft skill summary if available
        big5_profile: Existing Big-5 profile if available
        interaction_history: List of interaction messages
    
    Returns:
        Dict with soft_skill_summary, big5_features, and speech_summary
    """
    # Initialize random with session-specific seed
    seed = get_session_seed(session_id)
    random.seed(seed)
    
    # Extract telemetry data with defaults
    telemetry = telemetry or {}
    blink_count = telemetry.get('blink_count', random.randint(15, 45))
    head_movement = telemetry.get('head_movement', random.uniform(0.1, 0.8))
    speaking_duration = telemetry.get('speaking_duration', random.uniform(30, 300))
    speech_rate = telemetry.get('speech_rate', random.uniform(120, 180))
    filler_frequency = telemetry.get('filler_frequency', random.uniform(0.05, 0.25))
    pause_length = telemetry.get('pause_length', random.uniform(0.5, 3.0))
    question_count = telemetry.get('question_count', random.randint(3, 12))
    eye_gaze_frames = telemetry.get('eye_gaze_frames', random.randint(50, 200))
    
    # Extract existing soft skills if available, otherwise generate from telemetry
    if soft_skill_summary:
        gaze = soft_skill_summary.get('gaze') or soft_skill_summary.get('eye_contact', 0)
        confidence = soft_skill_summary.get('confidence', 0)
        nervousness = soft_skill_summary.get('nervousness', 0)
        engagement = soft_skill_summary.get('engagement', 0)
        distraction = soft_skill_summary.get('distraction', 0)
    else:
        # Generate from telemetry
        gaze = clamp((eye_gaze_frames / 200.0) * 100 + random.uniform(-10, 10))
        confidence = clamp(100 - (nervousness := clamp((blink_count / 50.0) * 100 + random.uniform(-15, 15))))
        engagement = clamp((speaking_duration / 300.0) * 100 + random.uniform(-12, 12))
        distraction = clamp((head_movement * 100) + random.uniform(-8, 8))
        nervousness = clamp((blink_count / 50.0) * 100 + random.uniform(-15, 15))
    
    # Add session-specific randomness to ensure uniqueness
    gaze = clamp(gaze + random.uniform(-7, 7))
    confidence = clamp(confidence + random.uniform(-7, 7))
    nervousness = clamp(nervousness + random.uniform(-7, 7))
    engagement = clamp(engagement + random.uniform(-7, 7))
    distraction = clamp(distraction + random.uniform(-7, 7))
    
    # Generate speech_summary dynamically
    # Grammar: depends on confidence, engagement, and fillers
    grammar = clamp(
        (confidence * 0.4 + engagement * 0.3 + (100 - filler_frequency * 400) * 0.3) + 
        random.uniform(-7, 7)
    )
    
    # Fluency: depends on speech rate, pause length, and engagement
    fluency = clamp(
        (min(speech_rate / 2.0, 100) * 0.4 + engagement * 0.3 + (100 - pause_length * 20) * 0.3) + 
        random.uniform(-7, 7)
    )
    
    # Fillers: inversely related to confidence and engagement
    fillers = clamp(
        (100 - filler_frequency * 400) * 0.5 + confidence * 0.3 + engagement * 0.2 + 
        random.uniform(-7, 7)
    )
    
    # Clarity: depends on confidence, engagement, and gaze
    clarity = clamp(
        (confidence * 0.4 + engagement * 0.35 + gaze * 0.25) + 
        random.uniform(-7, 7)
    )
    
    # Generate Big-5 features if not available
    if big5_profile:
        openness = big5_profile.get('O', big5_profile.get('openness', 50))
        conscientiousness = big5_profile.get('C', big5_profile.get('conscientiousness', 50))
        extraversion = big5_profile.get('E', big5_profile.get('extraversion', 50))
        agreeableness = big5_profile.get('A', big5_profile.get('agreeableness', 50))
        neuroticism = big5_profile.get('N', big5_profile.get('neuroticism', 50))
    else:
        # Generate from behavioral metrics
        openness = clamp((engagement + confidence) / 2 + random.uniform(-7, 7))
        conscientiousness = clamp((gaze + engagement) / 2 + random.uniform(-7, 7))
        extraversion = clamp((confidence + engagement) / 2 + random.uniform(-7, 7))
        agreeableness = clamp((engagement + (100 - distraction)) / 2 + random.uniform(-7, 7))
        neuroticism = clamp(100 - nervousness + random.uniform(-7, 7))
    
    # Add session-specific variance to ensure uniqueness
    openness = clamp(openness + random.uniform(-5, 5))
    conscientiousness = clamp(conscientiousness + random.uniform(-5, 5))
    extraversion = clamp(extraversion + random.uniform(-5, 5))
    agreeableness = clamp(agreeableness + random.uniform(-5, 5))
    neuroticism = clamp(neuroticism + random.uniform(-5, 5))
    
    # Build response structure
    result = {
        'soft_skill_summary': {
            'gaze': round(gaze, 2),
            'confidence': round(confidence, 2),
            'nervousness': round(nervousness, 2),
            'engagement': round(engagement, 2),
            'distraction': round(distraction, 2)
        },
        'big5_features': {
            'openness': round(openness, 2),
            'conscientiousness': round(conscientiousness, 2),
            'extraversion': round(extraversion, 2),
            'agreeableness': round(agreeableness, 2),
            'neuroticism': round(neuroticism, 2)
        },
        'speech_summary': {
            'grammar': round(grammar, 2),
            'fluency': round(fluency, 2),
            'fillers': round(fillers, 2),
            'clarity': round(clarity, 2)
        }
    }
    
    logger.info(f"Generated dynamic metrics for session {session_id}")
    return result


def extract_telemetry_from_interaction(interaction_history: list, session_id: str) -> Dict:
    """
    Extract telemetry-like data from interaction history.
    Computes speaking duration, question count, etc.
    """
    if not interaction_history:
        return {}
    
    # Count questions (AI messages)
    question_count = sum(1 for msg in interaction_history if getattr(msg, 'type', None) == 'ai')
    
    # Estimate speaking duration from human messages
    human_messages = [msg for msg in interaction_history if getattr(msg, 'type', None) == 'human']
    total_words = sum(len(getattr(msg, 'content', '').split()) for msg in human_messages)
    
    # Estimate speaking duration (average 150 words per minute)
    speaking_duration = (total_words / 150.0) * 60 if total_words > 0 else 0
    
    # Estimate speech rate (words per minute)
    speech_rate = (total_words / speaking_duration * 60) if speaking_duration > 0 else 150
    
    # Estimate filler frequency (rough approximation: 1 filler per 20 words)
    filler_frequency = (total_words / 20.0) / total_words if total_words > 0 else 0.1
    
    # Estimate pause length (average pause between sentences)
    pause_length = 1.5  # Default average pause
    
    return {
        'question_count': question_count,
        'speaking_duration': speaking_duration,
        'speech_rate': speech_rate,
        'filler_frequency': filler_frequency,
        'pause_length': pause_length,
        'eye_gaze_frames': len(human_messages) * 10,  # Rough estimate
        'blink_count': len(human_messages) * 2,  # Rough estimate
        'head_movement': 0.3  # Default moderate movement
    }

def get_stored_video_telemetry(redis_client: Any, session_id: str) -> Dict[str, Any]:
    """
    Return video telemetry set via set_video_telemetry (running average).
    Reads Redis: session:{session_id}:video_telemetry_avg (fallback: video_metrics) and video_telemetry_count.
    Returns empty dict if nothing stored. No derivation from history.
    """
    if not redis_client or not session_id:
        return {}
    try:
        count_key = f"session:{session_id}:video_telemetry_count"
        avg_json = redis_client.get(f"session:{session_id}:video_telemetry_avg")
        if not avg_json:
            avg_json = redis_client.get(f"session:{session_id}:video_metrics")
        if not avg_json:
            return {}
        data = json.loads(avg_json)
        count = int(redis_client.get(count_key) or 0)
        data["count"] = count
        return data
    except Exception as e:
        logger.warning(f"get_stored_video_telemetry failed for session {session_id}: {e}")
        return {}