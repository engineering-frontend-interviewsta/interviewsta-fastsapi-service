"""
Interview session state management using Redis
"""
from redis import Redis
from typing import Dict, Any, Optional
import json
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class InterviewSessionManager:
    """Manage interview session state in Redis"""
    
    def __init__(self, redis_client: Redis, expire_seconds: int = 3600):
        self.redis = redis_client
        self.expire_seconds = expire_seconds
    
    def _get_key(self, session_id: str, suffix: str) -> str:
        """Generate Redis key for session data"""
        return f"session:{session_id}:{suffix}"
    
    def create_session(self, session_id: str, interview_type: str, user_id: str, payload: Dict[str, Any]) -> bool:
        """
        Create a new interview session
        
        Args:
            session_id: Unique session identifier
            interview_type: Type of interview (Technical, HR, etc.)
            user_id: Firebase user ID
            payload: Initial session data
            
        Returns:
            bool: True if session created successfully
        """
        try:
            session_data = {
                "session_id": session_id,
                "interview_type": interview_type,
                "user_id": user_id,
                "status": "initialized",
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
                "payload": payload,
                "messages": [],
                "history": "",
                "last_node": "",
            }
            
            # Store session state
            state_key = self._get_key(session_id, "state")
            self.redis.setex(state_key, self.expire_seconds, json.dumps(session_data))
            
            # Store session status
            status_key = self._get_key(session_id, "status")
            self.redis.setex(status_key, self.expire_seconds, "initialized")
            
            logger.info(f"Created session {session_id} for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating session {session_id}: {e}")
            return False
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session state
        
        Args:
            session_id: Session identifier
            
        Returns:
            Optional[Dict]: Session data or None if not found
        """
        try:
            state_key = self._get_key(session_id, "state")
            data = self.redis.get(state_key)
            
            if data:
                return json.loads(data)
            return None
            
        except Exception as e:
            logger.error(f"Error getting session {session_id}: {e}")
            return None
    
    def update_session(self, session_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update session state
        
        Args:
            session_id: Session identifier
            updates: Dictionary of fields to update
            
        Returns:
            bool: True if updated successfully
        """
        try:
            session = self.get_session(session_id)
            if not session:
                logger.warning(f"Session {session_id} not found for update")
                return False
            
            # Update fields
            session.update(updates)
            session["updated_at"] = datetime.utcnow().isoformat()
            
            # Save back to Redis
            state_key = self._get_key(session_id, "state")
            self.redis.setex(state_key, self.expire_seconds, json.dumps(session))
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating session {session_id}: {e}")
            return False
    
    def set_status(self, session_id: str, status: str) -> bool:
        """
        Update session status
        
        Args:
            session_id: Session identifier
            status: New status value
            
        Returns:
            bool: True if updated successfully
        """
        try:
            status_key = self._get_key(session_id, "status")
            self.redis.setex(status_key, self.expire_seconds, status)
            
            # Also update in session state
            self.update_session(session_id, {"status": status})
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting status for session {session_id}: {e}")
            return False
    
    def get_status(self, session_id: str) -> Optional[str]:
        """Get current session status"""
        try:
            status_key = self._get_key(session_id, "status")
            return self.redis.get(status_key)
        except Exception as e:
            logger.error(f"Error getting status for session {session_id}: {e}")
            return None
    
    def set_response(self, session_id: str, message: str, audio: Optional[str] = None, last_node: Optional[str] = None) -> bool:
        """
        Store AI response for session
        
        Args:
            session_id: Session identifier
            message: AI response text
            audio: Base64 encoded audio (optional)
            last_node: Current workflow node
            
        Returns:
            bool: True if stored successfully
        """
        try:
            response_data = {
                "message": message,
                "audio": audio,
                "last_node": last_node,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            response_key = self._get_key(session_id, "response")
            self.redis.setex(response_key, self.expire_seconds, json.dumps(response_data))
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting response for session {session_id}: {e}")
            return False
    
    def get_response(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get latest AI response"""
        try:
            response_key = self._get_key(session_id, "response")
            data = self.redis.get(response_key)
            
            if data:
                return json.loads(data)
            return None
            
        except Exception as e:
            logger.error(f"Error getting response for session {session_id}: {e}")
            return None
    
    def set_transcript(self, session_id: str, transcript: str) -> bool:
        """Store user transcription"""
        try:
            transcript_key = self._get_key(session_id, "transcript")
            self.redis.setex(transcript_key, self.expire_seconds, transcript)
            return True
        except Exception as e:
            logger.error(f"Error setting transcript for session {session_id}: {e}")
            return False
    
    def get_transcript(self, session_id: str) -> Optional[str]:
        """Get latest user transcription"""
        try:
            transcript_key = self._get_key(session_id, "transcript")
            return self.redis.get(transcript_key)
        except Exception as e:
            logger.error(f"Error getting transcript for session {session_id}: {e}")
            return None
    
    def delete_session(self, session_id: str) -> bool:
        """Delete all session data"""
        try:
            keys = [
                self._get_key(session_id, "state"),
                self._get_key(session_id, "status"),
                self._get_key(session_id, "response"),
                self._get_key(session_id, "transcript"),
            ]
            
            self.redis.delete(*keys)
            logger.info(f"Deleted session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting session {session_id}: {e}")
            return False
    
    def extend_expiry(self, session_id: str) -> bool:
        """Extend session expiry time"""
        try:
            keys = [
                self._get_key(session_id, "state"),
                self._get_key(session_id, "status"),
                self._get_key(session_id, "response"),
                self._get_key(session_id, "transcript"),
                self._get_key(session_id, "warning"),
            ]
            
            for key in keys:
                if self.redis.exists(key):
                    self.redis.expire(key, self.expire_seconds)
            
            return True
            
        except Exception as e:
            logger.error(f"Error extending expiry for session {session_id}: {e}")
            return False
    
    def set_warning(self, session_id: str, warning_type: str, message: str) -> bool:
        """Store warning message for SSE stream to pick up"""
        try:
            warning_data = {
                "type": warning_type,
                "message": message,
                "timestamp": datetime.utcnow().isoformat()
            }
            warning_key = self._get_key(session_id, "warning")
            self.redis.setex(warning_key, self.expire_seconds, json.dumps(warning_data))
            return True
        except Exception as e:
            logger.error(f"Error setting warning for session {session_id}: {e}")
            return False
    
    def get_warning(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get and clear warning message"""
        try:
            warning_key = self._get_key(session_id, "warning")
            data = self.redis.get(warning_key)
            if data:
                warning = json.loads(data)
                # Clear warning after reading
                self.redis.delete(warning_key)
                return warning
            return None
        except Exception as e:
            logger.error(f"Error getting warning for session {session_id}: {e}")
            return None
    
    def get_soft_skills_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Aggregate soft skills metrics from video quality data
        Similar to old code's _compute_soft_skill_summary
        """
        try:
            metrics_key = f"session:{session_id}:video_metrics"
            all_metrics = redis_client.lrange(metrics_key, 0, -1)  # Get all metrics
            
            if not all_metrics:
                return {
                    'eye_contact': 0,
                    'confidence': 0,
                    'nervousness': 0,
                    'engagement': 0,
                    'distraction': 0,
                    'verdict': 'No Data',
                    'speech_summary': {
                        'grammar': 0,
                        'fluency': 0,
                        'fillers': 0,
                        'clarity': 0
                    }
                }
            
            # Aggregate metrics
            gaze_values = []
            confidence_values = []
            nervousness_values = []
            engagement_values = []
            distraction_values = []
            
            for metric_str in all_metrics:
                try:
                    metric = json.loads(metric_str)
                    if metric.get("gaze") is not None:
                        gaze_values.append(metric["gaze"])
                    if metric.get("confidence") is not None:
                        confidence_values.append(metric["confidence"])
                    if metric.get("nervousness") is not None:
                        nervousness_values.append(metric["nervousness"])
                    if metric.get("engagement") is not None:
                        engagement_values.append(metric["engagement"])
                    if metric.get("distraction") is not None:
                        distraction_values.append(metric["distraction"])
                except:
                    continue
            
            # Compute averages
            eye_contact = sum(gaze_values) / len(gaze_values) if gaze_values else 0
            confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0
            nervousness = sum(nervousness_values) / len(nervousness_values) if nervousness_values else 0
            engagement = sum(engagement_values) / len(engagement_values) if engagement_values else 0
            distraction = sum(distraction_values) / len(distraction_values) if distraction_values else 0
            
            # Compute overall score (weighted average)
            normalized_engagement = engagement
            normalized_confidence = confidence
            normalized_eye_contact = eye_contact
            normalized_nervousness = 100 - nervousness  # Invert: lower nervousness = higher score
            normalized_distraction = 100 - distraction  # Invert: lower distraction = higher score
            
            overall_score = (
                normalized_engagement * 0.25 +
                normalized_confidence * 0.25 +
                normalized_eye_contact * 0.20 +
                normalized_nervousness * 0.15 +
                normalized_distraction * 0.15
            )
            
            # Determine verdict
            if overall_score >= 85:
                verdict = "Excellent"
            elif overall_score >= 70:
                verdict = "Good"
            else:
                verdict = "Needs Improvement"
            
            # Generate speech scores (simplified - in production might use actual speech analysis)
            speech_summary = {
                'grammar': min(100, max(0, confidence * 0.8)),
                'fluency': min(100, max(0, engagement * 0.9)),
                'fillers': min(100, max(0, (100 - nervousness) * 0.7)),
                'clarity': min(100, max(0, confidence * 0.85))
            }
            
            summary = {
                'eye_contact': round(eye_contact, 2),
                'confidence': round(confidence, 2),
                'nervousness': round(nervousness, 2),
                'engagement': round(engagement, 2),
                'distraction': round(distraction, 2),
                'overall_score': round(overall_score, 2),
                'verdict': verdict,
                'speech_summary': speech_summary
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error computing soft skills summary for session {session_id}: {e}")
            return None