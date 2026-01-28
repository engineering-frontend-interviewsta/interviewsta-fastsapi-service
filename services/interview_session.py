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
            ]
            
            for key in keys:
                if self.redis.exists(key):
                    self.redis.expire(key, self.expire_seconds)
            
            return True
            
        except Exception as e:
            logger.error(f"Error extending expiry for session {session_id}: {e}")
            return False
