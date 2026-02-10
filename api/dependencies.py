"""
FastAPI dependencies for authentication, database, Redis, etc.
"""
from fastapi import Depends, HTTPException, status, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from redis import Redis
from typing import Optional
import firebase_admin
from firebase_admin import auth, credentials
import json
import base64
import os
import logging

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Initialize Firebase Admin SDK
_firebase_initialized = False

def initialize_firebase():
    """Initialize Firebase Admin SDK"""
    global _firebase_initialized
    if not _firebase_initialized and not firebase_admin._apps:
        try:
            firebase_json_b64 = settings.FIREBASE_CREDENTIALS_JSON
            if firebase_json_b64:
                firebase_json = base64.b64decode(firebase_json_b64).decode("utf-8")
                cred = credentials.Certificate(json.loads(firebase_json))
                firebase_admin.initialize_app(cred)
                _firebase_initialized = True
                logger.info("Firebase Admin SDK initialized successfully")
            else:
                logger.warning("FIREBASE_CREDENTIALS_JSON not set, Firebase auth will not work")
        except Exception as e:
            logger.error(f"Failed to initialize Firebase: {e}")


# Initialize Firebase on module import
initialize_firebase()

# Security scheme
security = HTTPBearer()

# Redis connection pool
_redis_client: Optional[Redis] = None


def get_redis() -> Redis:
    """Get Redis client instance"""
    global _redis_client
    if _redis_client is None:
        _redis_client = Redis.from_url(
            settings.REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_keepalive=True,
            health_check_interval=30
        )
    return _redis_client


async def verify_firebase_token(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> dict:
    """
    Verify Firebase ID token and return user info
    
    Args:
        credentials: Bearer token from Authorization header
        
    Returns:
        dict: User information from Firebase token
        
    Raises:
        HTTPException: If token is invalid or expired
    """
    token = credentials.credentials
    
    try:
        # Verify the token
        decoded_token = auth.verify_id_token(token)
        
        # Extract user info
        user_info = {
            "uid": decoded_token["uid"],
            "email": decoded_token.get("email"),
            "email_verified": decoded_token.get("email_verified", False),
            "name": decoded_token.get("name"),
            "picture": decoded_token.get("picture"),
        }
        
        return user_info
        
    except auth.InvalidIdTokenError:
        logger.warning("Invalid Firebase token")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except auth.ExpiredIdTokenError:
        logger.warning("Expired Firebase token")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        logger.error(f"Error verifying Firebase token: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user(user_info: dict = Depends(verify_firebase_token)) -> dict:
    """
    Get current authenticated user
    
    Args:
        user_info: User info from Firebase token verification
        
    Returns:
        dict: Current user information
    """
    return user_info


async def get_optional_user(authorization: Optional[str] = Header(None)) -> Optional[dict]:
    """
    Get current user if authenticated, otherwise None
    Useful for endpoints that work with or without auth
    
    Args:
        authorization: Authorization header value
        
    Returns:
        Optional[dict]: User info if authenticated, None otherwise
    """
    if not authorization or not authorization.startswith("Bearer "):
        return None
    
    try:
        token = authorization.replace("Bearer ", "")
        decoded_token = auth.verify_id_token(token)
        return {
            "uid": decoded_token["uid"],
            "email": decoded_token.get("email"),
        }
    except Exception:
        return None


async def verify_token_from_query(token: Optional[str] = None) -> dict:
    """
    Verify Firebase token from query parameter (for SSE where headers aren't supported)
    
    Args:
        token: Firebase ID token from query parameter
        
    Returns:
        dict: User information from Firebase token
        
    Raises:
        HTTPException: If token is invalid or expired
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token required",
        )
    
    try:
        # Verify the token
        decoded_token = auth.verify_id_token(token)
        
        # Extract user info
        user_info = {
            "uid": decoded_token["uid"],
            "email": decoded_token.get("email"),
            "email_verified": decoded_token.get("email_verified", False),
            "name": decoded_token.get("name"),
            "picture": decoded_token.get("picture"),
        }
        
        return user_info
        
    except auth.InvalidIdTokenError:
        logger.warning("Invalid Firebase token from query parameter")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
        )
    except auth.ExpiredIdTokenError:
        logger.warning("Expired Firebase token from query parameter")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
        )
    except Exception as e:
        logger.error(f"Error verifying Firebase token from query: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
        )
