"""
FastAPI dependencies for authentication, database, Redis, etc.
"""
import asyncio
from fastapi import Depends, HTTPException, status, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from redis import Redis
from typing import Optional
import jwt
try:
    from jwt.exceptions import ExpiredSignatureError, PyJWTError
except ImportError:
    # PyJWT not fully installed or wrong package; use Exception and message checks
    ExpiredSignatureError = type("ExpiredSignatureError", (Exception,), {})
    PyJWTError = Exception
import logging

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

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

def _verify_jwt_token(token: str) -> dict:
    """
    Verify JWT token from DRF backend.
    
    Args:
        token: JWT access token
        
    Returns:
        dict: Decoded token payload
        
    Raises:
        ExpiredSignatureError / PyJWTError: If token is invalid or expired
    """
    # Use the same secret and algorithm as your DRF backend
    # For djangorestframework-simplejwt defaults:
    decoded = jwt.decode(
        token,
        'django-insecure-kkx4u+$)leey_1p9s8a$b7ayc*0an21$y9ho4#ntouyo7xns=b',  # Same as Django SECRET_KEY or SIGNING_KEY
        algorithms=["HS256"],  # Usually "HS256" or "RS256"
        # Add these if using djangorestframework-simplejwt defaults:
        options={
            "verify_signature": True,
            "verify_exp": True,
            "verify_aud": False,  # Set to True if you use audience claim
        }
    )
    return decoded

async def verify_jwt_token(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> dict:
    """
    Verify JWT token and return user info.
    Verification runs in a thread pool to avoid blocking the event loop.
    """
    token = credentials.credentials
    try:
        # Run JWT decode in thread pool (crypto operations can be CPU-bound)
        decoded_token = await asyncio.to_thread(_verify_jwt_token, token)
    except ExpiredSignatureError:
        logger.warning("Expired JWT token")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except PyJWTError as e:
        logger.warning(f"Invalid JWT token: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        logger.error(f"Error verifying JWT token: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Extract user info from JWT payload
    # Support both Firebase-style (sub/uid) and DRF-style (user_id) tokens
    uid = decoded_token.get("user_id") or decoded_token.get("uid") or decoded_token.get("sub")
    print("This is the decoded token:", decoded_token)

    user_info = {
        "user_id": decoded_token.get("user_id"),
        "uid": uid,
        "email": decoded_token.get("email"),
        "username": decoded_token.get("username"),
    }
    return user_info

async def get_current_user(user_info: dict = Depends(verify_jwt_token)) -> dict:
    """
    Get current authenticated user
    
    Args:
        user_info: User info from JWT token verification
        
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
        decoded_token = _verify_jwt_token(token)
        uid = decoded_token.get("user_id") or decoded_token.get("uid") or decoded_token.get("sub")
        print("This is the decoded token:", decoded_token)
        return {
            "user_id": decoded_token.get("user_id"),
            "uid": uid,
            "email": decoded_token.get("email"),
            "username": decoded_token.get("username"),
        }
    except Exception:
        return None

async def verify_token_from_query(token: Optional[str] = None) -> dict:
    """
    Verify JWT token from query parameter (for SSE where headers aren't supported)
    
    Args:
        token: JWT access token from query parameter
        
    Returns:
        dict: User information from JWT token
        
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
        decoded_token = _verify_jwt_token(token)
        uid = decoded_token.get("user_id") or decoded_token.get("uid") or decoded_token.get("sub")
        user_info = {
            "user_id": decoded_token.get("user_id"),
            "uid": uid,
            "email": decoded_token.get("email"),
            "username": decoded_token.get("username"),
        }
        return user_info
        
    except ExpiredSignatureError:
        logger.warning("Expired JWT token from query parameter")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
        )
    except PyJWTError as e:
        logger.warning(f"Invalid JWT token from query parameter: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
        )
    except Exception as e:
        logger.error(f"Error verifying JWT token from query: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
        )
