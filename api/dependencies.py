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
import os

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# JWT signing key: must match the secret used to sign tokens (e.g. frontend JWT_SECRET / Nest configService.get('JWT_SECRET')).
# Set JWT_SIGNING_KEY in env to match; same key is used for Bearer and X-Interview-Access-Token.
JWT_SIGNING_KEY = os.getenv("JWT_SIGNING_KEY", "django-insecure-kkx4u+$)leey_1p9s8a$b7ayc*0an21$y9ho4#ntouyo7xns=b")

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
        JWT_SIGNING_KEY,
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
    # Expected payload: { sub, email, name, roles } (same secret as JWT_SECRET on frontend)
    uid = decoded_token.get("user_id") or decoded_token.get("uid") or decoded_token.get("sub")
    user_info = {
        "user_id": decoded_token.get("user_id") or uid,
        "uid": uid,
        "email": decoded_token.get("email"),
        "username": decoded_token.get("username") or decoded_token.get("name"),
        "name": decoded_token.get("name"),
        "roles": decoded_token.get("roles") or [],
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
        return {
            "user_id": decoded_token.get("user_id") or uid,
            "uid": uid,
            "email": decoded_token.get("email"),
            "username": decoded_token.get("username") or decoded_token.get("name"),
            "name": decoded_token.get("name"),
            "roles": decoded_token.get("roles") or [],
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
            "user_id": decoded_token.get("user_id") or uid,
            "uid": uid,
            "email": decoded_token.get("email"),
            "username": decoded_token.get("username") or decoded_token.get("name"),
            "name": decoded_token.get("name"),
            "roles": decoded_token.get("roles") or [],
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


def _decode_jwt_payload(token: str) -> dict:
    """Decode JWT with same key as auth; returns payload dict. Raises PyJWTError on failure."""
    return jwt.decode(
        token,
        JWT_SIGNING_KEY,
        algorithms=["HS256"],
        options={"verify_signature": True, "verify_exp": True, "verify_aud": False},
    )


def _normalize_interview_access_decoded(decoded: dict) -> dict:
    """Copy camelCase keys from X-Interview-Access-Token JWT into snake_case so payload validation gets them."""
    out = dict(decoded)
    if "feedbackItemId" in decoded:
        out["feedback_item_id"] = decoded["feedbackItemId"]
    if "interviewTestId" in decoded:
        out["interview_test_id"] = decoded["interviewTestId"]
    if "fastapiInterviewType" in decoded:
        out["fastapi_interview_type"] = decoded["fastapiInterviewType"]
    return out


async def get_interview_access_payload(
    x_interview_access_token: Optional[str] = Header(None, alias="X-Interview-Access-Token"),
) -> "InterviewAccessTokenPayload":
    """
    Decode and validate the X-Interview-Access-Token header (JWT, same key as Bearer).
    Required for interview endpoints; provides interview_test_id, fastapi_interview_type, etc.
    """
    from schemas.interview import InterviewAccessTokenPayload

    if not x_interview_access_token or not x_interview_access_token.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="X-Interview-Access-Token header is required",
        )
    token = x_interview_access_token.strip()
    try:
        decoded = await asyncio.to_thread(_decode_jwt_payload, token)
    except ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Interview access token has expired",
        )
    except PyJWTError as e:
        logger.warning(f"Invalid interview access token: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid interview access token",
        )
    # Normalize camelCase from JWT so payload gets feedback_item_id etc.
    decoded = _normalize_interview_access_decoded(decoded)
    try:
        return InterviewAccessTokenPayload.model_validate(decoded)
    except Exception as e:
        logger.warning(f"Interview access token payload validation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid interview access token payload",
        )


async def get_interview_access_payload_from_token(token: str) -> "InterviewAccessTokenPayload":
    """
    Decode and validate interview access token from string (e.g. query param for SSE stream).
    Same contract as get_interview_access_payload; use when header is not available (EventSource).
    """
    from schemas.interview import InterviewAccessTokenPayload

    if not token or not token.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Interview access token is required",
        )
    token = token.strip()
    try:
        decoded = await asyncio.to_thread(_decode_jwt_payload, token)
    except ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Interview access token has expired",
        )
    except PyJWTError as e:
        logger.warning(f"Invalid interview access token: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid interview access token",
        )
    decoded = _normalize_interview_access_decoded(decoded)
    try:
        return InterviewAccessTokenPayload.model_validate(decoded)
    except Exception as e:
        logger.warning(f"Interview access token payload validation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid interview access token payload",
        )
