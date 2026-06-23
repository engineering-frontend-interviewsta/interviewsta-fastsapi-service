"""
FastAPI dependencies for authentication and shared resources.
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
    ExpiredSignatureError = type("ExpiredSignatureError", (Exception,), {})
    PyJWTError = Exception
import logging
import os

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

JWT_SIGNING_KEY = os.getenv(
    "JWT_SIGNING_KEY",
    settings.JWT_SECRET if hasattr(settings, "JWT_SECRET") else "change-me",
)

security = HTTPBearer()

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
    decoded = jwt.decode(
        token,
        JWT_SIGNING_KEY,
        algorithms=["HS256"],
        options={
            "verify_signature": True,
            "verify_exp": True,
            "verify_aud": False,
        },
    )
    return decoded


async def verify_jwt_token(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> dict:
    """Verify JWT token and return user info."""
    token = credentials.credentials
    try:
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

    uid = decoded_token.get("user_id") or decoded_token.get("uid") or decoded_token.get("sub")
    return {
        "user_id": decoded_token.get("user_id") or uid,
        "uid": uid,
        "email": decoded_token.get("email"),
        "username": decoded_token.get("username") or decoded_token.get("name"),
        "name": decoded_token.get("name"),
        "roles": decoded_token.get("roles") or [],
    }


async def get_current_user(user_info: dict = Depends(verify_jwt_token)) -> dict:
    return user_info


async def get_optional_user(authorization: Optional[str] = Header(None)) -> Optional[dict]:
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
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token required",
        )
    try:
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
    except ExpiredSignatureError:
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
