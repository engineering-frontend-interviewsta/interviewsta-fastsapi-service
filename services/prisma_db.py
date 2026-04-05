"""
Prisma Client Python lifecycle and connection helpers.

Set PRISMA_DATABASE_URL in .env, or rely on config DB_HOST / DB_PORT / DB_USERNAME / DB_PASSWORD / DB_NAME
(see config.Settings.get_prisma_database_url). Must match prisma/schema.prisma env var name.
"""
import logging
import os
from typing import Any, Optional

from config import get_settings

logger = logging.getLogger(__name__)

_prisma_client: Optional[Any] = None


def ensure_prisma_env() -> str:
    """Ensure PRISMA_DATABASE_URL is set for the generated client and return the URL (no password logged)."""
    url = get_settings().get_prisma_database_url()
    os.environ["PRISMA_DATABASE_URL"] = url
    return url


async def get_prisma():
    """Lazy singleton async Prisma client; call await client.connect() before queries."""
    global _prisma_client
    ensure_prisma_env()
    from prisma import Prisma

    if _prisma_client is None:
        _prisma_client = Prisma()
    return _prisma_client


async def connect_prisma() -> None:
    """Connect the singleton client (e.g. FastAPI lifespan)."""
    client = await get_prisma()
    await client.connect()
    logger.info("Prisma connected to PostgreSQL")


async def disconnect_prisma() -> None:
    global _prisma_client
    if _prisma_client is not None:
        await _prisma_client.disconnect()
        _prisma_client = None
        logger.info("Prisma disconnected")
