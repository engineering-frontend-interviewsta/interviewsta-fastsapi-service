#!/usr/bin/env python3
"""Verify Prisma can connect to PostgreSQL (uses PRISMA_DATABASE_URL / config DB_*)."""
import asyncio
import sys
from pathlib import Path

# Project root on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.prisma_db import ensure_prisma_env


async def main() -> None:
    ensure_prisma_env()
    from prisma import Prisma

    db = Prisma()
    await db.connect()
    try:
        rows = await db.query_raw("SELECT 1 AS ok")
        print("Prisma OK:", rows)
    finally:
        await db.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
