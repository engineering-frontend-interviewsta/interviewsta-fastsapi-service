#!/usr/bin/env python3
"""Print interview_tests listing and fastapi_interview_type counts (uses Prisma + .env PRISMA_DATABASE_URL)."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path


def load_env() -> None:
    p = Path(__file__).resolve().parents[1] / ".env"
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


async def main() -> None:
    load_env()
    from prisma import Prisma

    db = Prisma()
    await db.connect()
    try:
        agg = await db.query_raw(
            """
            SELECT fastapi_interview_type::text AS t, COUNT(*)::int AS c
            FROM interview_tests
            GROUP BY fastapi_interview_type
            ORDER BY c DESC, t NULLS LAST
            """
        )
        print("=== fastapi_interview_type counts ===")
        for row in agg:
            print(f"  {row['t']!r}: {row['c']}")

        total = await db.query_raw("SELECT COUNT(*)::int AS n FROM interview_tests")
        print(f"=== total: {total[0]['n']} ===")

        phases = await db.query_raw(
            "SELECT COUNT(*)::int AS n FROM interview_test_interview_phases"
        )
        print(f"=== interview_test_interview_phases rows: {phases[0]['n']} ===")

        rows = await db.query_raw(
            """
            SELECT id::text, title, fastapi_interview_type::text AS fit, is_active, code
            FROM interview_tests
            ORDER BY fastapi_interview_type NULLS LAST, title
            """
        )
        print("\n=== all rows (id, title, type, is_active, code) ===")
        for r in rows:
            print(
                f"{r['id']}\t{r['title']}\t{r['fit']}\t{r['is_active']}\t{r['code']!r}"
            )
    finally:
        await db.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
