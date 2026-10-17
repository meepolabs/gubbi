"""Payload builders and row readers shared by the ingest endpoint tests.

The app, client and user fixtures live in ``tests/api/v1/ingest/conftest.py``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

import asyncpg
from httpx import AsyncClient

API_PREFIX = "/api/v1"
ENDPOINT = f"{API_PREFIX}/ingest/conversations"

TEST_USER_ID = UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")


NIGHTLY_HOUR = 3
OLD_ACTIVITY = datetime(2020, 1, 1, tzinfo=UTC)
PAST_ONBOARDING = timedelta(days=8)
CREATED = datetime(2026, 2, 1, 8, tzinfo=UTC)
UPDATED = datetime(2026, 2, 3, 20, tzinfo=UTC)


def build_conv(
    platform_id: str,
    platform: str = "chatgpt",
    title: str = "Test conversation",
    msg_count: int = 2,
    message_times: list[datetime] | None = None,
    *,
    with_ids: bool = False,
) -> dict[str, Any]:
    """A conversation of ``msg_count`` alternating turns ``Message 0``, ``Message 1``, ...

    ``with_ids`` gives message ``i`` the source id ``{platform_id}-m{i}``, so a
    re-send with a larger ``msg_count`` repeats the earlier ids.
    """
    now = datetime.now(UTC)
    messages: list[dict[str, Any]] = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"Message {i}"}
        for i in range(msg_count)
    ]
    if with_ids:
        for i, message in enumerate(messages):
            message["message_id"] = f"{platform_id}-m{i}"
    for message, stamp in zip(messages, message_times or [], strict=False):
        message["timestamp"] = stamp.isoformat()
    return {
        "platform": platform,
        "platform_id": platform_id,
        "title": title,
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
        "messages": messages,
    }


def headers() -> dict[str, str]:
    return {"X-Auth-User-Id": str(TEST_USER_ID)}


async def post(client: AsyncClient, source: str, *convs: dict[str, Any]) -> dict[str, Any]:
    resp = await client.post(
        ENDPOINT, json={"source": source, "conversations": list(convs)}, headers=headers()
    )
    assert resp.status_code == 200, resp.text
    body: dict[str, Any] = resp.json()
    return body


async def job_for(pool: asyncpg.Pool, platform_id: str) -> asyncpg.Record:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT ej.*
            FROM extraction_jobs ej
            JOIN conversations c ON c.id = ej.conversation_id
            WHERE c.user_id = $1 AND c.platform_id = $2
            """,
            TEST_USER_ID,
            platform_id,
        )
    assert row is not None, f"no extraction job for {platform_id}"
    return row


async def job_by_id(pool: asyncpg.Pool, job_id: UUID) -> asyncpg.Record:
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM extraction_jobs WHERE id = $1", job_id)
    assert row is not None, f"no extraction job {job_id}"
    return row


async def counts_for(pool: asyncpg.Pool, platform_id: str) -> tuple[int, int]:
    """Count the user's conversations with ``platform_id`` and all their jobs."""
    async with pool.acquire() as conn:
        conversations = await conn.fetchval(
            "SELECT COUNT(*) FROM conversations WHERE user_id = $1 AND platform_id = $2",
            TEST_USER_ID,
            platform_id,
        )
        jobs = await conn.fetchval(
            "SELECT COUNT(*) FROM extraction_jobs WHERE user_id = $1", TEST_USER_ID
        )
    return int(conversations), int(jobs)


async def backdate_jobs(pool: asyncpg.Pool, age: timedelta) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE extraction_jobs SET created_at = created_at - $2::interval WHERE user_id = $1",
            TEST_USER_ID,
            age,
        )


async def admit_all_jobs(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE extraction_jobs SET admitted_at = now(), lane = 'live' WHERE user_id = $1",
            TEST_USER_ID,
        )


MONTHLY = 1000


async def spend(pool: asyncpg.Pool, weighted: int) -> None:
    """Charge the user ``weighted`` tokens this month, monthly allowance first."""
    monthly = min(weighted, MONTHLY)
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO llm_charges
                (user_id, job_kind, job_id, attempt, input_tokens, cached_input_tokens,
                 output_tokens, weighted, monthly_part, credit_part, month)
            VALUES ($1, 'paste', gen_random_uuid(), 1, $2, 0, 0, $2, $3, $4,
                    date_trunc('month', now() AT TIME ZONE 'UTC')::date)
            """,
            TEST_USER_ID,
            weighted,
            monthly,
            weighted - monthly,
        )
