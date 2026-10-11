"""Shared fixtures for the extraction_jobs repository tests: two users and a conversation."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from uuid import UUID

import asyncpg
import pytest_asyncio

USER_A = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
USER_B = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def rls_users(admin_pool: asyncpg.Pool) -> AsyncIterator[tuple[UUID, UUID]]:
    """Create two test users (A, B) and tear them down after the test."""
    async with admin_pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO users (id, email, timezone, created_at, updated_at)
            VALUES
                ($1, 'extraction-test-a@test.local', 'UTC', now(), now()),
                ($2, 'extraction-test-b@test.local', 'UTC', now(), now())
            ON CONFLICT (id) DO NOTHING
            """,
            USER_A,
            USER_B,
        )
    yield USER_A, USER_B
    async with admin_pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM extraction_jobs WHERE user_id IN ($1, $2)",
            USER_A,
            USER_B,
        )
        await conn.execute(
            "DELETE FROM users WHERE id IN ($1, $2)",
            USER_A,
            USER_B,
        )


@pytest_asyncio.fixture
async def conversation_id(
    admin_pool: asyncpg.Pool, rls_users: tuple[UUID, UUID]
) -> AsyncIterator[int]:
    """Seed a minimal conversation row for user A, return its integer id."""
    user_a, _ = rls_users
    async with admin_pool.acquire() as conn:
        # Need a topic first.
        topic_id = await conn.fetchval(
            """
            INSERT INTO topics (path, title, description, user_id, created_at, updated_at)
            VALUES ('inbox', 'Inbox', '', $1, now(), now())
            RETURNING id
            """,
            user_a,
        )
        conv_id = await conn.fetchval(
            """
            INSERT INTO conversations
                (topic_id, user_id, title_encrypted, title_nonce, slug, source,
                 summary_encrypted, summary_nonce, tags, participants,
                 message_count, created_at, updated_at, json_path, search_vector)
            VALUES ($1, $2,
                    'dummytitle'::bytea, 'nonce_title1'::bytea,
                    'test-conv', 'chatgpt', 'dummysum'::bytea, 'nonce_summ12'::bytea,
                    '{}', '{}', 0, now(), now(), 'test.json',
                    to_tsvector('english', 'test conversation'))
            RETURNING id
            """,
            topic_id,
            user_a,
        )
    yield int(conv_id)
    async with admin_pool.acquire() as conn:
        await conn.execute("DELETE FROM conversations WHERE id = $1", int(conv_id))
        await conn.execute("DELETE FROM topics WHERE id = $1", int(topic_id))


async def wait_until_blocked_on_a_lock(admin_pool: asyncpg.Pool) -> None:
    """Return once some backend waits on a lock; fail after about five seconds."""
    for _ in range(500):
        if await admin_pool.fetchval("SELECT count(*) FROM pg_locks WHERE NOT granted"):
            return
        await asyncio.sleep(0.01)
    raise AssertionError("no backend ever waited on a lock")
