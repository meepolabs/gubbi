"""Security test: extraction_jobs RLS isolation.

Verifies that user B cannot see user A's extraction_jobs rows when
accessing via the RLS-enforced app pool.

Run:
    pytest tests/security/test_extraction_jobs_rls.py -v
"""

from __future__ import annotations

from datetime import date
from uuid import UUID

import asyncpg
import pytest
import pytest_asyncio
from gubbi_common.db.user_scoped import user_scoped_connection

from gubbi.storage.repositories.extraction_jobs import (
    create_pending,
    get_status_counts,
)

pytestmark = pytest.mark.asyncio(loop_scope="session")

_USER_A = UUID("dddddddd-dddd-dddd-dddd-dddddddddddd")
_USER_B = UUID("eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def rls_test_users(admin_pool: asyncpg.Pool) -> tuple[UUID, UUID]:
    """Create two isolated users for cross-tenant RLS testing."""
    async with admin_pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO users (id, email, timezone, created_at, updated_at)
            VALUES
                ($1, 'rls-extraction-a@test.local', 'UTC', now(), now()),
                ($2, 'rls-extraction-b@test.local', 'UTC', now(), now())
            ON CONFLICT (id) DO NOTHING
            """,
            _USER_A,
            _USER_B,
        )
    yield _USER_A, _USER_B
    async with admin_pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM extraction_jobs WHERE user_id IN ($1, $2)",
            _USER_A,
            _USER_B,
        )
        await conn.execute("DELETE FROM users WHERE id IN ($1, $2)", _USER_A, _USER_B)


@pytest_asyncio.fixture
async def conversation_for_a(admin_pool: asyncpg.Pool, rls_test_users: tuple[UUID, UUID]) -> int:
    """Seed a minimal conversation row for user A and return its integer id."""
    user_a, _ = rls_test_users
    async with admin_pool.acquire() as conn:
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
                    'rls-test-conv', 'chatgpt', 'dummysum'::bytea, 'nonce_summ12'::bytea,
                    '{}', '{}', 0, now(), now(), 'rls-test.json',
                    to_tsvector('english', 'rls test conversation'))
            RETURNING id
            """,
            topic_id,
            user_a,
        )
    yield int(conv_id)
    async with admin_pool.acquire() as conn:
        await conn.execute("DELETE FROM conversations WHERE id = $1", int(conv_id))
        await conn.execute("DELETE FROM topics WHERE id = $1", int(topic_id))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_user_b_cannot_see_user_a_jobs(
    app_pool: asyncpg.Pool,
    rls_test_users: tuple[UUID, UUID],
    conversation_for_a: int,
) -> None:
    """User A inserts a job; user B's get_status_counts returns zeros."""
    user_a, user_b = rls_test_users

    # User A creates a pending job.
    async with user_scoped_connection(app_pool, user_id=user_a) as conn:
        await create_pending(
            conn, user_a, conversation_for_a, "chatgpt", period_start=date(2026, 5, 1)
        )
        a_counts = await get_status_counts(conn)

    # User A sees their own job.
    assert a_counts.in_flight_count == 1

    # User B sees nothing.
    async with user_scoped_connection(app_pool, user_id=user_b) as conn:
        b_counts = await get_status_counts(conn)

    assert b_counts.in_flight_count == 0
    assert b_counts.synced_count == 0
    assert b_counts.last_sync_at is None


async def test_user_b_cannot_read_user_a_row_by_pk(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    rls_test_users: tuple[UUID, UUID],
    conversation_for_a: int,
) -> None:
    """Even when user B knows user A's job UUID, RLS prevents a direct SELECT."""
    user_a, user_b = rls_test_users

    # User A creates a job; capture the UUID.
    async with user_scoped_connection(app_pool, user_id=user_a) as conn:
        job_id = await create_pending(
            conn, user_a, conversation_for_a, "chatgpt", period_start=date(2026, 5, 1)
        )

    # Admin pool can see the row (BYPASSRLS).
    async with admin_pool.acquire() as conn:
        row = await conn.fetchrow("SELECT id FROM extraction_jobs WHERE id = $1", job_id)
    assert row is not None, "Admin pool should see the row"

    # User B's app pool cannot see the row -- returns None (not a 403, because
    # RLS silently filters rows; the query succeeds but returns empty).
    async with user_scoped_connection(app_pool, user_id=user_b) as conn:
        row_b = await conn.fetchrow("SELECT id FROM extraction_jobs WHERE id = $1", job_id)
    assert row_b is None, "User B must not be able to read user A's job row"
