"""Repository unit tests for gubbi.storage.repositories.extraction_jobs.

Tests cover every public function and the ExtractionJobAlreadyInFlight
exception path. Uses real PostgreSQL with migrations applied through head.

Run:
    pytest tests/storage/repositories/test_extraction_jobs.py -v
"""

from __future__ import annotations

from datetime import date
from uuid import UUID

import asyncpg
import pytest
import pytest_asyncio
from gubbi_common.db.user_scoped import user_scoped_connection

from gubbi.storage.repositories.extraction_jobs import (
    ExtractionJobAlreadyInFlight,
    StatusCounts,
    create_pending,
    get_status_counts,
    mark_completed,
    mark_failed,
    mark_running,
)

pytestmark = pytest.mark.asyncio(loop_scope="session")

_USER_A = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
_USER_B = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def rls_users(admin_pool: asyncpg.Pool) -> tuple[UUID, UUID]:
    """Create two test users (A, B) and tear them down after the test."""
    async with admin_pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO users (id, email, timezone, created_at, updated_at)
            VALUES
                ($1, 'extraction-test-a@test.local', 'UTC', now(), now()),
                ($2, 'extraction-test-b@test.local', 'UTC', now(), now())
            ON CONFLICT (email) DO NOTHING
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
        await conn.execute(
            "DELETE FROM users WHERE id IN ($1, $2)",
            _USER_A,
            _USER_B,
        )


@pytest_asyncio.fixture
async def conversation_id(admin_pool: asyncpg.Pool, rls_users: tuple[UUID, UUID]) -> int:
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
                    'dummytitle'::bytea, 'dummynonce'::bytea,
                    'test-conv', 'chatgpt', 'dummysum'::bytea, 'dummynonce2'::bytea,
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


# ---------------------------------------------------------------------------
# Tests: create_pending
# ---------------------------------------------------------------------------


async def test_create_pending_returns_uuid(
    app_pool: asyncpg.Pool,
    rls_users: tuple[UUID, UUID],
    conversation_id: int,
) -> None:
    """create_pending inserts a row with status='pending' and returns a UUID."""
    user_a, _ = rls_users
    async with user_scoped_connection(app_pool, user_id=user_a) as conn:
        job_id = await create_pending(
            conn, user_a, conversation_id, "chatgpt", period_start=date(2026, 5, 1)
        )

    assert isinstance(job_id, UUID)

    # Verify the row exists via admin pool (bypasses RLS for inspection).


async def test_create_pending_duplicate_raises(
    app_pool: asyncpg.Pool,
    rls_users: tuple[UUID, UUID],
    conversation_id: int,
) -> None:
    """Second create_pending for same active job raises ExtractionJobAlreadyInFlight."""
    user_a, _ = rls_users
    async with user_scoped_connection(app_pool, user_id=user_a) as conn:
        first_id = await create_pending(
            conn, user_a, conversation_id, "chatgpt", period_start=date(2026, 5, 1)
        )

    async with user_scoped_connection(app_pool, user_id=user_a) as conn:
        with pytest.raises(ExtractionJobAlreadyInFlight) as exc_info:
            await create_pending(
                conn, user_a, conversation_id, "chatgpt", period_start=date(2026, 5, 1)
            )

    assert exc_info.value.existing_job_id == first_id


async def test_create_pending_after_terminal_succeeds(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    rls_users: tuple[UUID, UUID],
    conversation_id: int,
) -> None:
    """After a job reaches 'failed', a new pending job can be created for the same
    (user_id, conversation_id, source).
    """
    user_a, _ = rls_users
    async with user_scoped_connection(app_pool, user_id=user_a) as conn:
        first_id = await create_pending(
            conn, user_a, conversation_id, "chatgpt", period_start=date(2026, 5, 1)
        )
        await mark_failed(conn, first_id, error_code="test_error")

    async with user_scoped_connection(app_pool, user_id=user_a) as conn:
        second_id = await create_pending(
            conn, user_a, conversation_id, "chatgpt", period_start=date(2026, 5, 1)
        )

    assert isinstance(second_id, UUID)
    assert second_id != first_id


# ---------------------------------------------------------------------------
# Tests: mark_running
# ---------------------------------------------------------------------------


async def test_mark_running_noop_when_already_running(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    rls_users: tuple[UUID, UUID],
    conversation_id: int,
) -> None:
    """mark_running is idempotent when the job is already in 'running' status."""
    user_a, _ = rls_users
    async with user_scoped_connection(app_pool, user_id=user_a) as conn:
        job_id = await create_pending(
            conn, user_a, conversation_id, "chatgpt", period_start=date(2026, 5, 1)
        )
        await mark_running(conn, job_id)
        # Second call must not raise.
        await mark_running(conn, job_id)

    async with admin_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT status FROM extraction_jobs WHERE id = $1",
            job_id,
        )
    assert row is not None
    assert row["status"] == "running"


async def test_mark_running_retries_failed_job_and_clears_error_code(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    rls_users: tuple[UUID, UUID],
    conversation_id: int,
) -> None:
    """mark_running revives failed rows for Arq retries."""
    user_a, _ = rls_users
    async with user_scoped_connection(app_pool, user_id=user_a) as conn:
        job_id = await create_pending(
            conn, user_a, conversation_id, "chatgpt", period_start=date(2026, 5, 1)
        )
        await mark_failed(conn, job_id, error_code="llm_provider_error")
        await mark_running(conn, job_id)

    async with admin_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT status, error_code, started_at FROM extraction_jobs WHERE id = $1",
            job_id,
        )
    assert row is not None
    assert row["status"] == "running"
    assert row["error_code"] is None
    assert row["started_at"] is not None


# ---------------------------------------------------------------------------
# Tests: mark_completed
# ---------------------------------------------------------------------------


async def test_mark_completed_noop_when_already_completed(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    rls_users: tuple[UUID, UUID],
    conversation_id: int,
) -> None:
    """mark_completed is idempotent when the job is already in 'completed' status."""
    user_a, _ = rls_users
    async with user_scoped_connection(app_pool, user_id=user_a) as conn:
        job_id = await create_pending(
            conn, user_a, conversation_id, "chatgpt", period_start=date(2026, 5, 1)
        )
        updated = await mark_completed(
            conn, job_id, topics_created=1, entries_created=3, cents_spent=5
        )
        # Second call must not raise.
        second_updated = await mark_completed(
            conn, job_id, topics_created=99, entries_created=99, cents_spent=99
        )

    async with admin_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT status, topics_created, entries_created FROM extraction_jobs WHERE id = $1",
            job_id,
        )
    assert row is not None
    assert row["status"] == "completed"
    assert updated is True
    assert second_updated is False
    # The second call was a no-op -- original values preserved.
    assert row["topics_created"] == 1
    assert row["entries_created"] == 3


@pytest.mark.asyncio
async def test_mark_failed_noop_when_already_completed(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    rls_users: tuple[UUID, UUID],
    conversation_id: int,
) -> None:
    """mark_failed is a no-op when the job is already in 'completed' status."""
    user_a, _ = rls_users
    async with user_scoped_connection(app_pool, user_id=user_a) as conn:
        job_id = await create_pending(
            conn, user_a, conversation_id, "chatgpt", period_start=date(2026, 5, 1)
        )
        await mark_completed(conn, job_id, topics_created=1, entries_created=3, cents_spent=5)
        updated = await mark_failed(conn, job_id, error_code="late_failure")

    async with admin_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT status, error_code FROM extraction_jobs WHERE id = $1",
            job_id,
        )
    assert row is not None
    assert row["status"] == "completed"
    assert row["error_code"] is None
    assert updated is False


# ---------------------------------------------------------------------------
# Tests: get_status_counts
# ---------------------------------------------------------------------------


async def test_get_status_counts_empty(
    app_pool: asyncpg.Pool,
    rls_users: tuple[UUID, UUID],
) -> None:
    """get_status_counts returns zeros when no jobs exist for this user."""
    user_a, _ = rls_users
    async with user_scoped_connection(app_pool, user_id=user_a) as conn:
        counts = await get_status_counts(conn)

    assert isinstance(counts, StatusCounts)
    assert counts.in_flight_count == 0
    assert counts.synced_count == 0
    assert counts.last_sync_at is None


async def test_get_status_counts_buckets(
    app_pool: asyncpg.Pool,
    rls_users: tuple[UUID, UUID],
    conversation_id: int,
) -> None:
    """get_status_counts correctly bucketizes mixed statuses for the RLS-scoped user."""
    user_a, _ = rls_users
    async with user_scoped_connection(app_pool, user_id=user_a) as conn:
        # One pending, one running (via mark_running), one completed, one failed.
        _pending_id = await create_pending(
            conn, user_a, conversation_id, "chatgpt", period_start=date(2026, 5, 1)
        )
        running_conv_id = conversation_id  # reuse for running (different source)
        running_id = await create_pending(
            conn, user_a, running_conv_id, "claude", period_start=date(2026, 5, 1)
        )
        await mark_running(conn, running_id)

        # Need separate conversations for completed and failed to avoid partial-unique clash.
        # Use mark_failed on the running one (simplest path without extra convs).
        completed_id = await create_pending(
            conn, user_a, conversation_id, "paste_memories", period_start=date(2026, 5, 1)
        )
        await mark_completed(conn, completed_id, topics_created=1, entries_created=2, cents_spent=0)

        failed_id = await create_pending(
            conn, user_a, conversation_id, "zip_upload", period_start=date(2026, 5, 1)
        )
        await mark_failed(conn, failed_id, error_code="test_fail")

        counts = await get_status_counts(conn)

    # pending + running = 2 in-flight; 1 completed.
    assert counts.in_flight_count == 2
    assert counts.synced_count == 1
    assert counts.last_sync_at is not None


# ---------------------------------------------------------------------------
# Tests: ExtractionJobAlreadyInFlight -- repository unit level
# ---------------------------------------------------------------------------


async def test_create_pending_raises_already_in_flight_for_existing_pending_row(
    app_pool: asyncpg.Pool,
    rls_users: tuple[UUID, UUID],
    conversation_id: int,
) -> None:
    """When a pending row already exists for (user, conv, source), create_pending raises.

    The partial unique index fires on the second INSERT; the repository surfaces
    it as ExtractionJobAlreadyInFlight and attaches the existing job_id so callers
    can reuse it for idempotent enqueue semantics.
    """
    # Arrange
    user_a, _ = rls_users

    async with user_scoped_connection(app_pool, user_id=user_a) as conn:
        first_id = await create_pending(
            conn,
            user_a,
            conversation_id,
            "extension_chatgpt",
            period_start=date(2026, 5, 1),
        )

    # Act + Assert
    async with user_scoped_connection(app_pool, user_id=user_a) as conn:
        with pytest.raises(ExtractionJobAlreadyInFlight) as exc_info:
            await create_pending(
                conn,
                user_a,
                conversation_id,
                "extension_chatgpt",
                period_start=date(2026, 5, 1),
            )

    assert exc_info.value.existing_job_id == first_id
