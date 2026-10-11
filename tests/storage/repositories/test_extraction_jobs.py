"""Repository unit tests for gubbi.storage.repositories.extraction_jobs.

Tests cover every public function and the ExtractionJobAlreadyInFlight
exception path. Uses real PostgreSQL with migrations applied through head.

Run:
    pytest tests/storage/repositories/test_extraction_jobs.py -v
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, date, datetime
from typing import Any, cast
from uuid import UUID

import asyncpg
import pytest
import pytest_asyncio
from gubbi_common.db.user_scoped import user_scoped_connection

from gubbi.extraction.intake import TOKEN_COUNT_VERSION
from gubbi.extraction.retry_policy import DETERMINISTIC_MAX_ATTEMPTS, ErrorCode
from gubbi.storage.repositories.extraction_jobs import (
    ExtractionJobAlreadyInFlight,
    StatusCounts,
    create_pending,
    get_status_counts,
    mark_completed,
    mark_failed,
    mark_running,
    refresh_queued,
)
from gubbi.storage.repositories.extraction_queue import admit, revive_failed

pytestmark = pytest.mark.asyncio(loop_scope="session")

_USER_A = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
_USER_B = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
_PERIOD = date(2026, 5, 1)


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
        await mark_failed(conn, first_id, error_code=ErrorCode.INTERNAL_ERROR)

    async with user_scoped_connection(app_pool, user_id=user_a) as conn:
        second_id = await create_pending(
            conn, user_a, conversation_id, "chatgpt", period_start=date(2026, 5, 1)
        )

    assert isinstance(second_id, UUID)
    assert second_id != first_id


async def test_create_pending_stores_queue_fields(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    rls_users: tuple[UUID, UUID],
    conversation_id: int,
) -> None:
    """Every queue field passed to create_pending lands on the row."""
    user_a, _ = rls_users
    run_after = datetime(2026, 6, 2, 3, 0, tzinfo=UTC)
    activity_at = datetime(2026, 6, 1, 18, 30, tzinfo=UTC)
    async with user_scoped_connection(app_pool, user_id=user_a) as conn:
        job_id = await create_pending(
            conn,
            user_a,
            conversation_id,
            "extension_chatgpt",
            period_start=date(2026, 6, 1),
            lane="live",
            run_after=run_after,
            activity_at=activity_at,
            filtered_tokens=1234,
            token_count_version=1,
        )

    async with admin_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT status, period_start, lane, run_after, activity_at, filtered_tokens,"
            " token_count_version FROM extraction_jobs WHERE id = $1",
            job_id,
        )
    assert row is not None
    assert dict(row) == {
        "status": "pending",
        "period_start": date(2026, 6, 1),
        "lane": "live",
        "run_after": run_after,
        "activity_at": activity_at,
        "filtered_tokens": 1234,
        "token_count_version": 1,
    }


async def test_create_pending_omitted_fields_take_column_defaults(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    rls_users: tuple[UUID, UUID],
    conversation_id: int,
) -> None:
    """Fields left out of create_pending get the column defaults, period_start included."""
    user_a, _ = rls_users
    async with user_scoped_connection(app_pool, user_id=user_a) as conn:
        job_id = await create_pending(conn, user_a, conversation_id, "extension_chatgpt")

    async with admin_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT period_start = date_trunc('month', now() AT TIME ZONE 'UTC')::date"
            " AS is_current_month, lane, run_after <= now() AS is_runnable, activity_at,"
            " filtered_tokens, token_count_version FROM extraction_jobs WHERE id = $1",
            job_id,
        )
    assert row is not None
    assert dict(row) == {
        "is_current_month": True,
        "lane": "backfill",
        "is_runnable": True,
        "activity_at": None,
        "filtered_tokens": 0,
        "token_count_version": None,
    }


class _LookupHookConnection:
    """Delegates to a real connection; runs ``on_lookup`` before each in-flight lookup.

    ``on_lookup`` returns whether the lookup should report no active row.
    """

    def __init__(self, conn: asyncpg.Connection, on_lookup: Callable[[], Awaitable[bool]]) -> None:
        self._conn = conn
        self._on_lookup = on_lookup
        self.inserts = 0

    def transaction(self) -> Any:
        return self._conn.transaction()

    async def fetchrow(self, query: str, *args: object) -> asyncpg.Record | None:
        if query.lstrip().startswith("INSERT"):
            self.inserts += 1
        elif await self._on_lookup():
            return None
        return await self._conn.fetchrow(query, *args)


async def test_create_pending_retries_when_the_conflicting_job_finished_mid_recovery(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    rls_users: tuple[UUID, UUID],
    conversation_id: int,
) -> None:
    user_a, _ = rls_users
    async with user_scoped_connection(app_pool, user_id=user_a) as conn:
        first_id = await create_pending(conn, user_a, conversation_id, "chatgpt")

    async def finish_first_job() -> bool:
        await admin_pool.execute(
            "UPDATE extraction_jobs SET status = 'completed' WHERE id = $1", first_id
        )
        return False

    async with user_scoped_connection(app_pool, user_id=user_a) as conn:
        hooked = _LookupHookConnection(conn, finish_first_job)
        second_id = await create_pending(
            cast("asyncpg.Connection", hooked), user_a, conversation_id, "chatgpt"
        )

    assert second_id != first_id
    assert hooked.inserts == 2


async def test_create_pending_gives_up_when_no_active_row_is_ever_found(
    app_pool: asyncpg.Pool,
    rls_users: tuple[UUID, UUID],
    conversation_id: int,
) -> None:
    user_a, _ = rls_users
    async with user_scoped_connection(app_pool, user_id=user_a) as conn:
        await create_pending(conn, user_a, conversation_id, "chatgpt")

    async def report_no_active_row() -> bool:
        return True

    async with user_scoped_connection(app_pool, user_id=user_a) as conn:
        hooked = _LookupHookConnection(conn, report_no_active_row)
        with pytest.raises(RuntimeError, match="no active row found after 3 attempts"):
            await create_pending(
                cast("asyncpg.Connection", hooked), user_a, conversation_id, "chatgpt"
            )

    assert hooked.inserts == 3


async def test_reused_job_cannot_finish_until_the_reusing_transaction_ends(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    rls_users: tuple[UUID, UUID],
    conversation_id: int,
) -> None:
    """The in-flight lookup locks the reused row, so a completion waits for the commit."""
    user_a, _ = rls_users
    async with user_scoped_connection(app_pool, user_id=user_a) as conn:
        job_id = await create_pending(conn, user_a, conversation_id, "chatgpt")

    async def _complete() -> str:
        return str(
            await admin_pool.execute(
                "UPDATE extraction_jobs SET status = 'completed' WHERE id = $1", job_id
            )
        )

    async with user_scoped_connection(app_pool, user_id=user_a) as conn:
        with pytest.raises(ExtractionJobAlreadyInFlight) as reused:
            await create_pending(conn, user_a, conversation_id, "chatgpt")
        completion = asyncio.create_task(_complete())
        await _wait_until_blocked_on_a_lock(admin_pool)
        refreshed = await _refresh(conn, reused.value.existing_job_id)
        completed_before_commit = completion.done()

    assert (reused.value.existing_job_id, refreshed) == (job_id, True)
    assert completed_before_commit is False
    assert await completion == "UPDATE 1"


# ---------------------------------------------------------------------------
# Tests: refresh_queued
# ---------------------------------------------------------------------------

_REFRESH_RUN_AFTER = datetime(2026, 7, 2, 3, 0, tzinfo=UTC)
_REFRESH_ACTIVITY_AT = datetime(2026, 7, 1, 21, 0, tzinfo=UTC)
_REFRESHED = {
    "lane": "live",
    "run_after": _REFRESH_RUN_AFTER,
    "activity_at": _REFRESH_ACTIVITY_AT,
    "filtered_tokens": 5000,
    "token_count_version": 2,
}


async def _refresh(conn: asyncpg.Connection, job_id: UUID) -> bool:
    return await refresh_queued(
        conn,
        job_id,
        lane="live",
        run_after=_REFRESH_RUN_AFTER,
        activity_at=_REFRESH_ACTIVITY_AT,
        filtered_tokens=5000,
        token_count_version=2,
    )


async def _queue_fields(admin_pool: asyncpg.Pool, job_id: UUID) -> dict[str, object]:
    row = await admin_pool.fetchrow(
        "SELECT lane, run_after, activity_at, filtered_tokens, token_count_version"
        " FROM extraction_jobs WHERE id = $1",
        job_id,
    )
    assert row is not None
    return dict(row)


async def test_refresh_queued_rewrites_unadmitted_pending_job(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    rls_users: tuple[UUID, UUID],
    conversation_id: int,
) -> None:
    user_a, _ = rls_users
    async with user_scoped_connection(app_pool, user_id=user_a) as conn:
        job_id = await create_pending(
            conn, user_a, conversation_id, "chatgpt", lane="backfill", filtered_tokens=10
        )
        refreshed = await _refresh(conn, job_id)

    assert refreshed is True
    assert await _queue_fields(admin_pool, job_id) == _REFRESHED


@pytest.mark.parametrize(
    "state_sql",
    [
        pytest.param(
            "UPDATE extraction_jobs SET admitted_at = now(), token_count_version = 1,"
            " budget_period_start = date '2026-07-01' WHERE id = $1",
            id="admitted-pending",
        ),
        pytest.param("UPDATE extraction_jobs SET status = 'running' WHERE id = $1", id="running"),
    ],
)
async def test_refresh_queued_leaves_admitted_or_running_job_unchanged(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    rls_users: tuple[UUID, UUID],
    conversation_id: int,
    state_sql: str,
) -> None:
    user_a, _ = rls_users
    async with user_scoped_connection(app_pool, user_id=user_a) as conn:
        job_id = await create_pending(
            conn, user_a, conversation_id, "chatgpt", lane="backfill", filtered_tokens=10
        )
    await admin_pool.execute(state_sql, job_id)
    before = await _queue_fields(admin_pool, job_id)

    async with user_scoped_connection(app_pool, user_id=user_a) as conn:
        refreshed = await _refresh(conn, job_id)

    assert refreshed is False
    assert await _queue_fields(admin_pool, job_id) == before


# ---------------------------------------------------------------------------
# Tests: mark_running
# ---------------------------------------------------------------------------


async def _create_admitted(
    conn: asyncpg.Connection, user_id: UUID, conversation_id: int, source: str = "chatgpt"
) -> UUID:
    job_id = await create_pending(
        conn, user_id, conversation_id, source, token_count_version=TOKEN_COUNT_VERSION
    )
    await admit(conn, [job_id], _PERIOD)
    return job_id


async def _revive(conn: asyncpg.Connection, job_id: UUID) -> None:
    """Move a failed row back to pending with its backoff elapsed, as the scheduler does."""
    assert await revive_failed(conn, [job_id]) == {job_id}
    await conn.execute(
        "UPDATE extraction_jobs SET run_after = now() - interval '1 minute' WHERE id = $1", job_id
    )


async def _claim_state(admin_pool: asyncpg.Pool, job_id: UUID) -> tuple[str, int]:
    async with admin_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT status, attempts FROM extraction_jobs WHERE id = $1", job_id
        )
    assert row is not None
    return row["status"], row["attempts"]


async def test_mark_running_claims_an_admitted_pending_job_once(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    rls_users: tuple[UUID, UUID],
    conversation_id: int,
) -> None:
    user_a, _ = rls_users
    async with user_scoped_connection(app_pool, user_id=user_a) as conn:
        job_id = await _create_admitted(conn, user_a, conversation_id)
        first = await mark_running(conn, job_id)
        second = await mark_running(conn, job_id)

    assert (first, second) == (True, False)
    assert await _claim_state(admin_pool, job_id) == ("running", 1)


async def _unadmitted(conn: asyncpg.Connection, user_id: UUID, conversation_id: int) -> UUID:
    return await create_pending(conn, user_id, conversation_id, "chatgpt")


async def _failed_past_backoff(
    conn: asyncpg.Connection, user_id: UUID, conversation_id: int
) -> UUID:
    job_id = await _create_admitted(conn, user_id, conversation_id)
    await mark_running(conn, job_id)
    await mark_failed(conn, job_id, error_code=ErrorCode.LLM_PROVIDER_ERROR)
    await conn.execute(
        "UPDATE extraction_jobs SET run_after = now() - interval '1 minute' WHERE id = $1", job_id
    )
    return job_id


async def _backed_off(conn: asyncpg.Connection, user_id: UUID, conversation_id: int) -> UUID:
    job_id = await _create_admitted(conn, user_id, conversation_id)
    await conn.execute(
        "UPDATE extraction_jobs SET run_after = now() + interval '1 hour' WHERE id = $1", job_id
    )
    return job_id


async def _completed(conn: asyncpg.Connection, user_id: UUID, conversation_id: int) -> UUID:
    job_id = await _create_admitted(conn, user_id, conversation_id)
    await mark_completed(conn, job_id, topics_created=0, entries_created=0, cents_spent=0)
    return job_id


@pytest.mark.parametrize(
    "make_job",
    [
        pytest.param(_unadmitted, id="pending-not-admitted"),
        pytest.param(_failed_past_backoff, id="failed"),
        pytest.param(_backed_off, id="run-after-in-the-future"),
        pytest.param(_completed, id="completed"),
    ],
)
async def test_mark_running_does_not_claim_a_row_the_scheduler_has_not_released(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    rls_users: tuple[UUID, UUID],
    conversation_id: int,
    make_job: Callable[[asyncpg.Connection, UUID, int], Awaitable[UUID]],
) -> None:
    """Each row fails exactly one claim condition; a delivery of it changes nothing."""
    user_a, _ = rls_users
    async with user_scoped_connection(app_pool, user_id=user_a) as conn:
        job_id = await make_job(conn, user_a, conversation_id)
    before = await _claim_state(admin_pool, job_id)

    async with user_scoped_connection(app_pool, user_id=user_a) as conn:
        claimed = await mark_running(conn, job_id)

    assert claimed is False
    assert await _claim_state(admin_pool, job_id) == before


async def test_mark_running_claims_a_revived_retry_and_clears_error_code(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    rls_users: tuple[UUID, UUID],
    conversation_id: int,
) -> None:
    user_a, _ = rls_users
    async with user_scoped_connection(app_pool, user_id=user_a) as conn:
        job_id = await _create_admitted(conn, user_a, conversation_id)
        await mark_running(conn, job_id)
        await mark_failed(conn, job_id, error_code=ErrorCode.LLM_PROVIDER_ERROR)
        await _revive(conn, job_id)
        claimed = await mark_running(conn, job_id)

    async with admin_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT status, error_code, attempts FROM extraction_jobs WHERE id = $1", job_id
        )
    assert claimed is True
    assert row is not None
    assert (row["status"], row["error_code"], row["attempts"]) == ("running", None, 2)


async def test_a_claim_waiting_on_an_uncommitted_claim_does_not_also_claim(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    rls_users: tuple[UUID, UUID],
    conversation_id: int,
) -> None:
    """The second claim blocks on the first's row lock, then sees 'running' and backs off."""
    user_a, _ = rls_users
    async with user_scoped_connection(app_pool, user_id=user_a) as conn:
        job_id = await _create_admitted(conn, user_a, conversation_id)

    async def _second_claim() -> bool:
        async with user_scoped_connection(app_pool, user_id=user_a) as conn:
            return await mark_running(conn, job_id)

    async with user_scoped_connection(app_pool, user_id=user_a) as first:
        first_claim = await mark_running(first, job_id)
        second = asyncio.create_task(_second_claim())
        await _wait_until_blocked_on_a_lock(admin_pool)
    second_claim = await second

    assert (first_claim, second_claim) == (True, False)


async def _wait_until_blocked_on_a_lock(admin_pool: asyncpg.Pool) -> None:
    for _ in range(500):
        if await admin_pool.fetchval("SELECT count(*) FROM pg_locks WHERE NOT granted"):
            return
        await asyncio.sleep(0.01)
    raise AssertionError("the second claim never waited on the first's lock")


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
        updated = await mark_failed(conn, job_id, error_code=ErrorCode.INTERNAL_ERROR)

    async with admin_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT status, error_code FROM extraction_jobs WHERE id = $1",
            job_id,
        )
    assert row is not None
    assert row["status"] == "completed"
    assert row["error_code"] is None
    assert updated is None


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
        running_id = await _create_admitted(conn, user_a, running_conv_id, "claude")
        assert await mark_running(conn, running_id)

        # Need separate conversations for completed and failed to avoid partial-unique clash.
        # Use mark_failed on the running one (simplest path without extra convs).
        completed_id = await create_pending(
            conn, user_a, conversation_id, "paste_memories", period_start=date(2026, 5, 1)
        )
        await mark_completed(conn, completed_id, topics_created=1, entries_created=2, cents_spent=0)

        failed_id = await create_pending(
            conn, user_a, conversation_id, "zip_upload", period_start=date(2026, 5, 1)
        )
        await mark_failed(conn, failed_id, error_code=ErrorCode.INTERNAL_ERROR)

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


# ---------------------------------------------------------------------------
# Tests: attempts and retry scheduling
# ---------------------------------------------------------------------------


async def _job_retry_state(admin_pool: asyncpg.Pool, job_id: UUID) -> asyncpg.Record:
    async with admin_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT attempts, error_code, run_after = 'infinity' AS held,"
            " run_after > now() AS deferred FROM extraction_jobs WHERE id = $1",
            job_id,
        )
    assert row is not None
    return row


async def test_every_claim_counts_one_attempt(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    rls_users: tuple[UUID, UUID],
    conversation_id: int,
) -> None:
    user_a, _ = rls_users
    async with user_scoped_connection(app_pool, user_id=user_a) as conn:
        job_id = await _create_admitted(conn, user_a, conversation_id)
        await mark_running(conn, job_id)
        await mark_failed(conn, job_id, error_code=ErrorCode.LLM_PROVIDER_ERROR)
        await _revive(conn, job_id)
        await mark_running(conn, job_id)
        await mark_running(conn, job_id)

    assert (await _job_retry_state(admin_pool, job_id))["attempts"] == 2


@pytest.mark.parametrize(
    ("code", "held", "deferred"),
    [
        pytest.param(ErrorCode.LLM_REFUSED, True, True, id="terminal-held"),
        pytest.param(ErrorCode.LLM_RATE_LIMITED, False, True, id="transient-backoff"),
    ],
)
async def test_mark_failed_schedules_the_next_run_by_error_class(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    rls_users: tuple[UUID, UUID],
    conversation_id: int,
    code: ErrorCode,
    held: bool,
    deferred: bool,
) -> None:
    user_a, _ = rls_users
    async with user_scoped_connection(app_pool, user_id=user_a) as conn:
        job_id = await _create_admitted(conn, user_a, conversation_id)
        assert await mark_running(conn, job_id)
        await mark_failed(conn, job_id, error_code=code)

    row = await _job_retry_state(admin_pool, job_id)
    assert (row["error_code"], row["held"], row["deferred"]) == (code.value, held, deferred)


async def test_mark_failed_holds_a_deterministic_failure_after_its_last_attempt(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    rls_users: tuple[UUID, UUID],
    conversation_id: int,
) -> None:
    user_a, _ = rls_users
    async with user_scoped_connection(app_pool, user_id=user_a) as conn:
        job_id = await _create_admitted(conn, user_a, conversation_id)
        for attempt in range(DETERMINISTIC_MAX_ATTEMPTS):
            if attempt:
                await _revive(conn, job_id)
            assert await mark_running(conn, job_id)
            await mark_failed(conn, job_id, error_code=ErrorCode.LLM_SCHEMA)

    row = await _job_retry_state(admin_pool, job_id)
    assert (row["attempts"], row["held"]) == (DETERMINISTIC_MAX_ATTEMPTS, True)
