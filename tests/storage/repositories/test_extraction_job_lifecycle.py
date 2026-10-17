"""Repository tests for the job lifecycle in extraction_jobs.

Covers mark_running, mark_completed, mark_failed, finished_attempt,
create_follow_up and get_status_counts. Uses real PostgreSQL with migrations
applied through head.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, date, datetime, timedelta
from uuid import UUID

import asyncpg
import pytest
from gubbi_common.db.user_scoped import user_scoped_connection

from gubbi.extraction.retry_policy import DETERMINISTIC_MAX_ATTEMPTS, ErrorCode
from gubbi.storage.repositories.extraction_jobs import (
    StatusCounts,
    create_follow_up,
    create_pending,
    finished_attempt,
    get_status_counts,
    mark_completed,
    mark_failed,
    mark_running,
)
from gubbi.storage.repositories.extraction_queue import admit, revive_failed
from tests.storage.repositories.conftest import wait_until_blocked_on_a_lock

pytestmark = pytest.mark.asyncio(loop_scope="session")


# ---------------------------------------------------------------------------
# Tests: mark_running
# ---------------------------------------------------------------------------


async def _create_admitted(
    conn: asyncpg.Connection, user_id: UUID, conversation_id: int, source: str = "chatgpt"
) -> UUID:
    job_id = await create_pending(conn, user_id, conversation_id, source)
    await admit(conn, [job_id])
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
        await wait_until_blocked_on_a_lock(admin_pool)
    second_claim = await second

    assert (first_claim, second_claim) == (True, False)


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
# Tests: finished_attempt / create_follow_up
# ---------------------------------------------------------------------------


async def _resave(admin_pool: asyncpg.Pool, conversation_id: int, delta: timedelta) -> None:
    """Move the conversation's updated_at to ``delta`` from the job's started_at."""
    await admin_pool.execute(
        "UPDATE conversations SET updated_at ="
        " (SELECT started_at FROM extraction_jobs WHERE conversation_id = $1)"
        " + $2::interval WHERE id = $1",
        conversation_id,
        delta,
    )


@pytest.mark.parametrize(
    ("delta", "resaved"),
    [
        pytest.param(timedelta(seconds=1), True, id="saved-after-start"),
        pytest.param(timedelta(0), False, id="saved-at-start"),
        pytest.param(timedelta(seconds=-1), False, id="saved-before-start"),
    ],
)
async def test_finished_attempt_reports_a_resave_after_the_attempt_started(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    rls_users: tuple[UUID, UUID],
    conversation_id: int,
    delta: timedelta,
    resaved: bool,
) -> None:
    user_a, _ = rls_users
    async with user_scoped_connection(app_pool, user_id=user_a) as conn:
        job_id = await _create_admitted(conn, user_a, conversation_id)
        assert await mark_running(conn, job_id)
    await _resave(admin_pool, conversation_id, delta)

    async with user_scoped_connection(app_pool, user_id=user_a) as conn:
        attempt = await finished_attempt(conn, job_id)

    assert attempt is not None
    assert (attempt.user_id, attempt.attempt, attempt.lane) == (user_a, 1, "backfill")
    assert attempt.resaved_during_run is resaved


@pytest.mark.parametrize(
    ("watermark", "delta", "resaved"),
    [
        pytest.param(2, timedelta(seconds=1), False, id="watermark-at-max-saved-after-start"),
        pytest.param(1, timedelta(seconds=-1), True, id="watermark-below-max-saved-before-start"),
    ],
)
async def test_finished_attempt_reports_a_resave_from_messages_above_the_watermark(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    rls_users: tuple[UUID, UUID],
    conversation_id: int,
    watermark: int,
    delta: timedelta,
    resaved: bool,
) -> None:
    user_a, _ = rls_users
    for position in range(3):
        await _add_message(admin_pool, user_a, conversation_id, None, position)
    await admin_pool.execute(
        "UPDATE conversations SET extracted_through_position = $2 WHERE id = $1",
        conversation_id,
        watermark,
    )
    async with user_scoped_connection(app_pool, user_id=user_a) as conn:
        job_id = await _create_admitted(conn, user_a, conversation_id)
        assert await mark_running(conn, job_id)
    await _resave(admin_pool, conversation_id, delta)

    async with user_scoped_connection(app_pool, user_id=user_a) as conn:
        attempt = await finished_attempt(conn, job_id)

    assert attempt is not None
    assert attempt.resaved_during_run is resaved


async def test_create_follow_up_queues_a_new_job_and_reopens_the_conversation(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    rls_users: tuple[UUID, UUID],
    conversation_id: int,
) -> None:
    user_a, _ = rls_users
    run_after = datetime(2026, 7, 2, 3, tzinfo=UTC)
    async with user_scoped_connection(app_pool, user_id=user_a) as conn:
        job_id = await create_pending(conn, user_a, conversation_id, "chatgpt", lane="live")
        await admit(conn, [job_id])
        assert await mark_running(conn, job_id)
        await conn.execute(
            "UPDATE conversations SET processed_at = now() WHERE id = $1", conversation_id
        )
        assert await mark_completed(
            conn, job_id, topics_created=0, entries_created=0, cents_spent=0
        )
        attempt = await finished_attempt(conn, job_id)
        assert attempt is not None
        follow_up = await create_follow_up(conn, attempt, run_after)

    row = await admin_pool.fetchrow(
        "SELECT j.status, j.lane, j.source, j.run_after, j.admitted_at, c.processed_at"
        " FROM extraction_jobs AS j JOIN conversations AS c ON c.id = j.conversation_id"
        " WHERE j.id = $1",
        follow_up,
    )
    assert follow_up is not None
    assert follow_up != job_id
    assert dict(row) == {
        "status": "pending",
        "lane": "live",
        "source": "chatgpt",
        "run_after": run_after,
        "admitted_at": None,
        "processed_at": None,
    }


_OLD_ACTIVITY = datetime(2026, 1, 1, tzinfo=UTC)
_RESAVED_MESSAGE_AT = datetime(2026, 3, 4, 5, 6, tzinfo=UTC)
_CONVERSATION_UPDATED_AT = datetime(2026, 2, 1, tzinfo=UTC)


async def _add_message(
    admin_pool: asyncpg.Pool,
    user_id: UUID,
    conversation_id: int,
    stamp: datetime | None,
    position: int = 0,
) -> None:
    await admin_pool.execute(
        "INSERT INTO messages (conversation_id, user_id, role, timestamp, position,"
        " content_encrypted, content_nonce)"
        " VALUES ($1, $2, 'user', $3, $4, 'x'::bytea, 'nonce_msg_12'::bytea)",
        conversation_id,
        user_id,
        stamp,
        position,
    )


async def _follow_up_activity_at(
    app_pool: asyncpg.Pool, admin_pool: asyncpg.Pool, user_id: UUID, conversation_id: int
) -> datetime:
    async with user_scoped_connection(app_pool, user_id=user_id) as conn:
        job_id = await create_pending(
            conn, user_id, conversation_id, "chatgpt", activity_at=_OLD_ACTIVITY
        )
        await admit(conn, [job_id])
        assert await mark_running(conn, job_id)
        assert await mark_completed(
            conn, job_id, topics_created=0, entries_created=0, cents_spent=0
        )
        attempt = await finished_attempt(conn, job_id)
        assert attempt is not None
        follow_up = await create_follow_up(conn, attempt, datetime(2026, 7, 2, tzinfo=UTC))
    activity_at: datetime = await admin_pool.fetchval(
        "SELECT activity_at FROM extraction_jobs WHERE id = $1", follow_up
    )
    return activity_at


@pytest.mark.parametrize(
    ("stamps", "expected"),
    [
        pytest.param(
            [_RESAVED_MESSAGE_AT - timedelta(days=1), _RESAVED_MESSAGE_AT, None],
            _RESAVED_MESSAGE_AT,
            id="latest-message",
        ),
        pytest.param([None], _CONVERSATION_UPDATED_AT, id="undated-messages-use-updated-at"),
        pytest.param([], _CONVERSATION_UPDATED_AT, id="no-messages-use-updated-at"),
    ],
)
async def test_create_follow_up_takes_activity_at_from_the_conversation_as_saved_now(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    rls_users: tuple[UUID, UUID],
    conversation_id: int,
    stamps: list[datetime | None],
    expected: datetime,
) -> None:
    user_a, _ = rls_users
    await admin_pool.execute(
        "UPDATE conversations SET updated_at = $2 WHERE id = $1",
        conversation_id,
        _CONVERSATION_UPDATED_AT,
    )
    for stamp in stamps:
        await _add_message(admin_pool, user_a, conversation_id, stamp)

    activity_at = await _follow_up_activity_at(app_pool, admin_pool, user_a, conversation_id)

    assert activity_at == expected


async def test_create_follow_up_never_takes_an_activity_at_later_than_now(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    rls_users: tuple[UUID, UUID],
    conversation_id: int,
) -> None:
    user_a, _ = rls_users
    await _add_message(admin_pool, user_a, conversation_id, datetime(2999, 1, 1, tzinfo=UTC))
    before: datetime = await admin_pool.fetchval("SELECT now()")

    activity_at = await _follow_up_activity_at(app_pool, admin_pool, user_a, conversation_id)

    after: datetime = await admin_pool.fetchval("SELECT now()")
    assert before <= activity_at <= after


async def test_create_follow_up_returns_none_when_the_conversation_already_has_an_active_job(
    app_pool: asyncpg.Pool,
    rls_users: tuple[UUID, UUID],
    conversation_id: int,
) -> None:
    user_a, _ = rls_users
    async with user_scoped_connection(app_pool, user_id=user_a) as conn:
        job_id = await _create_admitted(conn, user_a, conversation_id)
        assert await mark_running(conn, job_id)
        attempt = await finished_attempt(conn, job_id)
        assert attempt is not None
        follow_up = await create_follow_up(conn, attempt, datetime(2026, 7, 2, tzinfo=UTC))

    assert follow_up is None


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
