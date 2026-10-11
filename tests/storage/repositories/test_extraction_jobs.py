"""Repository tests for creating and refreshing queued jobs in extraction_jobs.

Covers create_pending (including the ExtractionJobAlreadyInFlight path) and
refresh_queued. Uses real PostgreSQL with migrations applied through head.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, date, datetime
from typing import Any, cast
from uuid import UUID

import asyncpg
import pytest
from gubbi_common.db.user_scoped import user_scoped_connection

from gubbi.extraction.retry_policy import ErrorCode
from gubbi.storage.repositories.extraction_jobs import (
    ExtractionJobAlreadyInFlight,
    create_pending,
    mark_failed,
    refresh_queued,
)
from tests.storage.repositories.conftest import wait_until_blocked_on_a_lock

pytestmark = pytest.mark.asyncio(loop_scope="session")


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


@pytest.mark.parametrize(
    "source",
    [
        pytest.param("chatgpt", id="test_create_pending_duplicate_raises"),
        pytest.param(
            "extension_chatgpt",
            id="test_create_pending_raises_already_in_flight_for_existing_pending_row",
        ),
    ],
)
async def test_create_pending_raises_already_in_flight_for_an_active_job(
    app_pool: asyncpg.Pool,
    rls_users: tuple[UUID, UUID],
    conversation_id: int,
    source: str,
) -> None:
    """A second create_pending for an active (user, conversation, source) raises.

    The partial unique index fires on the second INSERT; the repository surfaces
    it as ExtractionJobAlreadyInFlight and attaches the existing job_id so callers
    can reuse it for idempotent enqueue semantics.
    """
    user_a, _ = rls_users
    async with user_scoped_connection(app_pool, user_id=user_a) as conn:
        first_id = await create_pending(
            conn, user_a, conversation_id, source, period_start=date(2026, 5, 1)
        )

    async with user_scoped_connection(app_pool, user_id=user_a) as conn:
        with pytest.raises(ExtractionJobAlreadyInFlight) as exc_info:
            await create_pending(
                conn, user_a, conversation_id, source, period_start=date(2026, 5, 1)
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
        )

    async with admin_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT status, period_start, lane, run_after, activity_at"
            " FROM extraction_jobs WHERE id = $1",
            job_id,
        )
    assert row is not None
    assert dict(row) == {
        "status": "pending",
        "period_start": date(2026, 6, 1),
        "lane": "live",
        "run_after": run_after,
        "activity_at": activity_at,
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
            " AS is_current_month, lane, run_after <= now() AS is_runnable, activity_at"
            " FROM extraction_jobs WHERE id = $1",
            job_id,
        )
    assert row is not None
    assert dict(row) == {
        "is_current_month": True,
        "lane": "backfill",
        "is_runnable": True,
        "activity_at": None,
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
        await wait_until_blocked_on_a_lock(admin_pool)
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
}


async def _refresh(conn: asyncpg.Connection, job_id: UUID) -> bool:
    return await refresh_queued(
        conn,
        job_id,
        lane="live",
        run_after=_REFRESH_RUN_AFTER,
        activity_at=_REFRESH_ACTIVITY_AT,
    )


async def _queue_fields(admin_pool: asyncpg.Pool, job_id: UUID) -> dict[str, object]:
    row = await admin_pool.fetchrow(
        "SELECT lane, run_after, activity_at FROM extraction_jobs WHERE id = $1",
        job_id,
    )
    assert row is not None
    return dict(row)


async def test_refresh_queued_rewrites_an_undispatched_pending_job(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    rls_users: tuple[UUID, UUID],
    conversation_id: int,
) -> None:
    user_a, _ = rls_users
    async with user_scoped_connection(app_pool, user_id=user_a) as conn:
        job_id = await create_pending(conn, user_a, conversation_id, "chatgpt", lane="backfill")
        refreshed = await _refresh(conn, job_id)

    assert refreshed is True
    assert await _queue_fields(admin_pool, job_id) == _REFRESHED


@pytest.mark.parametrize(
    "state_sql",
    [
        pytest.param(
            "UPDATE extraction_jobs SET admitted_at = now() WHERE id = $1",
            id="dispatched-pending",
        ),
        pytest.param("UPDATE extraction_jobs SET status = 'running' WHERE id = $1", id="running"),
    ],
)
async def test_refresh_queued_leaves_a_dispatched_or_running_job_unchanged(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    rls_users: tuple[UUID, UUID],
    conversation_id: int,
    state_sql: str,
) -> None:
    user_a, _ = rls_users
    async with user_scoped_connection(app_pool, user_id=user_a) as conn:
        job_id = await create_pending(conn, user_a, conversation_id, "chatgpt", lane="backfill")
    await admin_pool.execute(state_sql, job_id)
    before = await _queue_fields(admin_pool, job_id)

    async with user_scoped_connection(app_pool, user_id=user_a) as conn:
        refreshed = await _refresh(conn, job_id)

    assert refreshed is False
    assert await _queue_fields(admin_pool, job_id) == before
