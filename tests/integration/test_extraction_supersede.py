"""Integration tests: a failed extraction job with a newer job of its conversation is superseded."""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, Any
from uuid import UUID

import asyncpg
import pytest

from gubbi.extraction.retry_policy import ErrorCode
from gubbi.extraction.scheduler import ExtractionScheduler, SchedulerConfig, TickResult
from gubbi.storage.repositories import extraction_queue as queue_repo

if TYPE_CHECKING:
    from tests.fixtures.tenants import TenantSeed

pytestmark = [pytest.mark.asyncio(loop_scope="session"), pytest.mark.integration]

_CONFIG = SchedulerConfig(paused=False, budget_enabled=False, dispatch_bound=20)
_OLDER = timedelta(hours=-2)
_NEWER = timedelta(hours=-1)


class _RecordingArq:
    def __init__(self) -> None:
        self.job_ids: list[UUID] = []

    async def enqueue_job(
        self, function: str, *args: Any, _job_id: str | None = None, **kwargs: Any
    ) -> object:
        self.job_ids.append(UUID(args[2]))
        return object()


async def _insert(
    pool: asyncpg.Pool,
    seed: TenantSeed,
    *,
    status: str,
    created: timedelta,
    error_code: str | None = None,
    source: str = "chatgpt",
) -> UUID:
    """Insert one runnable-if-allowed job on ``seed``'s conversation, created ``created`` from now."""
    job_id: UUID = await pool.fetchval(
        """
        INSERT INTO extraction_jobs
            (user_id, conversation_id, source, status, error_code, run_after, created_at,
             attempts)
        VALUES ($1, $2, $6, $3, $4, now() - interval '1 minute', now() + $5::interval, 1)
        RETURNING id
        """,
        seed.user_id,
        seed.conversation_id,
        status,
        error_code,
        created,
        source,
    )
    return job_id


async def _state(pool: asyncpg.Pool, job_id: UUID) -> tuple[str, str | None, bool]:
    row = await pool.fetchrow(
        "SELECT status, error_code, run_after = 'infinity' AS held FROM extraction_jobs"
        " WHERE id = $1",
        job_id,
    )
    assert row is not None
    return row["status"], row["error_code"], row["held"]


async def _older_worker_lost(pool: asyncpg.Pool, seed: TenantSeed) -> UUID:
    return await _insert(
        pool, seed, status="failed", created=_OLDER, error_code=ErrorCode.WORKER_LOST.value
    )


_SUPERSEDED = ("failed", ErrorCode.SUPERSEDED.value, True)


@pytest.mark.parametrize(
    ("newer_status", "newer_code", "newer_source"),
    [
        pytest.param("failed", ErrorCode.LLM_REFUSED.value, "chatgpt", id="newer-terminal-failure"),
        pytest.param("completed", None, "chatgpt", id="newer-completed"),
        pytest.param("completed", None, "claude", id="newer-completed-other-source"),
    ],
)
async def test_a_failed_job_behind_a_finished_newer_job_is_superseded_not_run(
    clean_rls_db: asyncpg.Pool,
    seeded_a: TenantSeed,
    newer_status: str,
    newer_code: str | None,
    newer_source: str,
) -> None:
    older = await _older_worker_lost(clean_rls_db, seeded_a)
    await _insert(
        clean_rls_db,
        seeded_a,
        status=newer_status,
        created=_NEWER,
        error_code=newer_code,
        source=newer_source,
    )
    arq = _RecordingArq()

    result = await ExtractionScheduler(clean_rls_db, arq, _CONFIG).tick()

    assert result == TickResult(superseded=1)
    assert arq.job_ids == []
    assert await _state(clean_rls_db, older) == _SUPERSEDED
    async with clean_rls_db.acquire() as conn, conn.transaction():
        assert await queue_repo.select_runnable(conn, limit=10, per_user=10) == []


@pytest.mark.parametrize(
    "newer_source",
    [
        pytest.param("chatgpt", id="same-source"),
        pytest.param("claude", id="other-source"),
    ],
)
async def test_a_failed_job_behind_a_pending_newer_job_is_superseded_and_the_newer_one_runs(
    clean_rls_db: asyncpg.Pool, seeded_a: TenantSeed, newer_source: str
) -> None:
    older = await _older_worker_lost(clean_rls_db, seeded_a)
    newer = await _insert(
        clean_rls_db, seeded_a, status="pending", created=_NEWER, source=newer_source
    )
    arq = _RecordingArq()

    result = await ExtractionScheduler(clean_rls_db, arq, _CONFIG).tick()

    assert result == TickResult(admitted=1, superseded=1, enqueued=1)
    assert arq.job_ids == [newer]
    assert await _state(clean_rls_db, older) == _SUPERSEDED


async def test_the_newer_of_two_retryable_failures_still_runs(
    clean_rls_db: asyncpg.Pool, seeded_a: TenantSeed
) -> None:
    older = await _older_worker_lost(clean_rls_db, seeded_a)
    newer = await _insert(
        clean_rls_db,
        seeded_a,
        status="failed",
        created=_NEWER,
        error_code=ErrorCode.WORKER_LOST.value,
    )
    arq = _RecordingArq()

    result = await ExtractionScheduler(clean_rls_db, arq, _CONFIG).tick()

    assert result == TickResult(admitted=1, superseded=1, enqueued=1)
    assert arq.job_ids == [newer]
    assert await _state(clean_rls_db, older) == _SUPERSEDED


async def test_supersede_leaves_a_terminal_failure_and_the_newest_job_untouched(
    clean_rls_db: asyncpg.Pool, seeded_a: TenantSeed
) -> None:
    refused = await _insert(
        clean_rls_db,
        seeded_a,
        status="failed",
        created=_OLDER,
        error_code=ErrorCode.LLM_REFUSED.value,
    )
    newest = await _insert(
        clean_rls_db,
        seeded_a,
        status="failed",
        created=_NEWER,
        error_code=ErrorCode.WORKER_LOST.value,
    )
    before = await clean_rls_db.fetch(
        "SELECT id, xmin::text AS version FROM extraction_jobs ORDER BY id"
    )

    async with clean_rls_db.acquire() as conn:
        changed = await queue_repo.supersede_failed(conn)

    assert changed == 0
    assert await _state(clean_rls_db, refused) == ("failed", ErrorCode.LLM_REFUSED.value, False)
    assert (await _state(clean_rls_db, newest))[1] == ErrorCode.WORKER_LOST.value
    after = await clean_rls_db.fetch(
        "SELECT id, xmin::text AS version FROM extraction_jobs ORDER BY id"
    )
    assert [tuple(row) for row in after] == [tuple(row) for row in before]


async def test_an_active_job_of_another_source_keeps_a_failed_job_of_its_conversation_from_running(
    clean_rls_db: asyncpg.Pool, seeded_a: TenantSeed
) -> None:
    await _insert(clean_rls_db, seeded_a, status="running", created=_OLDER, source="claude")
    failed = await _insert(
        clean_rls_db,
        seeded_a,
        status="failed",
        created=_NEWER,
        error_code=ErrorCode.WORKER_LOST.value,
    )

    async with clean_rls_db.acquire() as conn, conn.transaction():
        runnable = await queue_repo.select_runnable(conn, limit=10, per_user=10)

    assert runnable == []
    assert await _state(clean_rls_db, failed) == ("failed", ErrorCode.WORKER_LOST.value, False)
