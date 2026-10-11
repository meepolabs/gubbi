"""Integration tests for the extraction queue scheduler against a real database.

The Arq pool is a fake that records every enqueue; the admin pool is the
BYPASSRLS test pool the scheduler uses in production.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from datetime import timedelta
from typing import TYPE_CHECKING, Any
from uuid import UUID

import asyncpg
import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

from gubbi.config import get_settings
from gubbi.extraction.retry_policy import ErrorCode
from gubbi.extraction.scheduler import (
    EXTRACT_CONVERSATION_JOB,
    ExtractionScheduler,
    SchedulerConfig,
    TickResult,
)

if TYPE_CHECKING:
    from tests.fixtures.tenants import TenantSeed

pytestmark = [pytest.mark.asyncio(loop_scope="session"), pytest.mark.integration]

_CONFIG = SchedulerConfig(paused=False, budget_enabled=False, dispatch_bound=20)


@dataclass
class FakeArq:
    """Records enqueue calls; ``hook`` runs before each one and may raise or block."""

    calls: list[tuple[str, tuple[Any, ...], str | None]] = field(default_factory=list)
    hook: Callable[[str | None], Any] | None = None

    async def enqueue_job(
        self, function: str, *args: Any, _job_id: str | None = None, **kwargs: Any
    ) -> object:
        if self.hook is not None:
            await self.hook(_job_id)
        self.calls.append((function, args, _job_id))
        return object()

    @property
    def job_ids(self) -> list[UUID]:
        return [UUID(args[2]) for _, args, _ in self.calls]


@pytest.fixture(autouse=True)
def _fresh_settings() -> Iterator[None]:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


async def _insert_jobs(
    pool: asyncpg.Pool,
    seed: TenantSeed,
    count: int,
    *,
    lane: str = "live",
    status: str = "pending",
    error_code: str | None = None,
    activity_offset: timedelta = timedelta(0),
    admitted: bool = False,
    attempts: int = 0,
) -> list[UUID]:
    """Insert ``count`` runnable jobs, each on its own conversation, newest activity first."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            WITH convs AS (
                INSERT INTO conversations
                    (topic_id, user_id, title_encrypted, title_nonce, slug, source,
                     summary_encrypted, summary_nonce, message_count, json_path)
                SELECT $1, $2, 'title'::bytea, 'nonce_title1'::bytea,
                       'conv-' || gen_random_uuid(), 'chatgpt',
                       'summary'::bytea, 'nonce_summ12'::bytea, 0, 'x.json'
                FROM generate_series(1, $3) AS g(n)
                RETURNING id
            ), numbered AS (
                SELECT id, row_number() OVER (ORDER BY id) AS n FROM convs
            )
            INSERT INTO extraction_jobs
                (user_id, conversation_id, source, status, error_code, lane, run_after,
                 activity_at, filtered_tokens, admitted_at, budget_period_start, attempts)
            SELECT $2, numbered.id, 'chatgpt', $4, $5, $6, now() - interval '1 minute',
                   now() + $7::interval - numbered.n * interval '1 second', 100,
                   CASE WHEN $8 THEN now() - interval '1 hour' END,
                   CASE WHEN $8 THEN date_trunc('month', now())::date END,
                   $9
            FROM numbered
            ORDER BY numbered.n
            RETURNING id
            """,
            seed.topic_id,
            seed.user_id,
            count,
            status,
            error_code,
            lane,
            activity_offset,
            admitted,
            attempts,
        )
    return [UUID(str(row["id"])) for row in rows]


async def _row(pool: asyncpg.Pool, job_id: UUID) -> asyncpg.Record:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT status, admitted_at, defer_reason, error_code, run_after > now() AS backed_off"
            " FROM extraction_jobs WHERE id = $1",
            job_id,
        )
    assert row is not None
    return row


async def _admitted_count(pool: asyncpg.Pool) -> int:
    async with pool.acquire() as conn:
        return int(
            await conn.fetchval(
                "SELECT count(*) FROM extraction_jobs WHERE admitted_at IS NOT NULL"
            )
        )


async def test_a_paused_scheduler_enqueues_and_admits_nothing(
    clean_rls_db: asyncpg.Pool, seeded_a: TenantSeed
) -> None:
    await _insert_jobs(clean_rls_db, seeded_a, 3)
    arq = FakeArq()
    scheduler = ExtractionScheduler(clean_rls_db, arq, replace(_CONFIG, paused=True))

    result = await scheduler.tick()

    assert result is None
    assert arq.calls == []
    assert await _admitted_count(clean_rls_db) == 0


async def test_a_tick_admits_and_enqueues_with_the_worker_call_shape(
    clean_rls_db: asyncpg.Pool, seeded_a: TenantSeed
) -> None:
    (job_id,) = await _insert_jobs(clean_rls_db, seeded_a, 1)
    async with clean_rls_db.acquire() as conn:
        conversation_id = await conn.fetchval(
            "SELECT conversation_id FROM extraction_jobs WHERE id = $1", job_id
        )
    arq = FakeArq()

    result = await ExtractionScheduler(clean_rls_db, arq, _CONFIG).tick()

    assert result == TickResult(admitted=1, enqueued=1)
    assert arq.calls == [
        (
            EXTRACT_CONVERSATION_JOB,
            (conversation_id, str(seeded_a.user_id), str(job_id)),
            f"{job_id}:0",
        )
    ]
    row = await _row(clean_rls_db, job_id)
    assert (row["status"], row["admitted_at"] is not None) == ("pending", True)


async def test_only_one_of_two_scheduler_instances_ticks_at_a_time(
    clean_rls_db: asyncpg.Pool, seeded_a: TenantSeed
) -> None:
    await _insert_jobs(clean_rls_db, seeded_a, 1)
    inside, release = asyncio.Event(), asyncio.Event()

    async def _block(_job_id: str | None) -> None:
        inside.set()
        await release.wait()

    first_arq, second_arq = FakeArq(hook=_block), FakeArq()
    first = ExtractionScheduler(clean_rls_db, first_arq, _CONFIG)
    second = ExtractionScheduler(clean_rls_db, second_arq, _CONFIG)

    first_tick = asyncio.create_task(first.tick())
    await asyncio.wait_for(inside.wait(), 5)
    (late_job,) = await _insert_jobs(clean_rls_db, seeded_a, 1)
    second_result = await second.tick()
    release.set()
    first_result = await asyncio.wait_for(first_tick, 5)

    assert second_result is None
    assert second_arq.calls == []
    assert (await _row(clean_rls_db, late_job))["admitted_at"] is None
    assert first_result == TickResult(admitted=1, enqueued=1)


async def test_the_dispatch_bound_holds_and_later_live_work_overtakes_backfill(
    clean_rls_db: asyncpg.Pool, seeded_a: TenantSeed
) -> None:
    await _insert_jobs(clean_rls_db, seeded_a, 1000, lane="backfill")
    arq = FakeArq()
    scheduler = ExtractionScheduler(clean_rls_db, arq, _CONFIG)

    first = await scheduler.tick()
    (live_job,) = await _insert_jobs(
        clean_rls_db, seeded_a, 1, lane="live", activity_offset=timedelta(days=-30)
    )
    saturated = await scheduler.tick()
    async with clean_rls_db.acquire() as conn:
        await conn.execute(
            "UPDATE extraction_jobs SET status = 'completed' WHERE id = ANY($1::uuid[])",
            arq.job_ids[:5],
        )
    calls_before_drain = len(arq.calls)
    drained = await scheduler.tick()

    assert first == TickResult(admitted=20, enqueued=20)
    assert saturated == TickResult()
    assert drained == TickResult(admitted=5, enqueued=5)
    assert arq.job_ids[calls_before_drain] == live_job
    assert await _admitted_count(clean_rls_db) == 25


async def test_a_retryable_failed_row_runs_again_without_a_second_charge(
    clean_rls_db: asyncpg.Pool, seeded_a: TenantSeed
) -> None:
    (job_id,) = await _insert_jobs(
        clean_rls_db,
        seeded_a,
        1,
        status="failed",
        error_code=ErrorCode.WORKER_LOST.value,
        admitted=True,
        attempts=1,
    )
    before = (await _row(clean_rls_db, job_id))["admitted_at"]
    arq = FakeArq()

    result = await ExtractionScheduler(clean_rls_db, arq, _CONFIG).tick()

    row = await _row(clean_rls_db, job_id)
    assert result == TickResult(enqueued=1)
    assert [call[2] for call in arq.calls] == [f"{job_id}:1"]
    assert (row["status"], row["admitted_at"]) == ("pending", before)


async def test_over_budget_work_is_deferred_with_a_reason_not_dispatched(
    clean_rls_db: asyncpg.Pool, seeded_a: TenantSeed, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("JOURNAL_EXTRACTION_MONTHLY_CONVERSATIONS", "1")
    newest, older = await _insert_jobs(clean_rls_db, seeded_a, 2)
    arq = FakeArq()
    config = replace(_CONFIG, budget_enabled=True)

    result = await ExtractionScheduler(clean_rls_db, arq, config).tick()

    assert result == TickResult(admitted=1, deferred=1, enqueued=1)
    assert arq.job_ids == [newest]
    deferred = await _row(clean_rls_db, older)
    assert (deferred["status"], deferred["admitted_at"], deferred["defer_reason"]) == (
        "pending",
        None,
        "budget",
    )


async def test_a_lost_arq_job_is_re_enqueued_once_stale(
    clean_rls_db: asyncpg.Pool, seeded_a: TenantSeed
) -> None:
    (job_id,) = await _insert_jobs(clean_rls_db, seeded_a, 1, admitted=True)
    arq = FakeArq()

    result = await ExtractionScheduler(clean_rls_db, arq, _CONFIG).tick()

    assert result == TickResult(reenqueued=1)
    assert arq.job_ids == [job_id]


async def _failing_enqueue(_job_id: str | None) -> None:
    raise RedisConnectionError("redis down")


async def test_repeated_enqueue_failures_fail_the_row_as_enqueue_lost(
    clean_rls_db: asyncpg.Pool, seeded_a: TenantSeed
) -> None:
    (job_id,) = await _insert_jobs(clean_rls_db, seeded_a, 1)
    config = replace(_CONFIG, stale_after=timedelta(0), max_enqueue_failures=3)
    scheduler = ExtractionScheduler(clean_rls_db, FakeArq(hook=_failing_enqueue), config)

    first = await scheduler.tick()
    after_first = await _row(clean_rls_db, job_id)
    await scheduler.tick()
    third = await scheduler.tick()

    assert (first, after_first["status"]) == (TickResult(admitted=1, enqueue_failed=1), "pending")
    assert third == TickResult(enqueue_failed=1, enqueue_lost=1)
    row = await _row(clean_rls_db, job_id)
    assert (row["status"], row["error_code"], row["backed_off"]) == (
        "failed",
        ErrorCode.ENQUEUE_LOST.value,
        True,
    )


async def test_a_row_a_worker_claimed_is_not_failed_as_enqueue_lost(
    clean_rls_db: asyncpg.Pool, seeded_a: TenantSeed
) -> None:
    (job_id,) = await _insert_jobs(clean_rls_db, seeded_a, 1, admitted=True)

    async def _claim_then_fail(_job_id: str | None) -> None:
        async with clean_rls_db.acquire() as conn:
            await conn.execute(
                "UPDATE extraction_jobs SET status = 'running' WHERE id = $1", job_id
            )
        raise RedisConnectionError("redis down")

    config = replace(_CONFIG, max_enqueue_failures=1)
    result = await ExtractionScheduler(clean_rls_db, FakeArq(hook=_claim_then_fail), config).tick()

    assert result == TickResult(enqueue_failed=1)
    assert (await _row(clean_rls_db, job_id))["status"] == "running"


async def test_a_failed_row_stays_failed_while_its_conversation_has_an_active_job(
    clean_rls_db: asyncpg.Pool, seeded_a: TenantSeed
) -> None:
    (active,) = await _insert_jobs(clean_rls_db, seeded_a, 1, admitted=True)
    (failed,) = await _insert_jobs(
        clean_rls_db, seeded_a, 1, status="failed", error_code=ErrorCode.WORKER_LOST.value
    )
    async with clean_rls_db.acquire() as conn:
        await conn.execute(
            "UPDATE extraction_jobs SET status = 'running',"
            " conversation_id = (SELECT conversation_id FROM extraction_jobs WHERE id = $2)"
            " WHERE id = $1",
            active,
            failed,
        )
    arq = FakeArq()

    result = await ExtractionScheduler(clean_rls_db, arq, _CONFIG).tick()

    row = await _row(clean_rls_db, failed)
    assert result == TickResult()
    assert arq.calls == []
    assert (row["status"], row["admitted_at"]) == ("failed", None)
