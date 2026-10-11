"""Integration tests for the extraction queue scheduler against a real database.

The Arq pool is a fake that records every enqueue; the admin pool is the
BYPASSRLS test pool the scheduler uses in production.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator
from contextlib import suppress
from dataclasses import dataclass, field, replace
from datetime import timedelta
from typing import TYPE_CHECKING, Any
from uuid import UUID

import asyncpg
import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

from gubbi.config import get_settings
from gubbi.extraction.intake import TOKEN_COUNT_VERSION
from gubbi.extraction.retry_policy import TRANSIENT_BASE_DELAY, ErrorCode
from gubbi.extraction.scheduler import (
    EXTRACT_CONVERSATION_JOB,
    ExtractionScheduler,
    SchedulerConfig,
    TickResult,
)
from gubbi.storage.repositories import extraction_jobs
from gubbi.storage.repositories import extraction_queue as queue_repo

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
    filtered_tokens: int = 100,
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
                 activity_at, filtered_tokens, token_count_version, admitted_at,
                 budget_period_start, attempts)
            SELECT $2, numbered.id, 'chatgpt', $4, $5, $6, now() - interval '1 minute',
                   now() + $7::interval - numbered.n * interval '1 second', $10, $11,
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
            filtered_tokens,
            TOKEN_COUNT_VERSION,
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


@pytest.mark.parametrize("budget_enabled", [False, True], ids=["budget_off", "budget_on"])
async def test_a_retryable_failed_row_runs_again_without_a_second_charge(
    clean_rls_db: asyncpg.Pool,
    seeded_a: TenantSeed,
    monkeypatch: pytest.MonkeyPatch,
    budget_enabled: bool,
) -> None:
    # Its own earlier charge uses the user's only monthly conversation: deciding
    # the row again would defer it.
    monkeypatch.setenv("JOURNAL_EXTRACTION_MONTHLY_CONVERSATIONS", "1")
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
    config = replace(_CONFIG, budget_enabled=budget_enabled)

    result = await ExtractionScheduler(clean_rls_db, arq, config).tick()

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


async def _elapse_backoff(pool: asyncpg.Pool, job_id: UUID) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE extraction_jobs SET run_after = now() - interval '1 second' WHERE id = $1",
            job_id,
        )


async def test_an_enqueue_failure_fails_the_row_as_enqueue_lost(
    clean_rls_db: asyncpg.Pool, seeded_a: TenantSeed
) -> None:
    (job_id,) = await _insert_jobs(clean_rls_db, seeded_a, 1)
    scheduler = ExtractionScheduler(clean_rls_db, FakeArq(hook=_failing_enqueue), _CONFIG)

    result = await scheduler.tick()

    assert result == TickResult(admitted=1, enqueue_failed=1, enqueue_lost=1)
    row = await _row(clean_rls_db, job_id)
    assert (row["status"], row["error_code"], row["backed_off"], row["admitted_at"] is None) == (
        "failed",
        ErrorCode.ENQUEUE_LOST.value,
        True,
        False,
    )


async def test_enqueue_failures_across_two_instances_fail_the_row_until_one_dispatches_it(
    clean_rls_db: asyncpg.Pool, seeded_a: TenantSeed
) -> None:
    (job_id,) = await _insert_jobs(clean_rls_db, seeded_a, 1)
    first_failing = ExtractionScheduler(clean_rls_db, FakeArq(hook=_failing_enqueue), _CONFIG)
    second_failing = ExtractionScheduler(clean_rls_db, FakeArq(hook=_failing_enqueue), _CONFIG)
    healthy_arq = FakeArq()
    healthy = ExtractionScheduler(clean_rls_db, healthy_arq, _CONFIG)

    first = await first_failing.tick()
    after_first = await _row(clean_rls_db, job_id)
    await _elapse_backoff(clean_rls_db, job_id)
    second = await second_failing.tick()
    after_second = await _row(clean_rls_db, job_id)
    await _elapse_backoff(clean_rls_db, job_id)
    third = await healthy.tick()

    lost = ("failed", ErrorCode.ENQUEUE_LOST.value)
    assert (first, (after_first["status"], after_first["error_code"])) == (
        TickResult(admitted=1, enqueue_failed=1, enqueue_lost=1),
        lost,
    )
    assert (second, (after_second["status"], after_second["error_code"])) == (
        TickResult(enqueue_failed=1, enqueue_lost=1),
        lost,
    )
    assert third == TickResult(enqueued=1)
    assert [call[2] for call in healthy_arq.calls] == [f"{job_id}:0"]
    assert (await _row(clean_rls_db, job_id))["status"] == "pending"


async def _dispatch_state(pool: asyncpg.Pool, job_id: UUID) -> asyncpg.Record:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT attempts, dispatch_failures, run_after - completed_at AS delay"
            " FROM extraction_jobs WHERE id = $1",
            job_id,
        )
    assert row is not None
    return row


async def test_consecutive_enqueue_losses_back_off_longer_without_counting_attempts(
    clean_rls_db: asyncpg.Pool, seeded_a: TenantSeed
) -> None:
    (job_id,) = await _insert_jobs(clean_rls_db, seeded_a, 1, attempts=1)
    scheduler = ExtractionScheduler(clean_rls_db, FakeArq(hook=_failing_enqueue), _CONFIG)

    await scheduler.tick()
    first = await _dispatch_state(clean_rls_db, job_id)
    await _elapse_backoff(clean_rls_db, job_id)
    await scheduler.tick()
    second = await _dispatch_state(clean_rls_db, job_id)

    assert (first["attempts"], first["dispatch_failures"]) == (1, 1)
    assert (second["attempts"], second["dispatch_failures"]) == (1, 2)
    # Jitter only shortens a delay, so a second delay past the first-attempt
    # base delay is backoff growth, not jitter.
    assert first["delay"] < second["delay"]
    assert second["delay"] > TRANSIENT_BASE_DELAY


async def test_a_worker_claim_resets_the_enqueue_loss_count(
    clean_rls_db: asyncpg.Pool, seeded_a: TenantSeed
) -> None:
    (job_id,) = await _insert_jobs(clean_rls_db, seeded_a, 1)
    await ExtractionScheduler(clean_rls_db, FakeArq(hook=_failing_enqueue), _CONFIG).tick()
    await _elapse_backoff(clean_rls_db, job_id)
    await ExtractionScheduler(clean_rls_db, FakeArq(), _CONFIG).tick()

    async with clean_rls_db.acquire() as conn:
        claimed = await extraction_jobs.mark_running(conn, job_id)
    state = await _dispatch_state(clean_rls_db, job_id)

    assert claimed is True
    assert (state["attempts"], state["dispatch_failures"]) == (1, 0)


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

    result = await ExtractionScheduler(clean_rls_db, FakeArq(hook=_claim_then_fail), _CONFIG).tick()

    assert result == TickResult(enqueue_failed=1)
    assert (await _row(clean_rls_db, job_id))["status"] == "running"


async def _share_conversation(pool: asyncpg.Pool, active: UUID, failed: UUID) -> None:
    """Move ``active`` onto ``failed``'s conversation and mark it running."""
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE extraction_jobs SET status = 'running',"
            " conversation_id = (SELECT conversation_id FROM extraction_jobs WHERE id = $2)"
            " WHERE id = $1",
            active,
            failed,
        )


@pytest.mark.parametrize("budget_enabled", [False, True], ids=["budget_off", "budget_on"])
async def test_a_failed_row_stays_failed_while_its_conversation_has_an_active_job(
    clean_rls_db: asyncpg.Pool,
    seeded_a: TenantSeed,
    monkeypatch: pytest.MonkeyPatch,
    budget_enabled: bool,
) -> None:
    # The running job uses one of two monthly conversations, so the later
    # candidate fits only if the blocked failed row is not counted as usage.
    monkeypatch.setenv("JOURNAL_EXTRACTION_MONTHLY_CONVERSATIONS", "2")
    (active,) = await _insert_jobs(clean_rls_db, seeded_a, 1, admitted=True)
    (failed,) = await _insert_jobs(
        clean_rls_db, seeded_a, 1, status="failed", error_code=ErrorCode.WORKER_LOST.value
    )
    (later,) = await _insert_jobs(clean_rls_db, seeded_a, 1, activity_offset=timedelta(days=-1))
    await _share_conversation(clean_rls_db, active, failed)
    arq = FakeArq()
    config = replace(_CONFIG, budget_enabled=budget_enabled)

    result = await ExtractionScheduler(clean_rls_db, arq, config).tick()

    row = await _row(clean_rls_db, failed)
    assert result == TickResult(admitted=1, enqueued=1)
    assert arq.job_ids == [later]
    assert (row["status"], row["admitted_at"], row["defer_reason"]) == ("failed", None, None)


async def test_blocked_failed_rows_filling_a_batch_do_not_hide_later_runnable_work(
    clean_rls_db: asyncpg.Pool, seeded_a: TenantSeed
) -> None:
    config = replace(_CONFIG, batch_size=3)
    blocked = await _insert_jobs(
        clean_rls_db,
        seeded_a,
        config.batch_size + 2,
        status="failed",
        error_code=ErrorCode.WORKER_LOST.value,
    )
    async with clean_rls_db.acquire() as conn:
        await conn.execute(
            "INSERT INTO extraction_jobs (user_id, conversation_id, source, status, created_at)"
            " SELECT user_id, conversation_id, source, 'running', created_at - interval '1 hour'"
            " FROM extraction_jobs WHERE id = ANY($1::uuid[])",
            blocked,
        )
    (later,) = await _insert_jobs(clean_rls_db, seeded_a, 1, activity_offset=timedelta(days=-1))
    arq = FakeArq()

    result = await ExtractionScheduler(clean_rls_db, arq, config).tick()

    assert result == TickResult(admitted=1, enqueued=1)
    assert arq.job_ids == [later]


async def test_a_failed_row_an_ingest_reactivates_mid_tick_is_skipped_without_aborting_the_tick(
    clean_rls_db: asyncpg.Pool, seeded_a: TenantSeed, monkeypatch: pytest.MonkeyPatch
) -> None:
    (failed,) = await _insert_jobs(
        clean_rls_db, seeded_a, 1, status="failed", error_code=ErrorCode.WORKER_LOST.value
    )
    (later,) = await _insert_jobs(clean_rls_db, seeded_a, 1, activity_offset=timedelta(days=-1))
    select_runnable = queue_repo.select_runnable

    async def _ingest_after_check(conn: asyncpg.Connection, limit: int) -> list[Any]:
        found = await select_runnable(conn, limit)
        async with clean_rls_db.acquire() as ingest:
            await ingest.execute(
                "INSERT INTO extraction_jobs (user_id, conversation_id, source, status)"
                " SELECT user_id, conversation_id, source, 'pending'"
                " FROM extraction_jobs WHERE id = $1",
                failed,
            )
        return found

    monkeypatch.setattr(queue_repo, "select_runnable", _ingest_after_check)
    arq = FakeArq()

    result = await ExtractionScheduler(clean_rls_db, arq, _CONFIG).tick()

    row = await _row(clean_rls_db, failed)
    assert result == TickResult(admitted=1, enqueued=1)
    assert arq.job_ids == [later]
    assert (row["status"], row["admitted_at"]) == ("failed", None)


# ---------------------------------------------------------------------------
# Deferral and capacity
# ---------------------------------------------------------------------------


async def test_deferred_rows_ahead_of_runnable_work_stop_hiding_it_within_two_ticks(
    clean_rls_db: asyncpg.Pool,
    seeded_a: TenantSeed,
    seeded_b: TenantSeed,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JOURNAL_EXTRACTION_MONTHLY_CONVERSATIONS", "1")
    await _insert_jobs(clean_rls_db, seeded_a, 1, status="completed", admitted=True)
    over_budget = await _insert_jobs(clean_rls_db, seeded_a, 550)
    (runnable,) = await _insert_jobs(clean_rls_db, seeded_b, 1, activity_offset=timedelta(days=-30))
    arq = FakeArq()
    scheduler = ExtractionScheduler(clean_rls_db, arq, replace(_CONFIG, budget_enabled=True))

    first = await scheduler.tick()
    second = await scheduler.tick()

    assert first == TickResult(deferred=500)
    assert second == TickResult(admitted=1, deferred=50, enqueued=1)
    assert arq.job_ids == [runnable]
    async with clean_rls_db.acquire() as conn:
        held = await conn.fetchval(
            "SELECT count(*) FROM extraction_jobs WHERE id = ANY($1::uuid[])"
            " AND defer_reason = 'budget' AND run_after > now()"
            " AND run_after <= now() + interval '1 hour'",
            over_budget,
        )
    assert held == len(over_budget)


async def test_a_full_dispatch_bound_still_records_later_limit_deferrals(
    clean_rls_db: asyncpg.Pool, seeded_a: TenantSeed, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Ceiling 200 with 100 tokens admitted: a 100-token live row fits; backfill
    # stops at 160, so a 100-token backfill row is held for the global ceiling.
    monkeypatch.setenv("JOURNAL_EXTRACTION_GLOBAL_MONTHLY_TOKEN_CEILING", "200")
    (inflight,) = await _insert_jobs(clean_rls_db, seeded_a, 1, admitted=True)
    async with clean_rls_db.acquire() as conn:
        await conn.execute("UPDATE extraction_jobs SET status = 'running' WHERE id = $1", inflight)
    (fitting,) = await _insert_jobs(clean_rls_db, seeded_a, 1)
    (blocked,) = await _insert_jobs(clean_rls_db, seeded_a, 1, lane="backfill")
    arq = FakeArq()
    config = replace(_CONFIG, budget_enabled=True, dispatch_bound=1)

    result = await ExtractionScheduler(clean_rls_db, arq, config).tick()

    fitting_row, blocked_row = await _row(clean_rls_db, fitting), await _row(clean_rls_db, blocked)
    assert result == TickResult(deferred=1)
    assert arq.calls == []
    assert (fitting_row["admitted_at"], fitting_row["defer_reason"]) == (None, None)
    assert (blocked_row["admitted_at"], blocked_row["defer_reason"]) == (None, "global")


async def test_a_row_that_fits_once_unplaced_work_is_uncounted_is_not_deferred(
    clean_rls_db: asyncpg.Pool,
    seeded_a: TenantSeed,
    seeded_b: TenantSeed,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Ceiling 200 with 100 tokens admitted: either 100-token live row fits
    # alone, both together do not. Neither is dispatched, so neither is charged.
    monkeypatch.setenv("JOURNAL_EXTRACTION_GLOBAL_MONTHLY_TOKEN_CEILING", "200")
    (inflight,) = await _insert_jobs(clean_rls_db, seeded_a, 1, admitted=True)
    async with clean_rls_db.acquire() as conn:
        await conn.execute("UPDATE extraction_jobs SET status = 'running' WHERE id = $1", inflight)
    (first,) = await _insert_jobs(clean_rls_db, seeded_a, 1)
    (second,) = await _insert_jobs(clean_rls_db, seeded_b, 1, activity_offset=timedelta(days=-1))
    config = replace(_CONFIG, budget_enabled=True, dispatch_bound=1)

    result = await ExtractionScheduler(clean_rls_db, FakeArq(), config).tick()

    rows = [await _row(clean_rls_db, job_id) for job_id in (first, second)]
    assert result == TickResult()
    assert [(row["admitted_at"], row["defer_reason"]) for row in rows] == [(None, None)] * 2


# ---------------------------------------------------------------------------
# Candidate locking
# ---------------------------------------------------------------------------


async def test_ingest_cannot_rewrite_a_candidate_while_the_tick_decides_it(
    clean_rls_db: asyncpg.Pool, seeded_a: TenantSeed, monkeypatch: pytest.MonkeyPatch
) -> None:
    (job_id,) = await _insert_jobs(clean_rls_db, seeded_a, 1)
    inflight_count = queue_repo.inflight_admitted_count
    refresh_outcome: list[str] = []

    async def _refresh_mid_tick(conn: asyncpg.Connection) -> int:
        async with clean_rls_db.acquire() as ingest, ingest.transaction():
            await ingest.execute("SET LOCAL lock_timeout = '200ms'")
            try:
                await ingest.execute(
                    "UPDATE extraction_jobs SET filtered_tokens = 999999"
                    " WHERE id = $1 AND status = 'pending' AND admitted_at IS NULL",
                    job_id,
                )
            except asyncpg.LockNotAvailableError:
                refresh_outcome.append("blocked")
            else:
                refresh_outcome.append("rewritten")
        return await inflight_count(conn)

    monkeypatch.setattr(queue_repo, "inflight_admitted_count", _refresh_mid_tick)

    result = await ExtractionScheduler(clean_rls_db, FakeArq(), _CONFIG).tick()

    async with clean_rls_db.acquire() as conn:
        tokens = await conn.fetchval(
            "SELECT filtered_tokens FROM extraction_jobs WHERE id = $1", job_id
        )
    assert result == TickResult(admitted=1, enqueued=1)
    assert (refresh_outcome, tokens) == (["blocked"], 100)


async def test_a_row_another_transaction_holds_is_skipped_not_waited_on(
    clean_rls_db: asyncpg.Pool, seeded_a: TenantSeed
) -> None:
    (held,) = await _insert_jobs(clean_rls_db, seeded_a, 1)
    (free,) = await _insert_jobs(clean_rls_db, seeded_a, 1, activity_offset=timedelta(days=-1))
    arq = FakeArq()

    async with clean_rls_db.acquire() as ingest, ingest.transaction():
        await ingest.execute("SELECT 1 FROM extraction_jobs WHERE id = $1 FOR UPDATE", held)
        result = await asyncio.wait_for(ExtractionScheduler(clean_rls_db, arq, _CONFIG).tick(), 5)

    assert result == TickResult(admitted=1, enqueued=1)
    assert arq.job_ids == [free]
    assert (await _row(clean_rls_db, held))["admitted_at"] is None


# ---------------------------------------------------------------------------
# Advisory lock release
# ---------------------------------------------------------------------------


class _TickCrashed(Exception):
    pass


async def _raise_mid_tick(_job_id: str | None) -> None:
    raise _TickCrashed


async def _block_forever(_job_id: str | None) -> None:
    await asyncio.Event().wait()


@pytest.mark.parametrize("hook", [_raise_mid_tick, _block_forever], ids=["raises", "cancelled"])
async def test_a_tick_that_ends_abnormally_releases_the_scheduler_lock(
    clean_rls_db: asyncpg.Pool,
    seeded_a: TenantSeed,
    hook: Callable[[str | None], Any],
) -> None:
    await _insert_jobs(clean_rls_db, seeded_a, 1)
    crashing = ExtractionScheduler(clean_rls_db, FakeArq(hook=hook), _CONFIG)
    healthy_arq = FakeArq()

    tick = asyncio.create_task(crashing.tick())
    done, _ = await asyncio.wait({tick}, timeout=0.5)
    if not done:
        tick.cancel()
    with suppress(_TickCrashed, asyncio.CancelledError):
        await tick
    result = await ExtractionScheduler(clean_rls_db, healthy_arq, _CONFIG).tick()

    assert result == TickResult()
