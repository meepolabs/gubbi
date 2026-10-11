"""Integration tests for the cross-tenant extraction queue repository (admin pool)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

import asyncpg
import pytest

from gubbi.extraction.intake import Lane
from gubbi.extraction.retry_policy import DETERMINISTIC_MAX_ATTEMPTS, ErrorCode, is_terminal
from gubbi.scripts.release_parked_extraction_jobs import release_parked
from gubbi.storage.repositories import extraction_queue as queue_repo
from gubbi.storage.repositories.extraction_queue import UserPeriodUsage

if TYPE_CHECKING:
    from tests.fixtures.tenants import TenantSeed

pytestmark = [pytest.mark.asyncio(loop_scope="session"), pytest.mark.integration]

_PERIOD = date(2026, 10, 1)
_OTHER_PERIOD = date(2026, 9, 1)
_NOW = datetime.now(UTC)
_PAST = _NOW - timedelta(hours=1)
_FUTURE = _NOW + timedelta(days=1)
_INFINITY = "infinity"
_ZERO_USAGE = UserPeriodUsage(0, 0, 0, 0, 0, 0)


@dataclass(frozen=True)
class Job:
    status: str = "pending"
    run_after: datetime | str = _PAST
    error_code: str | None = None
    lane: str = "live"
    source: str = "chatgpt"
    activity_at: datetime | None = None
    filtered_tokens: int = 0
    admitted_at: datetime | None = None
    budget_period_start: date | None = None
    created_at: datetime = _NOW
    attempts: int = 0


async def _insert(pool: asyncpg.Pool, seed: TenantSeed, job: Job) -> UUID:
    """Insert one job on its own conversation, so the active-job unique index never collides."""
    async with pool.acquire() as conn:
        conversation_id = await conn.fetchval(
            """
            INSERT INTO conversations
                (topic_id, user_id, title_encrypted, title_nonce, slug, source,
                 summary_encrypted, summary_nonce, message_count, json_path)
            VALUES ($1, $2, 'title'::bytea, 'nonce_title1'::bytea, $3, 'chatgpt',
                    'summary'::bytea, 'nonce_summ12'::bytea, 0, 'x.json')
            RETURNING id
            """,
            seed.topic_id,
            seed.user_id,
            f"conv-{uuid4().hex[:12]}",
        )
        job_id = await conn.fetchval(
            """
            INSERT INTO extraction_jobs
                (user_id, conversation_id, source, status, run_after, error_code, lane,
                 activity_at, filtered_tokens, admitted_at, budget_period_start, created_at,
                 attempts)
            VALUES ($1, $2, $3, $4, $5::text::timestamptz, $6, $7, $8, $9, $10, $11, $12, $13)
            RETURNING id
            """,
            seed.user_id,
            conversation_id,
            job.source,
            job.status,
            job.run_after if isinstance(job.run_after, str) else job.run_after.isoformat(),
            job.error_code,
            job.lane,
            job.activity_at,
            job.filtered_tokens,
            job.admitted_at,
            job.budget_period_start,
            job.created_at,
            job.attempts,
        )
    return UUID(str(job_id))


def _admitted(**fields: Any) -> Job:
    return Job(admitted_at=_PAST, budget_period_start=_PERIOD, **fields)


async def _usage(pool: asyncpg.Pool, user_id: UUID, period: date) -> UserPeriodUsage:
    async with pool.acquire() as conn:
        return (await queue_repo.user_period_usage(conn, [user_id], period))[user_id]


async def _runnable_ids(pool: asyncpg.Pool, limit: int = 100) -> list[UUID]:
    async with pool.acquire() as conn:
        jobs = await queue_repo.select_runnable(conn, limit)
    return [job.job_id for job in jobs]


# ---------------------------------------------------------------------------
# admit
# ---------------------------------------------------------------------------


async def test_admit_charges_a_row_only_once(
    clean_rls_db: asyncpg.Pool, seeded_a: TenantSeed
) -> None:
    job_id = await _insert(clean_rls_db, seeded_a, Job(filtered_tokens=700))

    async with clean_rls_db.acquire() as conn:
        first = await queue_repo.admit(conn, [job_id], _PERIOD)
        charged = await conn.fetchrow(
            "SELECT admitted_at, budget_period_start FROM extraction_jobs WHERE id = $1", job_id
        )
        second = await queue_repo.admit(conn, [job_id], _OTHER_PERIOD)
        after = await conn.fetchrow(
            "SELECT admitted_at, budget_period_start FROM extraction_jobs WHERE id = $1", job_id
        )

    assert (first, second) == (1, 0)
    assert after == charged
    assert await _usage(clean_rls_db, seeded_a.user_id, _PERIOD) == UserPeriodUsage(
        1, 700, 0, 0, 0, 0
    )
    assert await _usage(clean_rls_db, seeded_a.user_id, _OTHER_PERIOD) == _ZERO_USAGE


async def test_admit_clears_the_defer_reason(
    clean_rls_db: asyncpg.Pool, seeded_a: TenantSeed
) -> None:
    job_id = await _insert(clean_rls_db, seeded_a, Job())
    async with clean_rls_db.acquire() as conn:
        await queue_repo.defer(conn, job_id, "budget")
        await queue_repo.admit(conn, [job_id], _PERIOD)
        reason = await conn.fetchval(
            "SELECT defer_reason FROM extraction_jobs WHERE id = $1", job_id
        )

    assert reason is None


# ---------------------------------------------------------------------------
# usage
# ---------------------------------------------------------------------------


async def test_user_period_usage_counts_only_admitted_rows(
    clean_rls_db: asyncpg.Pool, seeded_a: TenantSeed, seeded_b: TenantSeed
) -> None:
    for job in (
        _admitted(lane="live", filtered_tokens=10),
        _admitted(lane="backfill", filtered_tokens=200),
        Job(
            admitted_at=_PAST,
            budget_period_start=_OTHER_PERIOD,
            lane="backfill",
            filtered_tokens=3000,
        ),
        Job(
            admitted_at=_PAST, budget_period_start=_OTHER_PERIOD, lane="live", filtered_tokens=40000
        ),
        Job(lane="backfill", filtered_tokens=500000),
        Job(lane="backfill", budget_period_start=_PERIOD, filtered_tokens=8000000),
        Job(lane="live", filtered_tokens=6000000),
    ):
        await _insert(clean_rls_db, seeded_a, job)
    await _insert(clean_rls_db, seeded_b, _admitted(lane="backfill", filtered_tokens=70000000))

    async with clean_rls_db.acquire() as conn:
        usage = await queue_repo.user_period_usage(
            conn, [seeded_a.user_id, seeded_b.user_id], _PERIOD
        )

    assert usage == {
        seeded_a.user_id: UserPeriodUsage(
            period_conversations=2,
            period_tokens=210,
            period_backfill_conversations=1,
            period_backfill_tokens=200,
            lifetime_backfill_conversations=2,
            lifetime_backfill_tokens=3200,
        ),
        seeded_b.user_id: UserPeriodUsage(1, 70000000, 1, 70000000, 1, 70000000),
    }


async def test_user_period_usage_maps_a_user_without_jobs_to_zero(
    clean_rls_db: asyncpg.Pool, tenant_a: UUID
) -> None:
    assert await _usage(clean_rls_db, tenant_a, _PERIOD) == _ZERO_USAGE


async def test_global_period_tokens_sums_admitted_rows_of_the_period(
    clean_rls_db: asyncpg.Pool, seeded_a: TenantSeed, seeded_b: TenantSeed
) -> None:
    await _insert(clean_rls_db, seeded_a, _admitted(filtered_tokens=1))
    await _insert(clean_rls_db, seeded_b, _admitted(filtered_tokens=20))
    await _insert(clean_rls_db, seeded_a, Job(budget_period_start=_PERIOD, filtered_tokens=300))
    await _insert(
        clean_rls_db,
        seeded_b,
        Job(admitted_at=_PAST, budget_period_start=_OTHER_PERIOD, filtered_tokens=4000),
    )

    async with clean_rls_db.acquire() as conn:
        total = await queue_repo.global_period_tokens(conn, _PERIOD)
        empty = await queue_repo.global_period_tokens(conn, date(2026, 1, 1))

    assert (total, empty) == (21, 0)


# ---------------------------------------------------------------------------
# select_runnable
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "excluded",
    [
        pytest.param(Job(run_after=_INFINITY), id="pending-parked"),
        pytest.param(
            Job(status="failed", error_code="worker_lost", run_after=_INFINITY),
            id="failed-retryable-parked",
        ),
        pytest.param(Job(status="failed", error_code="llm_refused"), id="failed-terminal"),
        pytest.param(
            Job(status="failed", error_code="llm_schema", attempts=DETERMINISTIC_MAX_ATTEMPTS),
            id="failed-deterministic-exhausted",
        ),
        pytest.param(
            Job(
                status="failed",
                error_code="internal_error",
                attempts=DETERMINISTIC_MAX_ATTEMPTS + 2,
            ),
            id="failed-deterministic-past-max",
        ),
        pytest.param(Job(run_after=_FUTURE), id="pending-not-due"),
        pytest.param(
            Job(status="failed", error_code="llm_rate_limited", run_after=_FUTURE),
            id="failed-backoff-not-elapsed",
        ),
        pytest.param(_admitted(), id="pending-already-admitted"),
        pytest.param(Job(status="running"), id="running"),
        pytest.param(Job(status="completed"), id="completed"),
    ],
)
async def test_select_runnable_excludes_rows_that_may_not_run(
    clean_rls_db: asyncpg.Pool, seeded_a: TenantSeed, excluded: Job
) -> None:
    pending = await _insert(clean_rls_db, seeded_a, Job())
    retryable = await _insert(
        clean_rls_db, seeded_a, Job(status="failed", error_code="worker_lost")
    )
    await _insert(clean_rls_db, seeded_a, excluded)

    assert set(await _runnable_ids(clean_rls_db)) == {pending, retryable}


async def test_select_runnable_retries_a_deterministic_failure_below_its_limit(
    clean_rls_db: asyncpg.Pool, seeded_a: TenantSeed
) -> None:
    below = await _insert(
        clean_rls_db,
        seeded_a,
        Job(status="failed", error_code="llm_schema", attempts=DETERMINISTIC_MAX_ATTEMPTS - 1),
    )

    assert await _runnable_ids(clean_rls_db) == [below]


@pytest.mark.parametrize(
    "error_code",
    [pytest.param(None, id="without-code"), pytest.param("legacy_code", id="unknown-code")],
)
async def test_select_runnable_retries_an_unclassified_failure(
    clean_rls_db: asyncpg.Pool, seeded_a: TenantSeed, error_code: str | None
) -> None:
    unclassified = await _insert(
        clean_rls_db,
        seeded_a,
        Job(status="failed", error_code=error_code, attempts=DETERMINISTIC_MAX_ATTEMPTS + 1),
    )

    assert await _runnable_ids(clean_rls_db) == [unclassified]


# ---------------------------------------------------------------------------
# terminal_failure_sql
# ---------------------------------------------------------------------------

_PREDICATE_CASES = [
    (code, attempts)
    for code in [*(member.value for member in ErrorCode), "legacy_code", None]
    for attempts in range(DETERMINISTIC_MAX_ATTEMPTS + 2)
]


async def test_terminal_failure_sql_agrees_with_is_terminal(clean_rls_db: asyncpg.Pool) -> None:
    codes = [code for code, _ in _PREDICATE_CASES]
    attempts = [n for _, n in _PREDICATE_CASES]
    async with clean_rls_db.acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT ord, {queue_repo.terminal_failure_sql(3)} AS terminal
            FROM unnest($1::text[], $2::int[]) WITH ORDINALITY AS t(error_code, attempts, ord)
            ORDER BY ord
            """,  # noqa: S608 - fixed predicate; values are bound
            codes,
            attempts,
            *queue_repo.terminal_failure_args(),
        )

    from_sql = [row["terminal"] for row in rows]
    assert from_sql == [is_terminal(code, n) for code, n in _PREDICATE_CASES]
    assert any(from_sql)
    assert not all(from_sql)


_HELD_FAILURE_CASES = [
    (code, attempts)
    for code in [*(member.value for member in ErrorCode), None, "legacy_unknown_code"]
    for attempts in range(1, DETERMINISTIC_MAX_ATTEMPTS + 2)
]


async def test_release_never_frees_a_failed_row_the_scheduler_will_not_select(
    clean_rls_db: asyncpg.Pool, seeded_a: TenantSeed
) -> None:
    job_ids = {
        case: await _insert(
            clean_rls_db,
            seeded_a,
            Job(status="failed", error_code=case[0], attempts=case[1], run_after=_INFINITY),
        )
        for case in _HELD_FAILURE_CASES
    }

    await release_parked(clean_rls_db)

    selected = set(await _runnable_ids(clean_rls_db, limit=len(job_ids) + 1))
    still_held = set(
        await clean_rls_db.fetchval(
            "SELECT array_agg(id) FROM extraction_jobs WHERE run_after = 'infinity'"
        )
        or []
    )
    outcome = {
        case: ("selected" if job_id in selected else "held" if job_id in still_held else "lost")
        for case, job_id in job_ids.items()
    }
    assert outcome == {
        case: "held" if is_terminal(*case) else "selected" for case in _HELD_FAILURE_CASES
    }
    assert set(outcome.values()) == {"held", "selected"}


async def test_select_runnable_orders_by_lane_then_newest_activity(
    clean_rls_db: asyncpg.Pool, seeded_a: TenantSeed, seeded_b: TenantSeed
) -> None:
    def at(hour: int) -> datetime:
        return datetime(2026, 10, 1, hour, tzinfo=UTC)

    backfill = await _insert(clean_rls_db, seeded_a, Job(lane="backfill", activity_at=at(23)))
    live_unknown = await _insert(clean_rls_db, seeded_b, Job(lane="live", activity_at=None))
    live_old = await _insert(clean_rls_db, seeded_a, Job(lane="live", activity_at=at(2)))
    live_new = await _insert(clean_rls_db, seeded_a, Job(lane="live", activity_at=at(9)))
    paste = await _insert(clean_rls_db, seeded_b, Job(lane="paste", activity_at=at(1)))
    expected = [paste, live_new, live_old, live_unknown, backfill]

    assert await _runnable_ids(clean_rls_db) == expected
    assert await _runnable_ids(clean_rls_db, limit=2) == expected[:2]


async def test_select_runnable_returns_the_scheduling_fields(
    clean_rls_db: asyncpg.Pool, seeded_a: TenantSeed
) -> None:
    activity = datetime(2026, 10, 2, 8, tzinfo=UTC)
    job_id = await _insert(
        clean_rls_db,
        seeded_a,
        Job(lane="backfill", source="claude", activity_at=activity, filtered_tokens=321),
    )
    async with clean_rls_db.acquire() as conn:
        await conn.execute("UPDATE extraction_jobs SET attempts = 2 WHERE id = $1", job_id)
        (job,) = await queue_repo.select_runnable(conn, 10)

    assert job.job_id == job_id
    assert job.user_id == seeded_a.user_id
    assert job.source == "claude"
    assert job.lane is Lane.BACKFILL
    assert job.activity_at == activity
    assert job.filtered_tokens == 321
    assert job.attempts == 2
    assert job.admitted_at is None


# ---------------------------------------------------------------------------
# defer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reason", ["budget", "global", "provider"])
async def test_defer_records_the_reason(
    clean_rls_db: asyncpg.Pool, seeded_a: TenantSeed, reason: str
) -> None:
    job_id = await _insert(clean_rls_db, seeded_a, Job())
    async with clean_rls_db.acquire() as conn:
        changed = await queue_repo.defer(conn, job_id, reason)
        stored = await conn.fetchval(
            "SELECT defer_reason FROM extraction_jobs WHERE id = $1", job_id
        )

    assert changed is True
    assert stored == reason


async def test_defer_leaves_a_finished_row_untouched(
    clean_rls_db: asyncpg.Pool, seeded_a: TenantSeed
) -> None:
    job_id = await _insert(clean_rls_db, seeded_a, Job(status="completed"))
    async with clean_rls_db.acquire() as conn:
        changed = await queue_repo.defer(conn, job_id, "budget")
        stored = await conn.fetchval(
            "SELECT defer_reason FROM extraction_jobs WHERE id = $1", job_id
        )

    assert changed is False
    assert stored is None


async def test_defer_rejects_an_unknown_reason(
    clean_rls_db: asyncpg.Pool, seeded_a: TenantSeed
) -> None:
    job_id = await _insert(clean_rls_db, seeded_a, Job())
    async with clean_rls_db.acquire() as conn:
        with pytest.raises(asyncpg.CheckViolationError):
            await queue_repo.defer(conn, job_id, "whim")


# ---------------------------------------------------------------------------
# dispatch bookkeeping
# ---------------------------------------------------------------------------


async def test_stale_admitted_pending_returns_old_admitted_pending_rows(
    clean_rls_db: asyncpg.Pool, seeded_a: TenantSeed
) -> None:
    old = _NOW - timedelta(minutes=30)
    older = await _insert(clean_rls_db, seeded_a, Job(admitted_at=old - timedelta(minutes=1)))
    stale = await _insert(clean_rls_db, seeded_a, Job(admitted_at=old))
    for job in (
        Job(admitted_at=_NOW - timedelta(minutes=1)),
        Job(admitted_at=old, status="running"),
        Job(admitted_at=old, status="failed", error_code="worker_lost"),
        Job(),
    ):
        await _insert(clean_rls_db, seeded_a, job)

    async with clean_rls_db.acquire() as conn:
        jobs = await queue_repo.stale_admitted_pending(conn, timedelta(minutes=5))

    assert [job.job_id for job in jobs] == [older, stale]


async def test_inflight_admitted_count_counts_admitted_pending_and_running(
    clean_rls_db: asyncpg.Pool, seeded_a: TenantSeed, seeded_b: TenantSeed
) -> None:
    await _insert(clean_rls_db, seeded_a, _admitted())
    await _insert(clean_rls_db, seeded_b, _admitted(status="running"))
    for job in (
        _admitted(status="completed"),
        _admitted(status="failed", error_code="worker_lost"),
        Job(),
        Job(status="running"),
    ):
        await _insert(clean_rls_db, seeded_a, job)

    async with clean_rls_db.acquire() as conn:
        assert await queue_repo.inflight_admitted_count(conn) == 2


# ---------------------------------------------------------------------------
# first_upload_at
# ---------------------------------------------------------------------------


async def test_first_upload_at_is_the_earliest_sync_job(
    clean_rls_db: asyncpg.Pool, seeded_a: TenantSeed, seeded_b: TenantSeed
) -> None:
    def day(n: int) -> datetime:
        return datetime(2026, 10, n, tzinfo=UTC)

    for job in (
        Job(lane="paste", source="chatgpt", created_at=day(1)),
        Job(lane="live", source="chatgpt", created_at=day(5)),
        Job(lane="backfill", source="chatgpt", created_at=day(4), status="completed"),
        Job(lane="backfill", source="claude", created_at=day(3)),
    ):
        await _insert(clean_rls_db, seeded_a, job)
    await _insert(clean_rls_db, seeded_b, Job(lane="live", source="chatgpt", created_at=day(2)))

    async with clean_rls_db.acquire() as conn:
        per_platform = await queue_repo.first_upload_at(conn, seeded_a.user_id, "chatgpt")
        any_platform = await queue_repo.first_upload_at(conn, seeded_a.user_id, None)
        unseen = await queue_repo.first_upload_at(conn, seeded_a.user_id, "gemini")

    assert (per_platform, any_platform, unseen) == (day(4), day(3), None)


async def test_first_upload_at_is_none_without_jobs(
    clean_rls_db: asyncpg.Pool, tenant_a: UUID
) -> None:
    async with clean_rls_db.acquire() as conn:
        assert await queue_repo.first_upload_at(conn, tenant_a, None) is None
