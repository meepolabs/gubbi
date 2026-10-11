"""Integration tests for the cross-tenant extraction queue repository (admin pool)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

import asyncpg
import pytest
from gubbi_common.db.user_scoped import user_scoped_connection

from gubbi.extraction.intake import Lane
from gubbi.extraction.retry_policy import DETERMINISTIC_MAX_ATTEMPTS, ErrorCode, is_terminal
from gubbi.scripts.release_held_extraction_jobs import release_held
from gubbi.storage.repositories import extraction_queue as queue_repo

if TYPE_CHECKING:
    from tests.fixtures.tenants import TenantSeed

pytestmark = [pytest.mark.asyncio(loop_scope="session"), pytest.mark.integration]

# Relative times are offsets from the database's now() at insert, so a long
# run never ages a fixture row past a threshold the test compares it with.
_NOW = timedelta(0)
_PAST = timedelta(hours=-1)
_FUTURE = timedelta(days=1)
_INFINITY = "infinity"


@dataclass(frozen=True)
class Job:
    status: str = "pending"
    run_after: timedelta | str = _PAST
    error_code: str | None = None
    lane: str = "live"
    source: str = "chatgpt"
    activity_at: datetime | None = None
    admitted_at: datetime | timedelta | None = None
    created_at: datetime | timedelta = _NOW
    attempts: int = 0


def _at(db_now: datetime, value: datetime | timedelta) -> datetime:
    return db_now + value if isinstance(value, timedelta) else value


async def _insert(pool: asyncpg.Pool, seed: TenantSeed, job: Job) -> UUID:
    """Insert one job on its own conversation, so the active-job unique index never collides."""
    async with pool.acquire() as conn:
        db_now: datetime = await conn.fetchval("SELECT now()")
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
                 activity_at, admitted_at, created_at, attempts)
            VALUES ($1, $2, $3, $4, $5::text::timestamptz, $6, $7, $8, $9, $10, $11)
            RETURNING id
            """,
            seed.user_id,
            conversation_id,
            job.source,
            job.status,
            job.run_after
            if isinstance(job.run_after, str)
            else _at(db_now, job.run_after).isoformat(),
            job.error_code,
            job.lane,
            job.activity_at,
            None if job.admitted_at is None else _at(db_now, job.admitted_at),
            _at(db_now, job.created_at),
            job.attempts,
        )
    return UUID(str(job_id))


def _admitted(**fields: Any) -> Job:
    return Job(admitted_at=_PAST, **fields)


async def _runnable_ids(pool: asyncpg.Pool, limit: int = 100, per_user: int = 100) -> list[UUID]:
    async with pool.acquire() as conn:
        jobs = await queue_repo.select_runnable(conn, limit, per_user)
    return [job.job_id for job in jobs]


# ---------------------------------------------------------------------------
# admit
# ---------------------------------------------------------------------------


async def test_admit_marks_a_pending_row_dispatched_and_clears_its_defer_reason(
    clean_rls_db: asyncpg.Pool, seeded_a: TenantSeed
) -> None:
    job_id = await _insert(clean_rls_db, seeded_a, Job())
    async with clean_rls_db.acquire() as conn:
        await conn.execute(
            "UPDATE extraction_jobs SET defer_reason = 'budget' WHERE id = $1", job_id
        )
        changed = await queue_repo.admit(conn, [job_id])
        row = await conn.fetchrow(
            "SELECT admitted_at IS NOT NULL AS dispatched, defer_reason"
            " FROM extraction_jobs WHERE id = $1",
            job_id,
        )

    assert changed == 1
    assert (row["dispatched"], row["defer_reason"]) == (True, None)


@pytest.mark.parametrize("status", ["running", "failed", "completed"])
async def test_admit_leaves_a_row_that_is_not_pending_alone(
    clean_rls_db: asyncpg.Pool, seeded_a: TenantSeed, status: str
) -> None:
    job_id = await _insert(clean_rls_db, seeded_a, Job(status=status))

    async with clean_rls_db.acquire() as conn:
        changed = await queue_repo.admit(conn, [job_id])
        admitted_at = await conn.fetchval(
            "SELECT admitted_at FROM extraction_jobs WHERE id = $1", job_id
        )

    assert (changed, admitted_at) == (0, None)


# ---------------------------------------------------------------------------
# select_runnable
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "excluded",
    [
        pytest.param(Job(run_after=_INFINITY), id="pending-held"),
        pytest.param(
            Job(status="failed", error_code="worker_lost", run_after=_INFINITY),
            id="failed-retryable-held",
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
        pytest.param(_admitted(), id="pending-already-dispatched"),
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


class _FetchThroughCursor:
    """Run ``fetch`` through a cursor and call ``between`` after the first row.

    The ranking finishes under the statement's snapshot before the first row
    is locked, so a row ``between`` changes and commits is locked in its new
    version, which only the quals on the locked row recheck.
    """

    def __init__(self, conn: asyncpg.Connection, between: Any) -> None:
        self._conn = conn
        self._between = between

    async def fetch(self, query: str, *args: object) -> list[asyncpg.Record]:
        async with self._conn.transaction():
            cursor = await self._conn.cursor(query, *args)
            first = await cursor.fetchrow()
            await self._between()
            rest = await cursor.fetch(100)
        return [first, *rest] if first is not None else rest


@pytest.mark.parametrize(
    "change",
    [
        pytest.param("run_after = now() + interval '1 day'", id="deferred"),
        pytest.param("admitted_at = now()", id="dispatched"),
    ],
)
async def test_select_runnable_drops_a_row_changed_after_its_snapshot_before_its_lock(
    clean_rls_db: asyncpg.Pool, seeded_a: TenantSeed, change: str
) -> None:
    first = await _insert(clean_rls_db, seeded_a, Job(lane="paste"))
    changed = await _insert(clean_rls_db, seeded_a, Job(lane="live"))

    async def _change_and_commit() -> None:
        await clean_rls_db.execute(
            f"UPDATE extraction_jobs SET {change} WHERE id = $1",  # noqa: S608 - fixed SQL
            changed,
        )

    async with clean_rls_db.acquire() as conn:
        jobs = await queue_repo.select_runnable(
            _FetchThroughCursor(conn, _change_and_commit),  # type: ignore[arg-type]
            10,
            10,
        )

    assert [job.job_id for job in jobs] == [first]


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


async def test_select_runnable_returns_only_the_newest_failed_row_of_a_conversation(
    clean_rls_db: asyncpg.Pool, seeded_a: TenantSeed
) -> None:
    failed = Job(status="failed", error_code="worker_lost")
    older = await _insert(clean_rls_db, seeded_a, replace(failed, created_at=_PAST))
    newer = await _insert(clean_rls_db, seeded_a, failed)
    async with clean_rls_db.acquire() as conn:
        await conn.execute(
            "UPDATE extraction_jobs SET conversation_id ="
            " (SELECT conversation_id FROM extraction_jobs WHERE id = $2) WHERE id = $1",
            older,
            newer,
        )

    assert await _runnable_ids(clean_rls_db) == [newer]


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

    await release_held(clean_rls_db)

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


async def test_select_runnable_takes_at_most_per_user_rows_of_each_user_before_the_limit(
    clean_rls_db: asyncpg.Pool, seeded_a: TenantSeed, seeded_b: TenantSeed
) -> None:
    def at(minute: int) -> datetime:
        return datetime(2026, 10, 1, 12, minute, tzinfo=UTC)

    busy = [
        await _insert(clean_rls_db, seeded_a, Job(lane="paste", activity_at=at(50 - m)))
        for m in range(5)
    ]
    quiet = await _insert(clean_rls_db, seeded_b, Job(lane="backfill", activity_at=at(0)))

    selected = await _runnable_ids(clean_rls_db, limit=3, per_user=2)

    assert selected == [busy[0], busy[1], quiet]


async def test_select_runnable_returns_the_scheduling_fields(
    clean_rls_db: asyncpg.Pool, seeded_a: TenantSeed
) -> None:
    activity = datetime(2026, 10, 2, 8, tzinfo=UTC)
    job_id = await _insert(
        clean_rls_db,
        seeded_a,
        Job(lane="backfill", source="claude", activity_at=activity),
    )
    async with clean_rls_db.acquire() as conn:
        await conn.execute("UPDATE extraction_jobs SET attempts = 2 WHERE id = $1", job_id)
        (job,) = await queue_repo.select_runnable(conn, 10, 10)

    assert job.job_id == job_id
    assert job.user_id == seeded_a.user_id
    assert job.source == "claude"
    assert job.lane is Lane.BACKFILL
    assert job.activity_at == activity
    assert job.attempts == 2
    assert job.admitted_at is None


# ---------------------------------------------------------------------------
# defer_lane
# ---------------------------------------------------------------------------


_HOUR = timedelta(hours=1)


async def _defer_lane(
    conn: asyncpg.Connection,
    lane: str = "live",
    reason: str = "budget",
    *,
    user_id: UUID | None,
    recheck: timedelta = _HOUR,
) -> int:
    return await queue_repo.defer_lane(conn, lane, reason, user_id=user_id, recheck=recheck)


async def _deferral(pool: asyncpg.Pool, job_id: UUID) -> tuple[str | None, bool]:
    row = await pool.fetchrow(
        "SELECT defer_reason, run_after > now() AS held FROM extraction_jobs WHERE id = $1",
        job_id,
    )
    assert row is not None
    return row["defer_reason"], row["held"]


@pytest.mark.parametrize("reason", ["budget", "global", "provider"])
async def test_defer_lane_records_the_reason(
    clean_rls_db: asyncpg.Pool, seeded_a: TenantSeed, reason: str
) -> None:
    job_id = await _insert(clean_rls_db, seeded_a, Job())
    async with clean_rls_db.acquire() as conn:
        changed = await _defer_lane(conn, reason=reason, user_id=seeded_a.user_id)

    assert changed == 1
    assert await _deferral(clean_rls_db, job_id) == (reason, True)


async def test_defer_lane_defers_only_the_users_runnable_rows_of_that_lane(
    clean_rls_db: asyncpg.Pool, seeded_a: TenantSeed, seeded_b: TenantSeed
) -> None:
    deferred = [
        await _insert(clean_rls_db, seeded_a, Job(lane="live")),
        await _insert(clean_rls_db, seeded_a, Job(lane="live", status="failed")),
    ]
    untouched = [
        await _insert(clean_rls_db, seeded_a, Job(lane="paste")),
        await _insert(clean_rls_db, seeded_a, _admitted(lane="live")),
        await _insert(clean_rls_db, seeded_a, Job(lane="live", status="running")),
        await _insert(clean_rls_db, seeded_a, Job(lane="live", run_after=_INFINITY)),
        await _insert(
            clean_rls_db, seeded_a, Job(lane="live", status="failed", error_code="llm_refused")
        ),
        await _insert(clean_rls_db, seeded_b, Job(lane="live")),
    ]

    async with clean_rls_db.acquire() as conn:
        changed = await _defer_lane(conn, user_id=seeded_a.user_id)

    assert changed == len(deferred)
    assert [(await _deferral(clean_rls_db, j))[0] for j in deferred] == ["budget"] * 2
    assert [(await _deferral(clean_rls_db, j))[0] for j in untouched] == [None] * 6


async def test_defer_lane_without_a_user_defers_the_lane_for_every_user(
    clean_rls_db: asyncpg.Pool, seeded_a: TenantSeed, seeded_b: TenantSeed
) -> None:
    jobs = [
        await _insert(clean_rls_db, seeded_a, Job(lane="backfill")),
        await _insert(clean_rls_db, seeded_b, Job(lane="backfill")),
    ]
    other_lane = await _insert(clean_rls_db, seeded_b, Job(lane="live"))

    async with clean_rls_db.acquire() as conn:
        changed = await _defer_lane(conn, "backfill", "global", user_id=None)

    assert changed == 2
    assert [await _deferral(clean_rls_db, j) for j in jobs] == [("global", True)] * 2
    assert await _deferral(clean_rls_db, other_lane) == (None, False)


@pytest.mark.parametrize(
    ("recheck", "expect_month_start"),
    [
        pytest.param(timedelta(minutes=5), False, id="recheck-first"),
        pytest.param(timedelta(days=40), True, id="next-month-first"),
    ],
)
async def test_defer_lane_holds_until_the_earlier_of_recheck_and_next_utc_month(
    clean_rls_db: asyncpg.Pool, seeded_a: TenantSeed, recheck: timedelta, expect_month_start: bool
) -> None:
    job_id = await _insert(clean_rls_db, seeded_a, Job())
    async with clean_rls_db.acquire() as conn, conn.transaction():
        await _defer_lane(conn, user_id=seeded_a.user_id, recheck=recheck)
        row = await conn.fetchrow(
            """
            SELECT run_after - now() AS wait,
                   run_after = (date_trunc('month', now() AT TIME ZONE 'UTC')
                                + interval '1 month') AT TIME ZONE 'UTC' AS at_month_start
            FROM extraction_jobs WHERE id = $1
            """,
            job_id,
        )

    assert row["at_month_start"] is expect_month_start
    if not expect_month_start:
        assert row["wait"] == recheck


async def test_defer_lane_rejects_an_unknown_reason(
    clean_rls_db: asyncpg.Pool, seeded_a: TenantSeed
) -> None:
    await _insert(clean_rls_db, seeded_a, Job())
    async with clean_rls_db.acquire() as conn:
        with pytest.raises(asyncpg.CheckViolationError):
            await _defer_lane(conn, reason="whim", user_id=seeded_a.user_id)


# ---------------------------------------------------------------------------
# revive_failed
# ---------------------------------------------------------------------------


async def test_revive_failed_skips_a_row_whose_conversation_became_active(
    clean_rls_db: asyncpg.Pool, seeded_a: TenantSeed
) -> None:
    blocked = await _insert(clean_rls_db, seeded_a, Job(status="failed", error_code="worker_lost"))
    free = await _insert(clean_rls_db, seeded_a, Job(status="failed", error_code="worker_lost"))
    async with clean_rls_db.acquire() as conn:
        await conn.execute(
            "INSERT INTO extraction_jobs (user_id, conversation_id, source, status)"
            " SELECT user_id, conversation_id, source, 'pending' FROM extraction_jobs"
            " WHERE id = $1",
            blocked,
        )
        async with conn.transaction():
            revived = await queue_repo.revive_failed(conn, [blocked, free])
            in_transaction = await conn.fetchval("SELECT 1")
        statuses = dict(
            await conn.fetch(
                "SELECT id, status FROM extraction_jobs WHERE id = ANY($1::uuid[])",
                [blocked, free],
            )
        )

    assert revived == {free}
    assert in_transaction == 1
    assert statuses == {blocked: "failed", free: "pending"}


# ---------------------------------------------------------------------------
# dispatch bookkeeping
# ---------------------------------------------------------------------------


async def test_stale_admitted_pending_returns_old_admitted_pending_rows(
    clean_rls_db: asyncpg.Pool, seeded_a: TenantSeed
) -> None:
    old = timedelta(minutes=-30)
    older = await _insert(clean_rls_db, seeded_a, Job(admitted_at=old - timedelta(minutes=1)))
    stale = await _insert(clean_rls_db, seeded_a, Job(admitted_at=old))
    for job in (
        Job(admitted_at=timedelta(minutes=-1)),
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


async def test_inflight_by_user_counts_each_users_dispatched_pending_and_running(
    clean_rls_db: asyncpg.Pool, seeded_a: TenantSeed, seeded_b: TenantSeed
) -> None:
    no_jobs = uuid4()
    for job in (
        _admitted(),
        _admitted(status="running"),
        _admitted(status="completed"),
        Job(),
        Job(status="running"),
    ):
        await _insert(clean_rls_db, seeded_a, job)
    await _insert(clean_rls_db, seeded_b, _admitted())

    async with clean_rls_db.acquire() as conn:
        counts = await queue_repo.inflight_by_user(
            conn, [seeded_a.user_id, seeded_b.user_id, no_jobs]
        )

    assert counts == {seeded_a.user_id: 2, seeded_b.user_id: 1, no_jobs: 0}


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


# ---------------------------------------------------------------------------
# Reads under RLS (the ingest request path runs as the user, not the admin pool)
# ---------------------------------------------------------------------------


async def test_first_upload_at_under_rls_sees_only_the_users_own_jobs(
    clean_rls_db: asyncpg.Pool,
    app_pool: asyncpg.Pool,
    seeded_a: TenantSeed,
    seeded_b: TenantSeed,
) -> None:
    def day(n: int) -> datetime:
        return datetime(2026, 10, n, tzinfo=UTC)

    await _insert(clean_rls_db, seeded_a, Job(lane="live", source="chatgpt", created_at=day(5)))
    await _insert(clean_rls_db, seeded_b, Job(lane="live", source="chatgpt", created_at=day(2)))

    async with user_scoped_connection(app_pool, user_id=seeded_a.user_id) as conn:
        own = await queue_repo.first_upload_at(conn, seeded_a.user_id, "chatgpt")
        own_any = await queue_repo.first_upload_at(conn, seeded_a.user_id, None)
        other = await queue_repo.first_upload_at(conn, seeded_b.user_id, None)

    assert (own, own_any, other) == (day(5), day(5), None)
