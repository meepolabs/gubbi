"""Integration tests for the extraction queue and budget gauges against a real database."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date, timedelta
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

import asyncpg
import pytest
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from gubbi.config import get_settings
from gubbi.extraction.intake import TOKEN_COUNT_VERSION
from gubbi.extraction.queue_metrics import record_queue_metrics
from gubbi.extraction.retry_policy import ErrorCode
from gubbi.extraction.scheduler import ExtractionScheduler, SchedulerConfig, TickResult
from gubbi.storage.repositories import extraction_queue as queue_repo
from gubbi.telemetry.attrs import MetricNames
from tests.fixtures.metrics import GaugePoints, collect_gauges, metric_reader  # noqa: F401

if TYPE_CHECKING:
    from tests.fixtures.tenants import TenantSeed

pytestmark = [pytest.mark.asyncio(loop_scope="session"), pytest.mark.integration]

_PERIOD = date(2026, 10, 1)
_OTHER_PERIOD = date(2026, 9, 1)
_INFINITY = "infinity"
_AGE_TOLERANCE_SECS = 120.0


@dataclass(frozen=True)
class Job:
    status: str = "pending"
    run_after: timedelta | str = timedelta(minutes=-10)
    lane: str = "live"
    defer_reason: str | None = None
    error_code: str | None = None
    admitted: bool = False
    budget_period_start: date | None = None
    filtered_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    token_count_version: int | None = TOKEN_COUNT_VERSION


@pytest.fixture(autouse=True)
def _fresh_settings() -> Iterator[None]:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


async def _insert(pool: asyncpg.Pool, seed: TenantSeed, job: Job) -> UUID:
    """Insert one job on its own conversation; ``run_after`` offsets are from now()."""
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
        run_after = (
            job.run_after
            if isinstance(job.run_after, str)
            else (await conn.fetchval("SELECT now()") + job.run_after).isoformat()
        )
        job_id = await conn.fetchval(
            """
            INSERT INTO extraction_jobs
                (user_id, conversation_id, source, status, run_after, lane, defer_reason,
                 error_code, admitted_at, budget_period_start, filtered_tokens, input_tokens,
                 output_tokens, cached_input_tokens, token_count_version)
            VALUES ($1, $2, 'chatgpt', $3, $4::text::timestamptz, $5, $6, $7,
                    CASE WHEN $8 THEN now() END, $9, $10, $11, $12, $13, $14)
            RETURNING id
            """,
            seed.user_id,
            conversation_id,
            job.status,
            run_after,
            job.lane,
            job.defer_reason,
            job.error_code,
            job.admitted,
            job.budget_period_start,
            job.filtered_tokens,
            job.input_tokens,
            job.output_tokens,
            job.cached_input_tokens,
            job.token_count_version,
        )
    return UUID(str(job_id))


async def _record(
    pool: asyncpg.Pool, reader: InMemoryMetricReader, *, budget_enabled: bool = False
) -> dict[str, GaugePoints]:
    async with pool.acquire() as conn:
        await record_queue_metrics(conn, _PERIOD, budget_enabled=budget_enabled)
    return collect_gauges(reader)


def _oldest_age(gauges: dict[str, GaugePoints]) -> float:
    points = gauges[MetricNames.EXTRACTION_QUEUE_OLDEST_RUNNABLE_AGE]
    assert list(points) == [()]
    return points[()]


def _by(gauges: dict[str, GaugePoints], name: str, key: str) -> dict[str, float]:
    return {dict(attrs)[key]: value for attrs, value in gauges[name].items()}


_DEFERRED = Job(run_after=timedelta(hours=1), defer_reason="budget")
_PARKED = Job(run_after=_INFINITY)


async def test_deferred_and_parked_rows_do_not_age_the_oldest_runnable_gauge(
    clean_rls_db: asyncpg.Pool,
    seeded_a: TenantSeed,
    metric_reader: InMemoryMetricReader,  # noqa: F811
) -> None:
    # Created first, so they are older than the runnable row by created_at.
    await _insert(clean_rls_db, seeded_a, _DEFERRED)
    await _insert(clean_rls_db, seeded_a, _PARKED)
    await _insert(clean_rls_db, seeded_a, Job(run_after=timedelta(hours=-3)))
    await _insert(clean_rls_db, seeded_a, Job(run_after=timedelta(hours=-1)))

    gauges = await _record(clean_rls_db, metric_reader)

    assert _oldest_age(gauges) == pytest.approx(3 * 3600, abs=_AGE_TOLERANCE_SECS)


async def test_the_oldest_runnable_gauge_is_zero_with_only_deferred_and_parked_rows(
    clean_rls_db: asyncpg.Pool,
    seeded_a: TenantSeed,
    metric_reader: InMemoryMetricReader,  # noqa: F811
) -> None:
    await _insert(clean_rls_db, seeded_a, _DEFERRED)
    await _insert(clean_rls_db, seeded_a, _PARKED)

    gauges = await _record(clean_rls_db, metric_reader)

    assert _oldest_age(gauges) == 0.0


async def test_a_failed_row_due_for_retry_ages_the_queue_and_a_terminal_one_does_not(
    clean_rls_db: asyncpg.Pool,
    seeded_a: TenantSeed,
    metric_reader: InMemoryMetricReader,  # noqa: F811
) -> None:
    await _insert(
        clean_rls_db,
        seeded_a,
        Job(status="failed", error_code=ErrorCode.SUPERSEDED.value, run_after=timedelta(days=-2)),
    )
    await _insert(
        clean_rls_db,
        seeded_a,
        Job(
            status="failed",
            error_code=ErrorCode.LLM_PROVIDER_ERROR.value,
            run_after=timedelta(hours=-2),
        ),
    )

    gauges = await _record(clean_rls_db, metric_reader)

    assert _oldest_age(gauges) == pytest.approx(2 * 3600, abs=_AGE_TOLERANCE_SECS)


async def test_depth_counts_runnable_rows_by_lane_and_deferred_counts_held_rows_by_reason(
    clean_rls_db: asyncpg.Pool,
    seeded_a: TenantSeed,
    metric_reader: InMemoryMetricReader,  # noqa: F811
) -> None:
    jobs: list[Job] = [
        *[Job(lane="backfill")] * 3,
        Job(lane="paste"),
        Job(lane="live", status="failed", error_code=ErrorCode.LLM_PROVIDER_ERROR.value),
        Job(lane="live", admitted=True, budget_period_start=_PERIOD),
        Job(lane="live", token_count_version=None),
        Job(lane="live", run_after=timedelta(hours=1)),
        *[Job(lane="backfill", run_after=timedelta(hours=1), defer_reason="global")] * 2,
        Job(lane="live", run_after=timedelta(hours=1), defer_reason="budget"),
        Job(lane="live", run_after=_INFINITY, defer_reason="budget"),
        Job(lane="live", defer_reason="budget"),
    ]
    for job in jobs:
        await _insert(clean_rls_db, seeded_a, job)

    gauges = await _record(clean_rls_db, metric_reader)

    assert _by(gauges, MetricNames.EXTRACTION_QUEUE_DEPTH, "lane") == {
        "backfill": 3.0,
        "live": 2.0,
        "paste": 1.0,
    }
    assert _by(gauges, MetricNames.EXTRACTION_QUEUE_DEFERRED, "defer_reason") == {
        "budget": 1.0,
        "global": 2.0,
    }


async def test_provider_tokens_sum_by_kind_over_rows_charged_to_the_period(
    clean_rls_db: asyncpg.Pool,
    seeded_a: TenantSeed,
    metric_reader: InMemoryMetricReader,  # noqa: F811
) -> None:
    charged = {"admitted": True, "budget_period_start": _PERIOD, "status": "completed"}
    await _insert(
        clean_rls_db,
        seeded_a,
        Job(**charged, input_tokens=100, output_tokens=10, cached_input_tokens=1),
    )
    await _insert(
        clean_rls_db,
        seeded_a,
        Job(**charged, input_tokens=200, output_tokens=20, cached_input_tokens=2),
    )
    await _insert(
        clean_rls_db,
        seeded_a,
        Job(
            admitted=True,
            budget_period_start=_OTHER_PERIOD,
            status="completed",
            input_tokens=5000,
            output_tokens=500,
            cached_input_tokens=50,
        ),
    )

    gauges = await _record(clean_rls_db, metric_reader)

    assert _by(gauges, MetricNames.EXTRACTION_PROVIDER_TOKENS_MONTH, "kind") == {
        "input": 300.0,
        "output": 30.0,
        "cached": 3.0,
    }


async def test_the_global_ratio_is_period_tokens_over_the_ceiling(
    clean_rls_db: asyncpg.Pool,
    seeded_a: TenantSeed,
    metric_reader: InMemoryMetricReader,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JOURNAL_EXTRACTION_GLOBAL_MONTHLY_TOKEN_CEILING", "1000")
    charged: dict[str, Any] = {"admitted": True, "status": "completed"}
    await _insert(
        clean_rls_db, seeded_a, Job(**charged, budget_period_start=_PERIOD, filtered_tokens=250)
    )
    await _insert(
        clean_rls_db, seeded_a, Job(**charged, budget_period_start=_PERIOD, filtered_tokens=50)
    )
    await _insert(
        clean_rls_db,
        seeded_a,
        Job(**charged, budget_period_start=_OTHER_PERIOD, filtered_tokens=600),
    )

    gauges = await _record(clean_rls_db, metric_reader, budget_enabled=True)

    assert gauges[MetricNames.EXTRACTION_BUDGET_GLOBAL_USED_RATIO] == {(): pytest.approx(0.3)}


@pytest.mark.parametrize(
    ("budget_enabled", "ceiling"),
    [(False, "1000"), (True, None)],
    ids=["budgets_disabled", "no_ceiling"],
)
async def test_the_global_ratio_is_not_recorded_without_an_active_ceiling(
    clean_rls_db: asyncpg.Pool,
    seeded_a: TenantSeed,
    metric_reader: InMemoryMetricReader,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    budget_enabled: bool,
    ceiling: str | None,
) -> None:
    if ceiling is None:
        monkeypatch.delenv("JOURNAL_EXTRACTION_GLOBAL_MONTHLY_TOKEN_CEILING", raising=False)
        monkeypatch.setenv("JOURNAL_LLM_BUDGET_ENABLED", "false")
    else:
        monkeypatch.setenv("JOURNAL_EXTRACTION_GLOBAL_MONTHLY_TOKEN_CEILING", ceiling)
    await _insert(
        clean_rls_db,
        seeded_a,
        Job(admitted=True, status="completed", budget_period_start=_PERIOD, filtered_tokens=10),
    )

    gauges = await _record(clean_rls_db, metric_reader, budget_enabled=budget_enabled)

    assert MetricNames.EXTRACTION_BUDGET_GLOBAL_USED_RATIO not in gauges
    assert gauges[MetricNames.EXTRACTION_QUEUE_OLDEST_RUNNABLE_AGE] == {(): 0.0}


class _Arq:
    async def enqueue_job(self, function: str, *args: Any, **kwargs: Any) -> object:
        return object()


_ONE_AT_A_TIME = SchedulerConfig(paused=False, budget_enabled=False, dispatch_bound=1)


async def test_a_scheduler_tick_records_the_queue_after_its_own_admissions(
    clean_rls_db: asyncpg.Pool,
    seeded_a: TenantSeed,
    metric_reader: InMemoryMetricReader,  # noqa: F811
) -> None:
    for _ in range(3):
        await _insert(clean_rls_db, seeded_a, Job(lane="backfill"))

    await ExtractionScheduler(clean_rls_db, _Arq(), _ONE_AT_A_TIME).tick()

    assert _by(collect_gauges(metric_reader), MetricNames.EXTRACTION_QUEUE_DEPTH, "lane") == {
        "backfill": 2.0,
        "live": 0.0,
        "paste": 0.0,
    }


async def test_a_failed_queue_metrics_read_does_not_fail_the_tick(
    clean_rls_db: asyncpg.Pool,
    seeded_a: TenantSeed,
    metric_reader: InMemoryMetricReader,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _insert(clean_rls_db, seeded_a, Job())

    async def _broken(conn: asyncpg.Connection, period: date) -> queue_repo.QueueSnapshot:
        raise asyncpg.PostgresError("metrics read failed")

    monkeypatch.setattr(queue_repo, "queue_snapshot", _broken)

    result = await ExtractionScheduler(clean_rls_db, _Arq(), _ONE_AT_A_TIME).tick()

    assert result == TickResult(admitted=1, enqueued=1)
    assert MetricNames.EXTRACTION_QUEUE_DEPTH not in collect_gauges(metric_reader)
