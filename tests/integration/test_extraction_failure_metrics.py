"""The failed-job counter, recorded by the extraction job against an RLS-enforced Postgres."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from redis.exceptions import ConnectionError as RedisConnectionError

from gubbi.extraction import orphan_cleanup
from gubbi.extraction.jobs import extract_conversation as job_module
from gubbi.extraction.jobs.extract_conversation import extract_conversation
from gubbi.extraction.llm.provider import LLMRateLimitError, LLMSchemaError
from gubbi.extraction.pipeline import PipelineFailure
from gubbi.extraction.retry_policy import DETERMINISTIC_MAX_ATTEMPTS, ErrorCode
from gubbi.extraction.scheduler import ExtractionScheduler, SchedulerConfig, TickResult
from gubbi.telemetry.attrs import MetricNames
from tests.fixtures.extraction_db import SeedMessage, job_ctx, seed_conversation, seed_job
from tests.fixtures.extraction_job import CALL_USAGE, ScriptedProvider
from tests.fixtures.metrics import GaugePoints, collect_gauges, metric_reader  # noqa: F401

if TYPE_CHECKING:
    from uuid import UUID

    import asyncpg

    from gubbi.crypto.cipher import ContentCipher

pytestmark = [pytest.mark.asyncio(loop_scope="session"), pytest.mark.integration]

_TRANSIENT_RETRY = (("error_class", "transient"), ("outcome", "retry"))
_MESSAGES = [SeedMessage("user", "Tell me about my week, please.")]


def _failures(reader: InMemoryMetricReader) -> GaugePoints:
    return collect_gauges(reader).get(MetricNames.EXTRACTION_JOBS_FAILED, {})


@pytest.mark.parametrize(
    ("error", "prior_attempts", "expected"),
    [
        pytest.param(
            LLMSchemaError("bad", usage=CALL_USAGE),
            DETERMINISTIC_MAX_ATTEMPTS - 1,
            (("error_class", "deterministic"), ("outcome", "terminal")),
            id="deterministic-at-attempt-cap",
        ),
        pytest.param(
            LLMSchemaError("bad", usage=CALL_USAGE),
            0,
            (("error_class", "deterministic"), ("outcome", "retry")),
            id="deterministic-first-attempt",
        ),
        pytest.param(
            LLMRateLimitError("slow"),
            DETERMINISTIC_MAX_ATTEMPTS - 1,
            _TRANSIENT_RETRY,
            id="transient",
        ),
    ],
)
async def test_a_failed_job_run_is_counted_once_by_class_and_outcome(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    cipher: ContentCipher,
    tenant_a: UUID,
    metric_reader: InMemoryMetricReader,  # noqa: F811
    error: Exception,
    prior_attempts: int,
    expected: tuple[tuple[str, str], ...],
) -> None:
    conv_id = await seed_conversation(admin_pool, cipher, tenant_a, _MESSAGES)
    job_id = await seed_job(app_pool, tenant_a, conv_id)
    await admin_pool.execute(
        "UPDATE extraction_jobs SET attempts = $2 WHERE id = $1", job_id, prior_attempts
    )

    with pytest.raises(PipelineFailure):
        await extract_conversation(
            job_ctx(app_pool, cipher, ScriptedProvider([error])),
            conv_id,
            str(tenant_a),
            str(job_id),
        )

    assert _failures(metric_reader) == {expected: 1.0}


async def test_a_failure_write_that_changes_no_row_is_not_counted(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    cipher: ContentCipher,
    tenant_a: UUID,
    metric_reader: InMemoryMetricReader,  # noqa: F811
) -> None:
    conv_id = await seed_conversation(admin_pool, cipher, tenant_a, _MESSAGES)
    job_id = await seed_job(app_pool, tenant_a, conv_id)
    await admin_pool.execute(
        "UPDATE extraction_jobs SET status = 'completed' WHERE id = $1", job_id
    )

    await job_module._mark_job_failed(
        app_pool, tenant_a, str(job_id), conv_id, ErrorCode.INTERNAL_ERROR
    )

    assert _failures(metric_reader) == {}


async def test_the_orphan_sweep_counts_each_row_it_fails_as_a_transient_retry(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    cipher: ContentCipher,
    tenant_a: UUID,
    metric_reader: InMemoryMetricReader,  # noqa: F811
) -> None:
    conv_id = await seed_conversation(admin_pool, cipher, tenant_a, _MESSAGES)
    job_id = await seed_job(app_pool, tenant_a, conv_id)
    await admin_pool.execute(
        "UPDATE extraction_jobs SET status = 'running', attempts = 1,"
        " started_at = now() - interval '1 day' WHERE id = $1",
        job_id,
    )

    assert await orphan_cleanup._sweep_running(admin_pool) == 1

    assert _failures(metric_reader) == {_TRANSIENT_RETRY: 1.0}


async def _failing_enqueue(*_args: Any, **_kwargs: Any) -> object:
    raise RedisConnectionError("redis down")


class _FailingArq:
    enqueue_job = staticmethod(_failing_enqueue)


async def test_a_lost_enqueue_is_counted_as_a_transient_retry(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    cipher: ContentCipher,
    tenant_a: UUID,
    metric_reader: InMemoryMetricReader,  # noqa: F811
) -> None:
    conv_id = await seed_conversation(admin_pool, cipher, tenant_a, _MESSAGES)
    await seed_job(app_pool, tenant_a, conv_id, admitted=False)
    config = SchedulerConfig(paused=False, budget_enabled=False, dispatch_bound=20)

    result = await ExtractionScheduler(admin_pool, _FailingArq(), config).tick()

    assert result == TickResult(admitted=1, enqueue_failed=1, enqueue_lost=1)
    assert _failures(metric_reader) == {_TRANSIENT_RETRY: 1.0}
