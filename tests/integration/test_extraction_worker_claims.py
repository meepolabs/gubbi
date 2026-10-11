"""Integration tests: Postgres alone decides whether a delivered extraction job runs.

Arq keeps no results and retries nothing; a delivery of a job the scheduler
has not released (cancelled and backed off, or already done) must not run.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest
from arq.worker import create_worker
from gubbi_common.db.user_scoped import user_scoped_connection

from gubbi.extraction.jobs.extract_conversation import extract_conversation
from gubbi.extraction.retry_policy import ErrorCode
from gubbi.extraction.worker import WorkerSettings
from gubbi.storage.repositories import extraction_jobs as jobs_repo
from gubbi.storage.repositories import extraction_queue as queue_repo
from tests.fixtures.extraction_db import (
    SeedMessage,
    charges_of,
    job_ctx,
    seed_conversation,
    seed_job,
)
from tests.fixtures.extraction_job import ScriptedProvider, chunk_output, entry

if TYPE_CHECKING:
    from uuid import UUID

    import asyncpg

    from gubbi.crypto.cipher import ContentCipher

pytestmark = [pytest.mark.asyncio(loop_scope="session"), pytest.mark.integration]

_SPOKEN_AT = datetime(2026, 5, 10, 9, 0, tzinfo=UTC)
_SLOW_CALL_SECS = 30.0


async def _seed(admin_pool: asyncpg.Pool, cipher: ContentCipher, user_id: UUID) -> int:
    messages = [
        SeedMessage("user", "Let me tell you about my training block.", _SPOKEN_AT),
        SeedMessage("assistant", "Go on.", _SPOKEN_AT),
    ]
    return await seed_conversation(admin_pool, cipher, user_id, messages, platform="chatgpt")


def _provider(*, delay: float = 0.0) -> ScriptedProvider:
    return ScriptedProvider(
        [chunk_output("health/running", [entry("Started a block")])], delay=delay
    )


async def _job_row(admin_pool: asyncpg.Pool, job_id: UUID) -> asyncpg.Record:
    async with admin_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT status, error_code, attempts, run_after > now() AS backed_off,"
            " cents_spent, llm_calls FROM extraction_jobs WHERE id = $1",
            job_id,
        )
    assert row is not None
    return row


async def test_arq_keeps_no_results_and_never_retries_a_job() -> None:
    worker = create_worker(WorkerSettings)

    assert (worker.keep_result_s, worker.max_tries, worker.retry_jobs) == (0, 1, False)


async def test_a_cancelled_job_does_not_run_again_before_its_backoff(
    app_pool: asyncpg.Pool, admin_pool: asyncpg.Pool, cipher: ContentCipher, tenant_a: UUID
) -> None:
    conv_id = await _seed(admin_pool, cipher, tenant_a)
    job_id = await seed_job(app_pool, tenant_a, conv_id)
    slow = _provider(delay=_SLOW_CALL_SECS)
    run = asyncio.create_task(
        extract_conversation(job_ctx(app_pool, cipher, slow), conv_id, str(tenant_a), str(job_id))
    )
    while not slow.user_contents:
        await asyncio.sleep(0.01)
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run
    cancelled = await _job_row(admin_pool, job_id)
    redelivered = _provider()

    result = await extract_conversation(
        job_ctx(app_pool, cipher, redelivered), conv_id, str(tenant_a), str(job_id)
    )

    assert (cancelled["status"], cancelled["error_code"], cancelled["backed_off"]) == (
        "failed",
        ErrorCode.DEADLINE_EXCEEDED.value,
        True,
    )
    assert result["skipped"] is True
    assert redelivered.user_contents == []
    assert await _job_row(admin_pool, job_id) == cancelled


async def test_a_revived_job_of_an_already_processed_conversation_completes_with_no_spend(
    app_pool: asyncpg.Pool, admin_pool: asyncpg.Pool, cipher: ContentCipher, tenant_a: UUID
) -> None:
    conv_id = await _seed(admin_pool, cipher, tenant_a)
    job_id = await seed_job(app_pool, tenant_a, conv_id)
    async with user_scoped_connection(app_pool, user_id=tenant_a) as conn:
        assert await jobs_repo.mark_running(conn, job_id)
        await jobs_repo.mark_failed(conn, job_id, error_code=ErrorCode.WORKER_LOST)
    async with admin_pool.acquire() as conn:
        assert await queue_repo.revive_failed(conn, [job_id]) == {job_id}
        await conn.execute(
            "UPDATE extraction_jobs SET run_after = now() WHERE id = $1",
            job_id,
        )
        await conn.execute("UPDATE conversations SET processed_at = now() WHERE id = $1", conv_id)
    provider = _provider()

    result = await extract_conversation(
        job_ctx(app_pool, cipher, provider), conv_id, str(tenant_a), str(job_id)
    )

    row = await _job_row(admin_pool, job_id)
    assert result["skipped"] is True
    assert provider.user_contents == []
    assert (row["status"], row["cents_spent"], row["llm_calls"]) == ("completed", 0, 0)
    assert await charges_of(admin_pool, job_id) == []
