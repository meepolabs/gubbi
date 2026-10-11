"""Integration tests: a finished job attempt is charged to its user's wallet.

The job runs end to end against a scripted provider on the real database, so
the charge is written in the same transaction as the attempt's terminal
transition.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from gubbi.config import get_settings
from gubbi.extraction.jobs.extract_conversation import extract_conversation
from gubbi.extraction.llm.provider import LLMRefusalError
from gubbi.extraction.pipeline import PipelineFailure
from gubbi.extraction.wallet import usage_weights, weighted
from tests.fixtures.extraction_db import (
    SeedMessage,
    charges_of,
    job_ctx,
    seed_conversation,
    seed_job,
)
from tests.fixtures.extraction_job import CALL_USAGE, ScriptedProvider, chunk_output, entry

if TYPE_CHECKING:
    from collections.abc import Iterator
    from uuid import UUID

    import asyncpg

    from gubbi.crypto.cipher import ContentCipher

pytestmark = [pytest.mark.asyncio(loop_scope="session"), pytest.mark.integration]

_SPOKEN_AT = datetime(2025, 11, 3, 9, 30, tzinfo=UTC)
_TRANSCRIPT = [
    SeedMessage("user", "I signed up for the spring half marathon today.", _SPOKEN_AT),
    SeedMessage("assistant", "Great, let's plan the training.", _SPOKEN_AT),
]


@pytest.fixture(autouse=True)
def _fresh_settings() -> Iterator[None]:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _call_weighted() -> int:
    return weighted(CALL_USAGE, usage_weights())


async def _seed(
    admin_pool: asyncpg.Pool, app_pool: asyncpg.Pool, cipher: ContentCipher, user: UUID
) -> tuple[int, UUID]:
    conv_id = await seed_conversation(admin_pool, cipher, user, _TRANSCRIPT, platform="chatgpt")
    return conv_id, await seed_job(app_pool, user, conv_id)


async def test_a_completed_attempt_is_charged_its_reported_usage(
    app_pool: asyncpg.Pool, admin_pool: asyncpg.Pool, cipher: ContentCipher, tenant_a: UUID
) -> None:
    conv_id, job_id = await _seed(admin_pool, app_pool, cipher, tenant_a)
    provider = ScriptedProvider([chunk_output("health/running", [entry("Signed up")])])

    await extract_conversation(
        job_ctx(app_pool, cipher, provider), conv_id, str(tenant_a), str(job_id)
    )

    [charge] = await charges_of(admin_pool, job_id)
    assert dict(charge) | {"month": None} == {
        "attempt": 1,
        "job_kind": "backfill",
        "input_tokens": CALL_USAGE.input,
        "cached_input_tokens": CALL_USAGE.cached_input,
        "output_tokens": CALL_USAGE.output,
        "weighted": _call_weighted(),
        "monthly_part": _call_weighted(),
        "credit_part": 0,
        "month": None,
    }


async def test_a_failed_attempt_is_charged_the_usage_of_its_calls(
    app_pool: asyncpg.Pool, admin_pool: asyncpg.Pool, cipher: ContentCipher, tenant_a: UUID
) -> None:
    conv_id, job_id = await _seed(admin_pool, app_pool, cipher, tenant_a)
    provider = ScriptedProvider([LLMRefusalError("no", usage=CALL_USAGE)])

    with pytest.raises(PipelineFailure):
        await extract_conversation(
            job_ctx(app_pool, cipher, provider), conv_id, str(tenant_a), str(job_id)
        )

    charges = await charges_of(admin_pool, job_id)
    assert [(c["attempt"], c["weighted"]) for c in charges] == [(1, _call_weighted())]


async def test_a_cancelled_attempt_is_charged_the_calls_that_finished_before_it(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    cipher: ContentCipher,
    tenant_a: UUID,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Triage makes one call before extraction: it finishes, the extraction call hangs.
    monkeypatch.setenv("JOURNAL_EXTRACTION_TRIAGE", "true")
    conv_id, job_id = await _seed(admin_pool, app_pool, cipher, tenant_a)
    second_call = asyncio.Event()

    class _HangsOnSecondCall(ScriptedProvider):
        async def complete(self, *args: object, **kwargs: object) -> object:
            if self.user_contents:
                second_call.set()
                await asyncio.Event().wait()
            return await super().complete(*args, **kwargs)  # type: ignore[arg-type]

    provider = _HangsOnSecondCall([{"worth_recording": True}])
    run = asyncio.create_task(
        extract_conversation(
            job_ctx(app_pool, cipher, provider), conv_id, str(tenant_a), str(job_id)
        )
    )
    await asyncio.wait_for(second_call.wait(), 10)
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run

    charges = await charges_of(admin_pool, job_id)
    assert [(c["attempt"], c["weighted"]) for c in charges] == [(1, _call_weighted())]


async def test_a_conversation_resaved_during_the_run_gets_a_follow_up_job(
    app_pool: asyncpg.Pool, admin_pool: asyncpg.Pool, cipher: ContentCipher, tenant_a: UUID
) -> None:
    conv_id, job_id = await _seed(admin_pool, app_pool, cipher, tenant_a)

    class _ResavingProvider(ScriptedProvider):
        async def complete(self, *args: object, **kwargs: object) -> object:
            await admin_pool.execute(
                "UPDATE conversations SET updated_at = now() + interval '1 second' WHERE id = $1",
                conv_id,
            )
            return await super().complete(*args, **kwargs)  # type: ignore[arg-type]

    provider = _ResavingProvider([chunk_output("health/running", [entry("Signed up")])])

    await extract_conversation(
        job_ctx(app_pool, cipher, provider), conv_id, str(tenant_a), str(job_id)
    )

    jobs = await admin_pool.fetch(
        "SELECT id, status, lane, admitted_at FROM extraction_jobs"
        " WHERE conversation_id = $1 ORDER BY created_at",
        conv_id,
    )
    processed_at = await admin_pool.fetchval(
        "SELECT processed_at FROM conversations WHERE id = $1", conv_id
    )
    assert [
        (j["id"] == job_id, j["status"], j["lane"], j["admitted_at"] is None) for j in jobs
    ] == [
        (True, "completed", "backfill", False),
        (False, "pending", "backfill", True),
    ]
    assert processed_at is None


async def test_a_conversation_not_resaved_during_the_run_gets_no_follow_up(
    app_pool: asyncpg.Pool, admin_pool: asyncpg.Pool, cipher: ContentCipher, tenant_a: UUID
) -> None:
    conv_id, job_id = await _seed(admin_pool, app_pool, cipher, tenant_a)
    await admin_pool.execute(
        "UPDATE conversations SET updated_at = now() - $2::interval WHERE id = $1",
        conv_id,
        timedelta(hours=1),
    )
    provider = ScriptedProvider([chunk_output("health/running", [entry("Signed up")])])

    await extract_conversation(
        job_ctx(app_pool, cipher, provider), conv_id, str(tenant_a), str(job_id)
    )

    count = await admin_pool.fetchval(
        "SELECT count(*) FROM extraction_jobs WHERE conversation_id = $1", conv_id
    )
    assert count == 1
