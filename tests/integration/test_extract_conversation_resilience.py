"""Integration tests: extraction worker resilience.

Tests cover:
- Idempotent re-entry after an LLM failure (retrying from scratch produces exactly N entries).
- Persistence rollback leaves no partial state.
- Pool not starved under concurrency (10 concurrent jobs against pool max=12).
- Multi-worker race: both workers for same conversation_id, exactly one writes.
- Duplicate execution of one job row: only the execution that claims it calls the model.
- Lifecycle UPDATEs: mark_running -> mark_completed / mark_failed.
- Audit rows at terminals: extraction_job.completed / extraction_job.failed.
- Retry idempotency: lifecycle UPDATEs are no-ops on second invocation.

Requires a running PostgreSQL instance with migrations applied through head.
Uses a scripted LLM provider -- no endpoint needed.

Run with:
    pytest tests/integration/test_extract_conversation_resilience.py -v
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any
from unittest.mock import patch
from uuid import UUID

import asyncpg
import pytest
import pytest_asyncio

from gubbi.constants import APP_POOL_SIZE_MAX
from gubbi.crypto.cipher import ContentCipher
from gubbi.extraction.jobs.extract_conversation import extract_conversation
from gubbi.extraction.llm.provider import LLMPermanentError
from gubbi.extraction.pipeline import PipelineFailure
from tests.fixtures.extraction_db import SeedMessage, job_ctx, seed_conversation, seed_job
from tests.fixtures.extraction_job import CALL_CENTS, ScriptedProvider, chunk_output, entry

pytestmark = [
    pytest.mark.asyncio(loop_scope="session"),
    pytest.mark.integration,
]

# Fixed test user + cipher.
_USER_UUID = UUID("22222222-3333-4444-5555-666666666666")
_USER_ID_STR = str(_USER_UUID)
_CIPHER = ContentCipher({1: bytes([1]) * 32})
_TOPIC = "test/resilience"
_SPOKEN_AT = datetime(2026, 5, 10, 9, 0, tzinfo=UTC)
_LLM_LATENCY_SECS = 0.1


@pytest_asyncio.fixture(autouse=True)
async def _seed_user(clean_rls_db: asyncpg.Pool) -> AsyncIterator[None]:
    """Seed the fixed test user before each test.

    clean_rls_db TRUNCATEs users (RESTART IDENTITY CASCADE) before yielding, so
    the user must be re-inserted per test. topics / conversations / extraction_jobs
    all FK to users(id), so without this every seed helper fails with a
    foreign-key violation.
    """
    async with clean_rls_db.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO users (id, email, timezone, created_at, updated_at)
            VALUES ($1, 'resilience-e2e@test.local', 'UTC', now(), now())
            ON CONFLICT (id) DO NOTHING
            """,
            _USER_UUID,
        )
    yield


async def _seed_conversation(admin_pool: asyncpg.Pool, text: str = "my training") -> int:
    messages = [
        SeedMessage("user", f"Let me tell you about {text}.", _SPOKEN_AT),
        SeedMessage("assistant", "Go on.", _SPOKEN_AT),
    ]
    return await seed_conversation(admin_pool, _CIPHER, _USER_UUID, messages, platform="claude")


def _two_entries(topic: str = _TOPIC) -> dict[str, Any]:
    return chunk_output(topic, [entry("Resilience entry one"), entry("Resilience entry two")])


def _ctx(app_pool: asyncpg.Pool, provider: ScriptedProvider) -> Any:
    return job_ctx(app_pool, _CIPHER, provider)


async def _count_topic_entries(admin_pool: asyncpg.Pool, topic: str) -> int:
    async with admin_pool.acquire() as conn:
        count: int = await conn.fetchval(
            "SELECT COUNT(*) FROM entries e JOIN topics t ON t.id = e.topic_id WHERE t.path = $1",
            topic,
        )
    return count


async def _processed_at(admin_pool: asyncpg.Pool, conv_id: int) -> datetime | None:
    async with admin_pool.acquire() as conn:
        value: datetime | None = await conn.fetchval(
            "SELECT processed_at FROM conversations WHERE id = $1", conv_id
        )
    return value


async def _terminal_audit_actor(admin_pool: asyncpg.Pool, action: str, job_id: UUID) -> Any:
    async with admin_pool.acquire() as conn:
        return await conn.fetchrow(
            "SELECT actor_type, actor_id, target_kind FROM audit_log"
            " WHERE action = $1 AND target_id = $2",
            action,
            str(job_id),
        )


async def test_idempotent_reentry_after_llm_failure(
    clean_rls_db: asyncpg.Pool,
    app_pool: asyncpg.Pool,
) -> None:
    """A failed run leaves the conversation unprocessed; the retry writes exactly N entries."""
    conv_id = await _seed_conversation(clean_rls_db)

    with pytest.raises(PipelineFailure):
        await extract_conversation(
            _ctx(app_pool, ScriptedProvider([LLMPermanentError("first run")])),
            conv_id,
            _USER_ID_STR,
        )
    assert await _processed_at(clean_rls_db, conv_id) is None

    result = await extract_conversation(
        _ctx(app_pool, ScriptedProvider([_two_entries()])), conv_id, _USER_ID_STR
    )

    assert result["skipped"] is False
    assert result["entries_created"] == 2
    assert await _count_topic_entries(clean_rls_db, _TOPIC) == 2
    assert await _processed_at(clean_rls_db, conv_id) is not None


async def test_persistence_rollback_leaves_no_partial_state(
    clean_rls_db: asyncpg.Pool,
    app_pool: asyncpg.Pool,
) -> None:
    """A write failure on the second entry rolls back the first, the topic and processed_at."""
    conv_id = await _seed_conversation(clean_rls_db)
    calls = 0

    async def _flaky_append(*args: object, **kwargs: object) -> int:
        nonlocal calls
        calls += 1
        if calls >= 2:
            raise RuntimeError("Simulated DB write failure on second entry")
        return await real_append(*args, **kwargs)  # type: ignore[arg-type]

    from gubbi.storage.repositories import entries as entry_repo

    real_append = entry_repo.append
    with (
        patch("gubbi.storage.repositories.entries.append", new=_flaky_append),
        pytest.raises(RuntimeError, match="Simulated DB write failure"),
    ):
        await extract_conversation(
            _ctx(app_pool, ScriptedProvider([_two_entries()])), conv_id, _USER_ID_STR
        )

    assert calls == 2
    assert await _count_topic_entries(clean_rls_db, _TOPIC) == 0
    assert await _processed_at(clean_rls_db, conv_id) is None
    async with clean_rls_db.acquire() as conn:
        topic = await conn.fetchval("SELECT id FROM topics WHERE path = $1", _TOPIC)
    assert topic is None


async def test_pool_not_starved_under_concurrency(
    clean_rls_db: asyncpg.Pool,
    app_pool: asyncpg.Pool,
) -> None:
    """Ten concurrent jobs whose LLM calls take time all complete against a 12-slot pool.

    No connection is held during the LLM call, so APP_POOL_SIZE_MAX=12 is
    sufficient headroom above max_jobs=10.
    """
    assert APP_POOL_SIZE_MAX == 12, f"Expected APP_POOL_SIZE_MAX=12, got {APP_POOL_SIZE_MAX}"
    conv_ids = [await _seed_conversation(clean_rls_db, f"plan {i}") for i in range(10)]

    async def _run_one(index: int, conv_id: int) -> dict[str, Any]:
        provider = ScriptedProvider(
            [chunk_output("test/concurrency", [entry(f"Entry for job {index}")])],
            delay=_LLM_LATENCY_SECS,
        )
        return await extract_conversation(_ctx(app_pool, provider), conv_id, _USER_ID_STR)

    results = await asyncio.gather(*[_run_one(i, cid) for i, cid in enumerate(conv_ids)])

    assert [r["skipped"] for r in results] == [False] * 10
    assert await _count_topic_entries(clean_rls_db, "test/concurrency") == 10


async def test_multi_worker_race_idempotent(
    clean_rls_db: asyncpg.Pool,
    app_pool: asyncpg.Pool,
) -> None:
    """Two workers for one conversation both reach persistence: exactly one writes.

    Both pass the Phase 1 check before either commits; the row lock taken
    in the persistence transaction serializes the recheck, so the second
    sees the first's processed_at and writes nothing.
    """
    conv_id = await _seed_conversation(clean_rls_db)
    providers = [
        ScriptedProvider(
            [chunk_output("test/race", [entry("Race entry")])], delay=_LLM_LATENCY_SECS
        )
        for _ in range(2)
    ]

    results = await asyncio.gather(
        *(extract_conversation(_ctx(app_pool, p), conv_id, _USER_ID_STR) for p in providers)
    )

    assert [len(p.user_contents) for p in providers] == [1, 1], "both reached the model"
    assert sorted(r["skipped"] for r in results) == [False, True]
    async with clean_rls_db.acquire() as conn:
        audit_count: int = await conn.fetchval(
            "SELECT COUNT(*) FROM audit_log WHERE action = 'conversation.extracted'"
            " AND target_id = $1",
            str(conv_id),
        )
    assert audit_count == 1
    assert await _count_topic_entries(clean_rls_db, "test/race") == 1
    assert await _processed_at(clean_rls_db, conv_id) is not None


async def test_a_second_execution_of_a_running_job_makes_no_model_call(
    clean_rls_db: asyncpg.Pool,
    app_pool: asyncpg.Pool,
) -> None:
    """Two executions of one job row: the one that does not claim it returns untouched."""
    conv_id = await _seed_conversation(clean_rls_db)
    job_id = await seed_job(app_pool, _USER_UUID, conv_id)
    providers = [
        ScriptedProvider([_two_entries()], delay=_LLM_LATENCY_SECS),
        ScriptedProvider([_two_entries()], delay=_LLM_LATENCY_SECS),
    ]

    results = await asyncio.gather(
        *(
            extract_conversation(_ctx(app_pool, p), conv_id, _USER_ID_STR, str(job_id))
            for p in providers
        )
    )

    assert sorted(len(p.user_contents) for p in providers) == [0, 1]
    assert sorted(r["skipped"] for r in results) == [False, True]
    assert await _count_topic_entries(clean_rls_db, _TOPIC) == 2
    async with clean_rls_db.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT status, llm_calls FROM extraction_jobs WHERE id = $1", job_id
        )
    assert row is not None
    assert (row["status"], row["llm_calls"]) == ("completed", 1)


# ---------------------------------------------------------------------------
# Lifecycle UPDATE integration tests
# ---------------------------------------------------------------------------


async def test_lifecycle_happy_path(
    clean_rls_db: asyncpg.Pool,
    app_pool: asyncpg.Pool,
) -> None:
    """pending -> running -> completed, with counters and a user-actor audit row."""
    conv_id = await _seed_conversation(clean_rls_db)
    job_id = await seed_job(app_pool, _USER_UUID, conv_id)

    result = await extract_conversation(
        _ctx(app_pool, ScriptedProvider([_two_entries()])), conv_id, _USER_ID_STR, str(job_id)
    )

    assert result["skipped"] is False
    assert result["entries_created"] == 2
    assert result["cents_spent"] == CALL_CENTS
    async with clean_rls_db.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT status, entries_created, topics_created, cents_spent, llm_calls,"
            " started_at, completed_at FROM extraction_jobs WHERE id = $1",
            job_id,
        )
    assert row is not None
    assert row["status"] == "completed"
    assert (row["entries_created"], row["topics_created"]) == (2, 1)
    assert (row["cents_spent"], row["llm_calls"]) == (CALL_CENTS, 1)
    assert row["started_at"] is not None
    assert row["completed_at"] is not None
    audit = await _terminal_audit_actor(clean_rls_db, "extraction_job.completed", job_id)
    assert audit is not None
    assert (audit["actor_type"], audit["actor_id"]) == ("user", _USER_ID_STR)
    assert audit["target_kind"] == "extraction_job"


async def test_lifecycle_on_failure(
    clean_rls_db: asyncpg.Pool,
    app_pool: asyncpg.Pool,
) -> None:
    """pending -> running -> failed, with error_code and a user-actor audit row."""
    conv_id = await _seed_conversation(clean_rls_db)
    job_id = await seed_job(app_pool, _USER_UUID, conv_id)

    with pytest.raises(PipelineFailure):
        await extract_conversation(
            _ctx(app_pool, ScriptedProvider([LLMPermanentError("hard failure")])),
            conv_id,
            _USER_ID_STR,
            str(job_id),
        )

    async with clean_rls_db.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT status, error_code, started_at, completed_at FROM extraction_jobs"
            " WHERE id = $1",
            job_id,
        )
    assert row is not None
    assert (row["status"], row["error_code"]) == ("failed", "llm_provider_error")
    assert row["started_at"] is not None
    assert row["completed_at"] is not None
    audit = await _terminal_audit_actor(clean_rls_db, "extraction_job.failed", job_id)
    assert audit is not None
    assert (audit["actor_type"], audit["actor_id"]) == ("user", _USER_ID_STR)
    assert audit["target_kind"] == "extraction_job"


async def test_lifecycle_retry_idempotency(
    clean_rls_db: asyncpg.Pool,
    app_pool: asyncpg.Pool,
) -> None:
    """A second run of a completed job short-circuits and leaves the row completed."""
    conv_id = await _seed_conversation(clean_rls_db)
    job_id = await seed_job(app_pool, _USER_UUID, conv_id)

    first = await extract_conversation(
        _ctx(app_pool, ScriptedProvider([_two_entries()])), conv_id, _USER_ID_STR, str(job_id)
    )
    second = await extract_conversation(
        _ctx(app_pool, ScriptedProvider([])), conv_id, _USER_ID_STR, str(job_id)
    )

    assert (first["skipped"], second["skipped"]) == (False, True)
    async with clean_rls_db.acquire() as conn:
        status = await conn.fetchval("SELECT status FROM extraction_jobs WHERE id = $1", job_id)
        completed_audits: int = await conn.fetchval(
            "SELECT COUNT(*) FROM audit_log"
            " WHERE action = 'extraction_job.completed' AND target_id = $1",
            str(job_id),
        )
    assert status == "completed"
    assert completed_audits == 1
