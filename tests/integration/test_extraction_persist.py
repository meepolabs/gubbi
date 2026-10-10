"""``extraction.persist`` against a real RLS-enforced Postgres."""

from __future__ import annotations

import asyncio
import math
from datetime import date, timedelta
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest
from gubbi_common.db.user_scoped import user_scoped_connection

from gubbi.constants import EXTRACTION_VERSION
from gubbi.extraction.llm.provider import LLMUsage
from gubbi.extraction.persist import (
    ExtractionToPersist,
    PersistResult,
    ensure_topic,
    persist_extraction,
)
from gubbi.extraction.pipeline import ExtractedCandidate
from gubbi.storage.embedding_service import EmbeddingService
from gubbi.storage.repositories import extraction_jobs as jobs_repo
from gubbi.storage.repositories import topics as topic_repo
from tests.fixtures.tenants import seed_for

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID

    import asyncpg

    from gubbi.crypto.cipher import ContentCipher
    from tests.fixtures.tenants import TenantSeed

pytestmark = [pytest.mark.asyncio(loop_scope="session"), pytest.mark.integration]

_DIM = 384
_NEAR = 0.93
_CLIENT = "chatgpt"
_TOPIC = "tenant-a/extracted"
_HASH = "a" * 64
# Distinct values so a column/parameter swap cannot pass.
_USAGE = LLMUsage(input=101, output=202, cached_input=303, reasoning=404)
_LLM_CALLS = 5
_CHUNKS = 6
_CENTS = 7

_ENTRY_SQL = (
    "SELECT date, client, conversation_id, duplicate_of_entry_id, source_conversation_id"
    " FROM entries WHERE id = $1"
)


def _basis(axis: int) -> list[float]:
    vector = [0.0] * _DIM
    vector[axis] = 1.0
    return vector


def _near(axis: int, spill_axis: int) -> list[float]:
    """Unit vector at cosine ``_NEAR`` from ``_basis(axis)``."""
    vector = [0.0] * _DIM
    vector[axis] = _NEAR
    vector[spill_axis] = math.sqrt(1.0 - _NEAR * _NEAR)
    return vector


def _candidate(content: str, day: date) -> ExtractedCandidate:
    return ExtractedCandidate(
        content=content, why="because", tags=("t",), entry_date=day, source_message_index=None
    )


def _run(
    seed: TenantSeed,
    job_id: UUID | None,
    candidates: Sequence[ExtractedCandidate],
    embeddings: Sequence[Sequence[float]],
) -> ExtractionToPersist:
    assert seed.conversation_id is not None
    return ExtractionToPersist(
        user_id=seed.user_id,
        conversation_id=seed.conversation_id,
        job_id=job_id,
        extraction_attempt_id=str(uuid4()),
        client=_CLIENT,
        topic_path=_TOPIC,
        topic_title="Extracted",
        candidates=candidates,
        embeddings=embeddings,
        content_hash=_HASH,
        usage=_USAGE,
        llm_calls=_LLM_CALLS,
        chunks=_CHUNKS,
        cents_spent=_CENTS,
    )


async def _persist(
    app_pool: asyncpg.Pool, cipher: ContentCipher, run: ExtractionToPersist
) -> PersistResult:
    async with (
        user_scoped_connection(app_pool, user_id=run.user_id) as conn,
        conn.transaction(),
    ):
        return await persist_extraction(conn, cipher, EmbeddingService(), run)


async def _create_job(app_pool: asyncpg.Pool, seed: TenantSeed) -> UUID:
    assert seed.conversation_id is not None
    async with user_scoped_connection(app_pool, user_id=seed.user_id) as conn:
        return await jobs_repo.create_pending(
            conn, seed.user_id, seed.conversation_id, _CLIENT, period_start=date(2026, 10, 1)
        )


async def test_ensure_topic_creates_a_missing_topic(
    app_pool: asyncpg.Pool, admin_pool: asyncpg.Pool, tenant_a: UUID
) -> None:
    async with user_scoped_connection(app_pool, user_id=tenant_a) as conn:
        topic_id, created = await ensure_topic(conn, _TOPIC, "Extracted")

    async with admin_pool.acquire() as conn:
        path = await conn.fetchval("SELECT path FROM topics WHERE id = $1", topic_id)
    assert created is True
    assert path == _TOPIC


async def test_ensure_topic_returns_an_existing_topic(
    app_pool: asyncpg.Pool, seeded_a: TenantSeed
) -> None:
    async with user_scoped_connection(app_pool, user_id=seeded_a.user_id) as conn:
        topic_id, created = await ensure_topic(conn, seeded_a.topic_path, "ignored")

    assert (topic_id, created) == (seeded_a.topic_id, False)


async def test_ensure_topic_race_loser_reuses_winner_and_keeps_its_transaction(
    app_pool: asyncpg.Pool, tenant_a: UUID
) -> None:
    async with user_scoped_connection(app_pool, user_id=tenant_a) as loser:
        async with user_scoped_connection(app_pool, user_id=tenant_a) as winner:
            winner_id = await topic_repo.create(winner, _TOPIC, title="Winner")
            # Blocks on the winner's uncommitted row; fails with a unique
            # violation once the winner commits on leaving this block.
            loser_task = asyncio.create_task(ensure_topic(loser, _TOPIC, "Loser"))
            await asyncio.sleep(0.2)
            assert not loser_task.done()
        loser_result = await asyncio.wait_for(loser_task, timeout=10)
        still_usable = await loser.fetchval("SELECT 1")

    assert loser_result == (winner_id, False)
    assert still_usable == 1


async def test_persist_extraction_inserts_links_and_flags_per_dedupe_decision(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    cipher: ContentCipher,
    seeded_a: TenantSeed,
) -> None:
    today = date.today()
    yesterday = today - timedelta(days=1)
    stored_today, stored_yesterday = seeded_a.entry_ids[0], seeded_a.entry_ids[1]
    job_id = await _create_job(app_pool, seeded_a)
    # Seeded entries carry embedding _basis(0); entry i is dated today - i days.
    candidates = [
        _candidate(f"Entry 0 for {seeded_a.topic_path}", today),  # exact repeat of stored
        _candidate("Bought a bike", today),  # new
        _candidate("Purchased a bicycle", today),  # near the previous candidate
        _candidate("bought a bike.", today),  # exact repeat of an earlier candidate
        _candidate("Moved desks", yesterday),  # near a stored entry
    ]
    embeddings = [_basis(0), _basis(1), _near(1, 2), _basis(1), _near(0, 3)]

    result = await _persist(app_pool, cipher, _run(seeded_a, job_id, candidates, embeddings))

    assert len(result.entry_ids) == 3
    bike, bicycle, desks = result.entry_ids
    assert result.linked_entry_ids == (stored_today,)
    assert result.suppressed == 2
    assert result.topic_created is True
    async with admin_pool.acquire() as conn:
        rows = {
            entry_id: await conn.fetchrow(_ENTRY_SQL, entry_id) for entry_id in result.entry_ids
        }
        embedded = await conn.fetchval(
            "SELECT count(*) FROM entry_embeddings WHERE entry_id = ANY($1::int[])",
            list(result.entry_ids),
        )
        stored_link = await conn.fetchval(
            "SELECT source_conversation_id FROM entries WHERE id = $1", stored_today
        )
        entry_count = await conn.fetchval(
            "SELECT count(*) FROM entries WHERE user_id = $1", seeded_a.user_id
        )
    assert {entry_id: rows[entry_id]["duplicate_of_entry_id"] for entry_id in rows} == {
        bike: None,
        bicycle: bike,
        desks: stored_yesterday,
    }
    assert [rows[entry_id]["date"] for entry_id in (bike, bicycle, desks)] == [
        today,
        today,
        yesterday,
    ]
    for row in rows.values():
        assert row["client"] == _CLIENT
        assert row["source_conversation_id"] == seeded_a.conversation_id
        assert row["conversation_id"] is None
    assert embedded == 3
    assert stored_link == seeded_a.conversation_id
    assert entry_count == len(seeded_a.entry_ids) + 3


async def test_persist_extraction_leaves_an_already_linked_entry_alone(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    cipher: ContentCipher,
    seeded_a: TenantSeed,
) -> None:
    other = await seed_for(admin_pool, seeded_a.user_id, topic_path="tenant-a/other", n_entries=0)
    stored = seeded_a.entry_ids[0]
    async with admin_pool.acquire() as conn:
        await conn.execute(
            "UPDATE entries SET source_conversation_id = $2 WHERE id = $1",
            stored,
            other.conversation_id,
        )
    repeat = _candidate(f"Entry 0 for {seeded_a.topic_path}", date.today())

    result = await _persist(app_pool, cipher, _run(seeded_a, None, [repeat], [_basis(0)]))

    async with admin_pool.acquire() as conn:
        link = await conn.fetchval(
            "SELECT source_conversation_id FROM entries WHERE id = $1", stored
        )
    assert result.entry_ids == ()
    assert result.linked_entry_ids == ()
    assert link == other.conversation_id


async def test_persist_extraction_records_state_usage_and_audit(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    cipher: ContentCipher,
    seeded_a: TenantSeed,
) -> None:
    job_id = await _create_job(app_pool, seeded_a)
    run = _run(seeded_a, job_id, [_candidate("Bought a bike", date.today())], [_basis(1)])

    await _persist(app_pool, cipher, run)

    async with admin_pool.acquire() as conn:
        conversation = await conn.fetchrow(
            "SELECT extracted_content_hash, extraction_version, processed_at"
            " FROM conversations WHERE id = $1",
            seeded_a.conversation_id,
        )
        job = await conn.fetchrow(
            "SELECT status, topics_created, entries_created, cents_spent, input_tokens,"
            " output_tokens, cached_input_tokens, reasoning_tokens, llm_calls, chunks"
            " FROM extraction_jobs WHERE id = $1",
            job_id,
        )
        actions = await conn.fetch(
            "SELECT action, target_id FROM audit_log WHERE actor_id = $1"
            " AND action = ANY($2::text[]) ORDER BY action",
            str(seeded_a.user_id),
            ["conversation.extracted", "extraction_job.completed"],
        )
    assert conversation is not None
    assert conversation["extracted_content_hash"] == _HASH
    assert conversation["extraction_version"] == EXTRACTION_VERSION
    assert conversation["processed_at"] is not None
    assert job is not None
    assert dict(job) == {
        "status": "completed",
        "topics_created": 1,
        "entries_created": 1,
        "cents_spent": _CENTS,
        "input_tokens": _USAGE.input,
        "output_tokens": _USAGE.output,
        "cached_input_tokens": _USAGE.cached_input,
        "reasoning_tokens": _USAGE.reasoning,
        "llm_calls": _LLM_CALLS,
        "chunks": _CHUNKS,
    }
    assert [(row["action"], row["target_id"]) for row in actions] == [
        ("conversation.extracted", str(seeded_a.conversation_id)),
        ("extraction_job.completed", str(job_id)),
    ]


async def test_extraction_to_persist_rejects_unpaired_embeddings(seeded_a: TenantSeed) -> None:
    with pytest.raises(ValueError, match="one embedding per candidate"):
        _run(seeded_a, None, [_candidate("Bought a bike", date.today())], [])
