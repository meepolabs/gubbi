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
from gubbi.storage.repositories import entries as entry_repo
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


def _candidate(
    content: str, day: date, *, why: str = "because", tags: tuple[str, ...] = ("t",)
) -> ExtractedCandidate:
    return ExtractedCandidate(
        content=content, why=why, tags=tags, entry_date=day, source_message_index=None
    )


def _run(
    seed: TenantSeed,
    job_id: UUID | None,
    candidates: Sequence[ExtractedCandidate],
    embeddings: Sequence[Sequence[float]],
    *,
    topic_path: str = _TOPIC,
) -> ExtractionToPersist:
    assert seed.conversation_id is not None
    return ExtractionToPersist(
        user_id=seed.user_id,
        conversation_id=seed.conversation_id,
        job_id=job_id,
        extraction_attempt_id=str(uuid4()),
        client=_CLIENT,
        topic_path=topic_path,
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


@pytest.mark.parametrize(
    ("title", "stored"),
    [
        pytest.param("Bike\x00 log <3!", "Bike log 3", id="control-and-symbols-stripped"),
        pytest.param("x" * 150, "x" * 100, id="capped-at-100"),
        pytest.param("<<!!>>", "Extracted", id="empty-falls-back-to-path"),
    ],
)
async def test_ensure_topic_stores_a_sanitized_title(
    app_pool: asyncpg.Pool, admin_pool: asyncpg.Pool, tenant_a: UUID, title: str, stored: str
) -> None:
    async with user_scoped_connection(app_pool, user_id=tenant_a) as conn:
        topic_id, _ = await ensure_topic(conn, _TOPIC, title)

    async with admin_pool.acquire() as conn:
        saved = await conn.fetchval("SELECT title FROM topics WHERE id = $1", topic_id)
    assert saved == stored


async def test_concurrent_runs_of_different_conversations_insert_a_shared_entry_once(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    cipher: ContentCipher,
    seeded_a: TenantSeed,
) -> None:
    other = await seed_for(admin_pool, seeded_a.user_id, topic_path="tenant-a/other", n_entries=0)
    shared = [_candidate("Bought a bike", date.today())]
    first_run = _run(seeded_a, None, shared, [_basis(1)], topic_path=seeded_a.topic_path)
    second_run = _run(other, None, shared, [_basis(1)], topic_path=seeded_a.topic_path)

    async with user_scoped_connection(app_pool, user_id=seeded_a.user_id) as first:
        first_result = await persist_extraction(first, cipher, EmbeddingService(), first_run)
        # The second run starts while the first is still uncommitted and may
        # only decide once the first commits on leaving this block.
        second_task = asyncio.create_task(_persist(app_pool, cipher, second_run))
        await asyncio.sleep(0.3)
        assert not second_task.done()
    second_result = await asyncio.wait_for(second_task, timeout=10)

    async with admin_pool.acquire() as conn:
        bike_rows = await conn.fetchval(
            "SELECT count(*) FROM entries WHERE user_id = $1 AND source_conversation_id IS NOT NULL",
            seeded_a.user_id,
        )
    assert len(first_result.entry_ids) == 1
    assert second_result.entry_ids == ()
    assert second_result.suppressed == 1
    assert bike_rows == 1


async def _lock_and_release(
    app_pool: asyncpg.Pool, user_id: UUID, day: date, client: str | None
) -> None:
    async with user_scoped_connection(app_pool, user_id=user_id) as conn:
        await entry_repo.lock_dedupe_days(conn, user_id, [day], client)


_LOCK_DAY = date(2026, 6, 1)


@pytest.mark.parametrize(
    ("held_client", "user", "day", "client", "waits"),
    [
        pytest.param(_CLIENT, "a", _LOCK_DAY, _CLIENT, True, id="same-user-day-client-waits"),
        pytest.param(_CLIENT, "a", _LOCK_DAY, None, True, id="unknown-client-waits-for-named"),
        pytest.param(None, "a", _LOCK_DAY, _CLIENT, True, id="named-client-waits-for-unknown"),
        pytest.param(None, "a", _LOCK_DAY, None, True, id="unknown-waits-for-unknown"),
        pytest.param(_CLIENT, "b", _LOCK_DAY, _CLIENT, False, id="other-user-proceeds"),
        pytest.param(
            _CLIENT, "a", _LOCK_DAY + timedelta(days=1), _CLIENT, False, id="other-day-proceeds"
        ),
        pytest.param(_CLIENT, "a", _LOCK_DAY, "claude", False, id="other-named-client-proceeds"),
    ],
)
async def test_lock_dedupe_days_blocks_exactly_the_runs_that_share_neighbours(
    app_pool: asyncpg.Pool,
    tenant_a: UUID,
    tenant_b: UUID,
    held_client: str | None,
    user: str,
    day: date,
    client: str | None,
    waits: bool,
) -> None:
    contender = tenant_a if user == "a" else tenant_b

    async with user_scoped_connection(app_pool, user_id=tenant_a) as holder:
        await entry_repo.lock_dedupe_days(holder, tenant_a, [_LOCK_DAY], held_client)
        task = asyncio.create_task(_lock_and_release(app_pool, contender, day, client))
        await asyncio.sleep(0.3)
        done_while_held = task.done()
    await asyncio.wait_for(task, timeout=10)

    assert done_while_held is not waits


_HOSTILE_TOPIC = "tenant-a/hostile"


@pytest.mark.parametrize(
    ("candidate", "content", "why", "tags"),
    [
        pytest.param(
            _candidate("Bought a bike", date.today(), tags=("a" * 200,)),
            "Bought a bike",
            "because",
            ("a" * 50,),
            id="tag-over-128-chars",
        ),
        pytest.param(
            _candidate("Bought\x00 a bike", date.today(), tags=("bi\x00ke",)),
            "Bought a bike",
            "because",
            ("bike",),
            id="nul-in-content-and-tag",
        ),
        pytest.param(
            _candidate("Bought a bike", date.today(), tags=("bike", "<>", "bike", "  ")),
            "Bought a bike",
            "because",
            ("bike",),
            id="empty-and-duplicate-tags-dropped",
        ),
        pytest.param(
            _candidate("x" * 1500, date.today(), why="y" * 6000),
            "x" * 1000,
            "y" * 5000,
            ("t",),
            id="oversize-content-and-why-capped",
        ),
    ],
)
async def test_persist_extraction_stores_candidates_as_the_mcp_write_path_would(
    app_pool: asyncpg.Pool,
    cipher: ContentCipher,
    seeded_a: TenantSeed,
    candidate: ExtractedCandidate,
    content: str,
    why: str,
    tags: tuple[str, ...],
) -> None:
    run = _run(seeded_a, None, [candidate], [_basis(1)], topic_path=_HOSTILE_TOPIC)

    result = await _persist(app_pool, cipher, run)

    async with user_scoped_connection(app_pool, user_id=seeded_a.user_id) as conn:
        _, entries, total = await entry_repo.read(conn, cipher, _HOSTILE_TOPIC)
    assert len(result.entry_ids) == 1
    assert total == 1
    (entry,) = entries
    assert (entry.id, entry.content, entry.reasoning, tuple(entry.tags)) == (
        result.entry_ids[0],
        content,
        why,
        tags,
    )


@pytest.mark.parametrize(
    "candidate",
    [
        pytest.param(
            _candidate('<parameter name="content">Bought a bike', date.today()),
            id="tool-call-syntax-in-content",
        ),
        pytest.param(
            _candidate("Bought a bike", date.today(), why="</parameter>"),
            id="tool-call-syntax-in-why",
        ),
        pytest.param(_candidate("\x00\x01 \x7f", date.today()), id="empty-after-sanitizing"),
    ],
)
async def test_persist_extraction_drops_an_unstorable_candidate_and_keeps_the_rest(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    cipher: ContentCipher,
    seeded_a: TenantSeed,
    candidate: ExtractedCandidate,
) -> None:
    kept = _candidate("Moved desks", date.today())
    run = _run(seeded_a, None, [candidate, kept], [_basis(1), _basis(2)])

    result = await _persist(app_pool, cipher, run)

    async with user_scoped_connection(app_pool, user_id=seeded_a.user_id) as conn:
        _, entries, _ = await entry_repo.read(conn, cipher, _TOPIC)
    assert [entry.content for entry in entries] == ["Moved desks"]
    assert len(result.entry_ids) == 1
    assert result.suppressed == 0


async def test_persist_extraction_dedupes_on_the_sanitized_text(
    app_pool: asyncpg.Pool, cipher: ContentCipher, seeded_a: TenantSeed
) -> None:
    candidates = [
        _candidate("Bought a bike", date.today()),
        _candidate("Bought a\x00 bike", date.today()),
    ]
    # Orthogonal embeddings: only the text can make the second a repeat.
    run = _run(seeded_a, None, candidates, [_basis(1), _basis(2)])

    result = await _persist(app_pool, cipher, run)

    assert len(result.entry_ids) == 1
    assert result.suppressed == 1
