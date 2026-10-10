"""Acceptance tests: the extraction job end to end against an RLS-enforced Postgres.

The model is a scripted provider; everything else is real: the role-filtered
load, chunking, the pipeline, embeddings, dedupe and persistence.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from gubbi.extraction.chunking import chunk_messages
from gubbi.extraction.jobs.extract_conversation import extract_conversation
from gubbi.extraction.llm.provider import LLMRefusalError
from gubbi.extraction.pipeline import PipelineFailure
from gubbi.models.conversation import Message
from tests.fixtures.extraction_db import SeedMessage, job_ctx, seed_conversation, seed_job
from tests.fixtures.extraction_job import (
    CALL_CENTS,
    CALL_USAGE,
    ScriptedProvider,
    chunk_output,
    entry,
)

if TYPE_CHECKING:
    from uuid import UUID

    import asyncpg

    from gubbi.crypto.cipher import ContentCipher

pytestmark = [pytest.mark.asyncio(loop_scope="session"), pytest.mark.integration]

_TOPIC = "health/running"
_PLATFORM = "chatgpt"
# Months before any run of this suite, so a run-date leak cannot pass.
_SPOKEN_AT = datetime(2025, 11, 3, 9, 30, tzinfo=UTC)
_BODY_CHARS = 3_000


def _dialogue(count: int) -> list[SeedMessage]:
    return [
        SeedMessage(
            role="user" if i % 2 == 0 else "assistant",
            content=f"turn {i} " + "about my training plan " * (_BODY_CHARS // 23),
            timestamp=_SPOKEN_AT + timedelta(minutes=i),
        )
        for i in range(count)
    ]


def _overlap_index(messages: list[SeedMessage]) -> int:
    """Index of a message the chunker repeats at the start of the second chunk."""
    as_messages = [
        Message(role=m.role, content=m.content, timestamp=m.timestamp.isoformat())
        for m in messages
        if m.timestamp is not None
    ]
    first, second, *_ = chunk_messages(as_messages)
    shared = {p.index for p in first.parts} & {p.index for p in second.parts}
    assert shared, "fixture must produce overlapping chunks"
    return max(shared)


async def _entries(admin_pool: asyncpg.Pool, conversation_id: int) -> list[asyncpg.Record]:
    async with admin_pool.acquire() as conn:
        return list(
            await conn.fetch(
                "SELECT e.date, e.created_at, e.client, e.source_conversation_id"
                " FROM entries e JOIN topics t ON t.id = e.topic_id"
                " WHERE t.path = $1 AND e.source_conversation_id = $2 ORDER BY e.id",
                _TOPIC,
                conversation_id,
            )
        )


async def _job_row(admin_pool: asyncpg.Pool, job_id: UUID) -> asyncpg.Record:
    async with admin_pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM extraction_jobs WHERE id = $1", job_id)
    assert row is not None
    return row


async def test_an_entry_repeated_in_two_chunks_overlap_is_stored_once(
    app_pool: asyncpg.Pool, admin_pool: asyncpg.Pool, cipher: ContentCipher, tenant_a: UUID
) -> None:
    messages = _dialogue(70)
    shared = _overlap_index(messages)
    conv_id = await seed_conversation(admin_pool, cipher, tenant_a, messages, platform=_PLATFORM)
    repeated = entry("Signed up for the spring half marathon", source=shared)
    provider = ScriptedProvider(
        [chunk_output(_TOPIC, [repeated]), chunk_output(_TOPIC, [repeated])]
    )

    result = await extract_conversation(job_ctx(app_pool, cipher, provider), conv_id, str(tenant_a))

    assert len(provider.user_contents) == 2, "the transcript must span two chunks"
    assert f"[{shared}]" in provider.user_contents[1], "the entry's message is in the overlap"
    assert result["entries_created"] == 1
    assert len(await _entries(admin_pool, conv_id)) == 1


async def test_entries_take_their_message_date_and_the_run_time_as_ingest_time(
    app_pool: asyncpg.Pool, admin_pool: asyncpg.Pool, cipher: ContentCipher, tenant_a: UUID
) -> None:
    messages = [
        SeedMessage("user", "I booked a physio appointment for my knee.", _SPOKEN_AT),
        SeedMessage("assistant", "Good call.", _SPOKEN_AT + timedelta(minutes=1)),
    ]
    conv_id = await seed_conversation(admin_pool, cipher, tenant_a, messages, platform=_PLATFORM)
    provider = ScriptedProvider([chunk_output(_TOPIC, [entry("Booked physio", source=0)])])
    before = datetime.now(UTC)

    await extract_conversation(job_ctx(app_pool, cipher, provider), conv_id, str(tenant_a))

    (row,) = await _entries(admin_pool, conv_id)
    assert row["date"] == _SPOKEN_AT.date()
    assert row["date"] != date.today()
    assert before <= row["created_at"] <= datetime.now(UTC)
    assert row["client"] == _PLATFORM


async def test_refusal_fails_the_job_as_llm_refused_after_one_call_with_usage_recorded(
    app_pool: asyncpg.Pool, admin_pool: asyncpg.Pool, cipher: ContentCipher, tenant_a: UUID
) -> None:
    messages = [SeedMessage("user", "Tell me about my week, please.", _SPOKEN_AT)]
    conv_id = await seed_conversation(admin_pool, cipher, tenant_a, messages)
    job_id = await seed_job(app_pool, tenant_a, conv_id)
    provider = ScriptedProvider([LLMRefusalError("refused", usage=CALL_USAGE)])

    with pytest.raises(PipelineFailure):
        await extract_conversation(
            job_ctx(app_pool, cipher, provider), conv_id, str(tenant_a), str(job_id)
        )

    row = await _job_row(admin_pool, job_id)
    assert len(provider.user_contents) == 1
    assert (row["status"], row["error_code"]) == ("failed", "llm_refused")
    assert (row["input_tokens"], row["output_tokens"]) == (CALL_USAGE.input, CALL_USAGE.output)
    assert row["cached_input_tokens"] == CALL_USAGE.cached_input
    assert row["reasoning_tokens"] == CALL_USAGE.reasoning
    assert (row["llm_calls"], row["chunks"], row["cents_spent"]) == (1, 1, CALL_CENTS)


async def test_nothing_to_record_completes_the_job_with_no_entries_and_no_topic(
    app_pool: asyncpg.Pool, admin_pool: asyncpg.Pool, cipher: ContentCipher, tenant_a: UUID
) -> None:
    messages = [SeedMessage("user", "What is the capital of Peru?", _SPOKEN_AT)]
    conv_id = await seed_conversation(admin_pool, cipher, tenant_a, messages)
    job_id = await seed_job(app_pool, tenant_a, conv_id)
    provider = ScriptedProvider([chunk_output(_TOPIC, [])])

    result = await extract_conversation(
        job_ctx(app_pool, cipher, provider), conv_id, str(tenant_a), str(job_id)
    )

    row = await _job_row(admin_pool, job_id)
    assert result["entries_created"] == 0
    assert (row["status"], row["entries_created"], row["topics_created"]) == ("completed", 0, 0)
    assert row["llm_calls"] == 1
    async with admin_pool.acquire() as conn:
        topic = await conn.fetchval("SELECT id FROM topics WHERE path = $1", _TOPIC)
        processed = await conn.fetchrow(
            "SELECT processed_at, extracted_content_hash FROM conversations WHERE id = $1",
            conv_id,
        )
    assert topic is None
    assert processed is not None
    assert processed["processed_at"] is not None
    assert processed["extracted_content_hash"] is not None


async def test_system_turns_stored_before_the_role_filter_never_reach_the_provider(
    app_pool: asyncpg.Pool, admin_pool: asyncpg.Pool, cipher: ContentCipher, tenant_a: UUID
) -> None:
    messages = [
        SeedMessage("system", "STORED-SYSTEM-PROMPT-MARKER", _SPOKEN_AT),
        SeedMessage("user", "I decided to run three times a week.", _SPOKEN_AT),
        SeedMessage("tool", "STORED-TOOL-OUTPUT-MARKER", _SPOKEN_AT),
    ]
    conv_id = await seed_conversation(admin_pool, cipher, tenant_a, messages)
    provider = ScriptedProvider([chunk_output(_TOPIC, [entry("Runs three times a week")])])

    await extract_conversation(job_ctx(app_pool, cipher, provider), conv_id, str(tenant_a))

    (sent,) = provider.user_contents
    assert "I decided to run three times a week." in sent
    assert "STORED-SYSTEM-PROMPT-MARKER" not in sent
    assert "STORED-TOOL-OUTPUT-MARKER" not in sent
