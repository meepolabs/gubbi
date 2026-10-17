"""Ingest of re-synced threads: the stored thread is extended, never replaced.

A conversation is identified by its platform id and each message by its source
message id. A re-send that adds messages is saved and queued for extraction; a
re-send that adds none (identical, or only edited text) is skipped and queued
for nothing.
"""

from __future__ import annotations

import asyncio
from typing import Any

import asyncpg
import pytest
import structlog
from fastapi import FastAPI
from httpx import AsyncClient
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from gubbi.constants import MAX_MSG_CHARS
from gubbi.crypto.cipher import decrypt_or_raise
from gubbi.telemetry.attrs import MetricNames
from tests.fixtures.ingest_app import TEST_USER_ID, build_conv, post
from tests.fixtures.metrics import collect_gauges, metric_reader  # noqa: F401

pytestmark = pytest.mark.asyncio(loop_scope="session")

SOURCE = "extension_claude"


def _claude(platform_id: str, msg_count: int = 2, **kwargs: Any) -> dict[str, Any]:
    return build_conv(platform_id, platform="claude", msg_count=msg_count, with_ids=True, **kwargs)


async def _conversation(pool: asyncpg.Pool, platform_id: str) -> asyncpg.Record:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, processed_at, message_count FROM conversations"
            " WHERE user_id = $1 AND platform_id = $2",
            TEST_USER_ID,
            platform_id,
        )
    assert row is not None, f"no conversation {platform_id}"
    return row


async def _transcript(pool: asyncpg.Pool, app: FastAPI, platform_id: str) -> list[str]:
    cipher = app.state.app_ctx.cipher
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT m.content_encrypted, m.content_nonce FROM messages m"
            " JOIN conversations c ON c.id = m.conversation_id"
            " WHERE c.user_id = $1 AND c.platform_id = $2 ORDER BY m.position",
            TEST_USER_ID,
            platform_id,
        )
    return [
        decrypt_or_raise(cipher, bytes(r["content_encrypted"]), bytes(r["content_nonce"]))
        for r in rows
    ]


async def _active_jobs(pool: asyncpg.Pool) -> int:
    async with pool.acquire() as conn:
        count = await conn.fetchval(
            "SELECT COUNT(*) FROM extraction_jobs"
            " WHERE user_id = $1 AND status NOT IN ('completed', 'failed')",
            TEST_USER_ID,
        )
    return int(count)


async def _finish_extraction(pool: asyncpg.Pool) -> None:
    """Complete every job and mark every conversation extracted, as a worker run would."""
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE extraction_jobs SET status = 'completed', completed_at = now()"
            " WHERE user_id = $1",
            TEST_USER_ID,
        )
        await conn.execute(
            "UPDATE conversations SET processed_at = now() WHERE user_id = $1", TEST_USER_ID
        )


async def test_same_title_threads_are_stored_apart(
    client: AsyncClient, pool: asyncpg.Pool, app_with_arq: FastAPI
) -> None:
    first = _claude("title-a")
    second = _claude("title-b")
    second["messages"] = [
        {**msg, "content": f"Other {i}"} for i, msg in enumerate(second["messages"])
    ]
    assert first["title"] == second["title"]

    data = await post(client, SOURCE, first, second)

    assert (data["conversations_saved"], data["conversations_skipped_dedupe"]) == (2, 0)
    assert await _transcript(pool, app_with_arq, "title-a") == ["Message 0", "Message 1"]
    assert await _transcript(pool, app_with_arq, "title-b") == ["Other 0", "Other 1"]


async def test_appended_messages_are_saved_and_queued_once(
    client: AsyncClient, pool: asyncpg.Pool, app_with_arq: FastAPI
) -> None:
    await post(client, SOURCE, _claude("grow-1"))
    await _finish_extraction(pool)

    data = await post(client, SOURCE, _claude("grow-1", msg_count=4))

    assert (data["conversations_saved"], data["extractions_enqueued"]) == (1, 1)
    assert data["conversations_skipped_dedupe"] == 0
    assert await _active_jobs(pool) == 1
    assert await _transcript(pool, app_with_arq, "grow-1") == [f"Message {i}" for i in range(4)]
    stored = await _conversation(pool, "grow-1")
    assert (stored["message_count"], stored["processed_at"]) == (4, None)


async def test_identical_replay_is_skipped_without_a_job(
    client: AsyncClient, pool: asyncpg.Pool
) -> None:
    conv = _claude("replay-1")
    await post(client, SOURCE, conv)
    await _finish_extraction(pool)

    data = await post(client, SOURCE, conv)

    assert (data["conversations_saved"], data["extractions_enqueued"]) == (0, 0)
    assert data["conversations_skipped_dedupe"] == 1
    assert await _active_jobs(pool) == 0


async def test_edited_text_only_is_rewritten_without_a_job(
    client: AsyncClient, pool: asyncpg.Pool, app_with_arq: FastAPI
) -> None:
    await post(client, SOURCE, _claude("edit-1"))
    await _finish_extraction(pool)
    extracted_at = (await _conversation(pool, "edit-1"))["processed_at"]
    edited = _claude("edit-1")
    edited["messages"][1]["content"] = "Message 1, corrected"

    data = await post(client, SOURCE, edited)

    assert (data["conversations_saved"], data["extractions_enqueued"]) == (0, 0)
    assert data["conversations_skipped_dedupe"] == 1
    assert await _active_jobs(pool) == 0
    assert await _transcript(pool, app_with_arq, "edit-1") == ["Message 0", "Message 1, corrected"]
    assert (await _conversation(pool, "edit-1"))["processed_at"] == extracted_at


async def test_concurrent_first_sync_stores_one_thread_and_one_job(
    client: AsyncClient, pool: asyncpg.Pool
) -> None:
    conv = _claude("race-1", msg_count=5)

    results = await asyncio.gather(post(client, SOURCE, conv), post(client, SOURCE, conv))

    assert sorted(r["conversations_saved"] for r in results) == [0, 1]
    assert sorted(r["conversations_skipped_dedupe"] for r in results) == [0, 1]
    assert sum(r["extractions_skipped_error"] for r in results) == 0
    async with pool.acquire() as conn:
        rows = await conn.fetchval(
            "SELECT COUNT(*) FROM conversations WHERE user_id = $1 AND platform_id = $2",
            TEST_USER_ID,
            "race-1",
        )
    assert rows == 1
    assert (await _conversation(pool, "race-1"))["message_count"] == 5
    assert await _active_jobs(pool) == 1


async def test_unmatched_thread_is_skipped_and_logged_while_the_rest_saves(
    client: AsyncClient, pool: asyncpg.Pool, app_with_arq: FastAPI
) -> None:
    await post(client, SOURCE, build_conv("legacy-1", platform="claude"))
    await _finish_extraction(pool)

    with structlog.testing.capture_logs() as logs:
        data = await post(client, SOURCE, _claude("legacy-1", msg_count=4), _claude("fresh-1"))

    assert (data["conversations_saved"], data["extractions_enqueued"]) == (1, 1)
    assert data["conversations_skipped_dedupe"] == 1
    assert await _active_jobs(pool) == 1
    assert await _transcript(pool, app_with_arq, "legacy-1") == ["Message 0", "Message 1"]
    assert await _transcript(pool, app_with_arq, "fresh-1") == ["Message 0", "Message 1"]
    conflicts = [e for e in logs if e["event"] == "ingest_legacy_identity_conflict"]
    assert conflicts == [
        {"event": "ingest_legacy_identity_conflict", "platform": "claude", "log_level": "warning"}
    ]


async def test_linked_entry_is_written_on_first_sync_and_left_alone_on_append(
    client: AsyncClient, pool: asyncpg.Pool
) -> None:
    await post(client, SOURCE, _claude("entry-1"))
    conv_id = (await _conversation(pool, "entry-1"))["id"]
    query = "SELECT * FROM entries WHERE conversation_id = $1"
    async with pool.acquire() as conn:
        before = [dict(r) for r in await conn.fetch(query, conv_id)]

    data = await post(client, SOURCE, _claude("entry-1", msg_count=4))

    assert data["conversations_saved"] == 1
    async with pool.acquire() as conn:
        after = [dict(r) for r in await conn.fetch(query, conv_id)]
    assert len(before) == 1
    assert after == before


async def test_truncated_messages_are_counted_and_logged_per_request(
    client: AsyncClient,
    metric_reader: InMemoryMetricReader,  # noqa: F811
) -> None:
    conv = _claude("long-1", msg_count=3)
    conv["messages"][0]["content"] = "x" * (MAX_MSG_CHARS + 1)
    conv["messages"][2]["content"] = "y" * (MAX_MSG_CHARS + 50)

    with structlog.testing.capture_logs() as logs:
        await post(client, SOURCE, conv)
        await post(client, SOURCE, conv)

    counted = collect_gauges(metric_reader)[MetricNames.INGEST_MESSAGES_TRUNCATED]
    assert counted == {(): 2.0}
    completed = [e for e in logs if e["event"] == "ingest_request_completed"]
    assert [e["messages_truncated"] for e in completed] == [2, 0]
