"""Integration checks for encryption-at-rest behavior and search outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import UUID

import asyncpg
import pytest
import pytest_asyncio
import structlog
from gubbi_common.db.user_scoped import user_scoped_connection
from mcp.server.fastmcp import FastMCP

from journalctl.config import get_settings
from journalctl.core.auth_context import current_user_id
from journalctl.core.context import AppContext
from journalctl.core.crypto import ContentCipher
from journalctl.models.conversation import Message
from journalctl.storage.embedding_service import EmbeddingService
from journalctl.storage.repositories import conversations as conv_repo
from journalctl.storage.repositories import entries as entry_repo
from journalctl.storage.repositories import topics as topic_repo
from journalctl.tools.registry import register_tools

pytestmark = pytest.mark.asyncio(loop_scope="session")


class _StaticEmbeddingService(EmbeddingService):
    def __init__(self) -> None:
        self._vector = [1.0] + [0.0] * 383

    def encode(self, text: str) -> list[float]:
        return list(self._vector)


async def _with_user(user_id: UUID, coro: Any) -> Any:
    token = current_user_id.set(user_id)
    try:
        return await coro
    finally:
        current_user_id.reset(token)


async def _insert_user(admin_pool: asyncpg.Pool, email: str) -> UUID:
    async with admin_pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO users (id, email, timezone, created_at, updated_at)
            VALUES (gen_random_uuid(), $1, 'UTC', now(), now())
            RETURNING id
            """,
            email,
        )
    if row is None:
        raise RuntimeError("failed to insert test user")
    return UUID(str(row["id"]))


@pytest_asyncio.fixture
async def search_tools(
    app_pool: asyncpg.Pool, cipher: ContentCipher
) -> tuple[dict, EmbeddingService]:
    embedding_service = _StaticEmbeddingService()
    app_ctx = AppContext(
        pool=app_pool,
        embedding_service=embedding_service,
        settings=get_settings(),
        logger=structlog.get_logger("test"),
        cipher=cipher,
    )
    mcp = FastMCP("test-encryption-correctness")
    register_tools(mcp, app_ctx)
    return ({name: tool.fn for name, tool in mcp._tool_manager._tools.items()}, embedding_service)


async def test_entry_storage_uses_ciphertext_and_tsvector(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    cipher: ContentCipher,
) -> None:
    user_id = await _insert_user(admin_pool, "enc-correct-entry@test.local")
    plaintext = "orchid ranking signal orchid orchid"

    async with user_scoped_connection(app_pool, user_id=user_id) as conn:
        await topic_repo.create(conn, topic="enc/entries", title="Encryption Entries")
        entry_id = await entry_repo.append(conn, cipher, topic="enc/entries", content=plaintext)

    async with admin_pool.acquire() as conn:
        entries_search_text_exists = await conn.fetchval(
            """
            SELECT EXISTS (
              SELECT 1
              FROM information_schema.columns
              WHERE table_schema = 'public' AND table_name = 'entries' AND column_name = 'search_text'
            )
            """
        )
        messages_search_text_exists = await conn.fetchval(
            """
            SELECT EXISTS (
              SELECT 1
              FROM information_schema.columns
              WHERE table_schema = 'public' AND table_name = 'messages' AND column_name = 'search_text'
            )
            """
        )
        row = await conn.fetchrow(
            "SELECT content_encrypted, search_vector::text AS sv FROM entries WHERE id = $1",
            entry_id,
        )

    assert entries_search_text_exists is False
    assert messages_search_text_exists is False
    assert row is not None
    encrypted = bytes(row["content_encrypted"])
    assert encrypted
    assert encrypted != plaintext.encode("utf-8")
    assert row["sv"] is not None


async def test_conversation_title_summary_are_encrypted(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    cipher: ContentCipher,
    tmp_journal: Path,
) -> None:
    user_id = await _insert_user(admin_pool, "enc-correct-conv@test.local")
    title = "Therapy session 2026-04-15"
    summary = "Discussed anxiety triggers and next steps"

    async with user_scoped_connection(app_pool, user_id=user_id) as conn:
        await topic_repo.create(conn, topic="enc/conversations", title="Encryption Conversations")
        conv_id, _summary, _is_update, _linked_entry_id = await conv_repo.save_conversation(
            conn,
            cipher,
            conversations_json_dir=tmp_journal / "conversations_json",
            topic="enc/conversations",
            title=title,
            messages=[
                Message(role="user", content="I feel overwhelmed", timestamp=None),
                Message(role="assistant", content="Lets break this down", timestamp=None),
            ],
            summary=summary,
        )

    async with admin_pool.acquire() as conn:
        title_exists = await conn.fetchval(
            """
            SELECT EXISTS (
              SELECT 1
              FROM information_schema.columns
              WHERE table_schema = 'public' AND table_name = 'conversations' AND column_name = 'title'
            )
            """
        )
        summary_exists = await conn.fetchval(
            """
            SELECT EXISTS (
              SELECT 1
              FROM information_schema.columns
              WHERE table_schema = 'public' AND table_name = 'conversations' AND column_name = 'summary'
            )
            """
        )
        row = await conn.fetchrow(
            """
            SELECT title_encrypted, title_nonce, summary_encrypted, summary_nonce
            FROM conversations
            WHERE id = $1
            """,
            conv_id,
        )

    assert title_exists is False
    assert summary_exists is False
    assert row is not None
    assert bytes(row["title_encrypted"])
    assert bytes(row["summary_encrypted"])
    assert len(bytes(row["title_nonce"])) == 12
    assert len(bytes(row["summary_nonce"])) == 12

    async with user_scoped_connection(app_pool, user_id=user_id) as conn:
        meta, _messages, _total = await conv_repo.read_conversation_by_id(conn, cipher, conv_id)
    assert meta.title == title
    assert meta.summary == summary


async def test_journal_search_returns_full_content_for_fts_and_semantic(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    cipher: ContentCipher,
    search_tools: tuple[dict, EmbeddingService],
) -> None:
    tools, embedding_service = search_tools
    user_id = await _insert_user(admin_pool, "enc-correct-search@test.local")

    async with user_scoped_connection(app_pool, user_id=user_id) as conn:
        await topic_repo.create(conn, topic="enc/search", title="Encryption Search")
        strong_entry_id = await entry_repo.append(
            conn,
            cipher,
            topic="enc/search",
            content="orchid orchid orchid ranked content block",
        )
        weak_entry_id = await entry_repo.append(
            conn,
            cipher,
            topic="enc/search",
            content="orchid once",
        )
        conv_id, _summary, _is_update, _linked_entry_id = await conv_repo.save_conversation(
            conn,
            cipher,
            conversations_json_dir=get_settings().conversations_json_dir,
            topic="enc/search",
            title="Therapy session orchid",
            messages=[Message(role="user", content="Need coping plan", timestamp=None)],
            summary="Full summary for search tool output",
        )
        await embedding_service.store_by_vector(conn, strong_entry_id, [1.0] + [0.0] * 383)
        await embedding_service.store_by_vector(conn, weak_entry_id, [0.0, 1.0] + [0.0] * 382)

    fts_result = await _with_user(user_id, tools["journal_search"](query="orchid", limit=10))
    assert fts_result["total"] >= 2

    positions: dict[int, int] = {}
    for idx, result in enumerate(fts_result["results"]):
        if result.get("entry_id") is not None:
            positions[int(result["entry_id"])] = idx
            assert "content" in result
            assert "<b>" not in result.get("content", "")
        if result.get("conversation_id") == conv_id:
            assert result.get("title") == "Therapy session orchid"
            assert result.get("summary") == "Full summary for search tool output"
            assert "<b>" not in (result.get("title", "") + result.get("summary", ""))

    assert strong_entry_id in positions
    assert weak_entry_id in positions
    assert positions[strong_entry_id] < positions[weak_entry_id]

    semantic_result = await _with_user(
        user_id,
        tools["journal_search"](query="semantic-vector-only-token", limit=5),
    )
    assert semantic_result["total"] >= 1
    semantic_rows = [r for r in semantic_result["results"] if r.get("entry_id") is not None]
    assert semantic_rows
    assert semantic_rows[0].get("content")
