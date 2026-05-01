"""Integration tests for TASK-02.06.1: repo INSERTs must auto-populate user_id
from the session GUC so they work under ``user_scoped_connection`` without
the caller passing user_id explicitly.
"""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

import asyncpg
import pytest
from gubbi_common.db.user_scoped import user_scoped_connection

from journalctl.core.crypto import ContentCipher
from journalctl.models.conversation import Message
from journalctl.storage.embedding_service import EmbeddingService
from journalctl.storage.repositories import conversations as conv_repo
from journalctl.storage.repositories import entries as entry_repo
from journalctl.storage.repositories import topics as topic_repo

pytestmark = pytest.mark.asyncio(loop_scope="session")


async def test_entry_insert_under_scoped_conn_binds_user_id(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    cipher: ContentCipher,
    tenant_a: UUID,
) -> None:
    """entries.append via user_scoped_connection -> INSERT populates user_id
    from the GUC. Raw SELECT via admin_pool confirms the row has the right owner."""
    async with user_scoped_connection(app_pool, user_id=tenant_a) as conn:
        topic_id = await topic_repo.create(
            conn, topic="roundtrip/repo-insert", title="Repo insert test"
        )
        entry_id = await entry_repo.append(
            conn, cipher, topic="roundtrip/repo-insert", content="test content"
        )

    async with admin_pool.acquire() as conn:
        row = await conn.fetchrow("SELECT user_id FROM entries WHERE id = $1", entry_id)
        topic_row = await conn.fetchrow("SELECT user_id FROM topics WHERE id = $1", topic_id)
    assert row is not None
    assert row["user_id"] == tenant_a
    assert topic_row is not None
    assert topic_row["user_id"] == tenant_a


async def test_repo_insert_unscoped_conn_raises_rls_violation(
    app_pool: asyncpg.Pool,
) -> None:
    """Without user_scoped_connection the GUC is unset so the sub-SELECT
    returns NULL.  RLS's WITH CHECK fires first (new-row policy) and blocks
    the insert with InsufficientPrivilegeError -- the intended default-deny."""
    _rls_re = "row-level security|row security|violates.*policy|new row violates"
    async with app_pool.acquire() as conn:
        with pytest.raises(asyncpg.PostgresError, match=_rls_re):
            await conn.execute(
                "INSERT INTO topics (path, title, description, user_id, created_at, updated_at)"
                " VALUES ('unscoped/path', 't', '',"
                " (SELECT NULLIF(current_setting('app.current_user_id', true), '')::uuid),"
                " now(), now())"
            )


async def test_conversation_insert_under_scoped_conn_binds_user_id(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    cipher: ContentCipher,
    tmp_path: Path,
    tenant_a: UUID,
) -> None:
    """save_conversation via user_scoped_connection -> INSERTs on conversations,
    messages, and entries (linked) all populate user_id from the GUC."""
    async with user_scoped_connection(app_pool, user_id=tenant_a) as conn:
        # save_conversation expects the topic to already exist
        await topic_repo.create(conn, topic="scoped-conv-test", title="Scoped conversation parent")
        messages = [
            Message(
                role="user",
                content="Hello there",
                timestamp=None,
            ),
            Message(
                role="assistant",
                content="Hi, how can I help?",
                timestamp=None,
            ),
        ]
        (
            conv_id,
            summary,
            is_update,
            linked_entry_id,
        ) = await conv_repo.save_conversation(
            conn,
            cipher,
            conversations_json_dir=tmp_path,
            topic="scoped-conv-test",
            title="Scoped conversation",
            messages=messages,
            summary="Test summary",
        )

    async with admin_pool.acquire() as conn:
        conv_row = await conn.fetchrow("SELECT user_id FROM conversations WHERE id = $1", conv_id)
        msg_rows = await conn.fetch(
            "SELECT user_id FROM messages WHERE conversation_id = $1",
            conv_id,
        )
        linked_row = await conn.fetchrow(
            "SELECT user_id FROM entries WHERE id = $1", linked_entry_id
        )
    assert conv_row is not None
    assert conv_row["user_id"] == tenant_a
    assert len(msg_rows) == 2
    assert all(r["user_id"] == tenant_a for r in msg_rows)
    assert linked_row is not None
    assert linked_row["user_id"] == tenant_a


async def test_embedding_insert_under_scoped_conn_binds_user_id(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    cipher: ContentCipher,
    tenant_a: UUID,
) -> None:
    """EmbeddingService.store_by_vector via user_scoped_connection ->
    INSERT into entry_embeddings populates user_id from the GUC.

    The repo-layer embedding INSERT is what powers journal_append_entry's
    post-commit embed step and the reindex worker. Proves user_id is
    bound correctly even when the embedding is stored as a separate
    transaction from the entries INSERT.
    """
    # store_by_vector does not load the ONNX model, so the default EmbeddingService
    # constructor is enough -- no encode() call in this test.
    embedding_service = EmbeddingService()

    async with user_scoped_connection(app_pool, user_id=tenant_a) as conn:
        await topic_repo.create(conn, topic="roundtrip/embed-check", title="Embedding user_id test")
        entry_id = await entry_repo.append(
            conn, cipher, topic="roundtrip/embed-check", content="content to embed"
        )

    unit_vector = [1.0] + [0.0] * 383
    async with user_scoped_connection(app_pool, user_id=tenant_a) as conn:
        await embedding_service.store_by_vector(conn, entry_id, unit_vector)

    async with admin_pool.acquire() as conn:
        embed_row = await conn.fetchrow(
            "SELECT user_id FROM entry_embeddings WHERE entry_id = $1", entry_id
        )
    assert embed_row is not None
    assert embed_row["user_id"] == tenant_a
