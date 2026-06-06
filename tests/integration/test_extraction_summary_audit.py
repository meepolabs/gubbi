"""Integration test: extraction worker writes a summary audit row.

Verifies that after a successful extraction run, a
``conversation.extracted`` audit row appears with the correct
actor_type, actor_id, target_kind, target_id, and metadata.

Requires a running PostgreSQL instance with migrations applied through
0020.  Uses mock LLM service so no API key is needed.

Run with:
    pytest tests/integration/test_extraction_summary_audit.py -v
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import asyncpg
import pytest

from gubbi.crypto.cipher import ContentCipher
from gubbi.extraction.jobs.extract_conversation import extract_conversation
from gubbi.extraction.service import CategorizationResult, ExtractedEntry, ExtractionEntriesResult

pytestmark = [
    pytest.mark.asyncio(loop_scope="session"),
    pytest.mark.integration,
]

_USER_UUID = UUID("11111111-2222-3333-4444-555555555555")
_USER_ID_STR = str(_USER_UUID)
_CIPHER = ContentCipher({1: bytes([1]) * 32})


async def _seed_user_and_conversation(conn: asyncpg.Connection) -> int:
    """Seed the test user, a topic, and a minimal encrypted conversation row.

    Returns the new conversation id. Run on the admin pool (BYPASSRLS) so the
    seed bypasses RLS; the worker reads it back RLS-scoped via journal_app.
    """
    await conn.execute(
        """
        INSERT INTO users (id, email, timezone, created_at, updated_at)
        VALUES ($1, 'summary-audit@test.local', 'UTC', now(), now())
        ON CONFLICT (id) DO NOTHING
        """,
        _USER_UUID,
    )
    title_ct, title_nonce = _CIPHER.encrypt("Test conv")
    summary_ct, summary_nonce = _CIPHER.encrypt("test summary")
    topic_id: int = await conn.fetchval(
        """
        INSERT INTO topics (path, title, description, user_id, created_at, updated_at)
        VALUES ('inbox-' || gen_random_uuid()::text, 'Inbox', '', $1, now(), now())
        RETURNING id
        """,
        _USER_UUID,
    )
    conv_id: int = await conn.fetchval(
        """
        INSERT INTO conversations
            (topic_id, user_id, title_encrypted, title_nonce, slug, source,
             summary_encrypted, summary_nonce, tags, participants,
             message_count, created_at, updated_at, json_path, search_vector,
             platform, platform_id)
        VALUES ($1, $2, $3, $4, gen_random_uuid()::text, 'chatgpt',
                $5, $6, '{}', '{}', 0, now(), now(), 'test.json',
                to_tsvector('english', 'test'),
                'chatgpt', 'test-platform')
        RETURNING id
        """,
        topic_id,
        _USER_UUID,
        title_ct,
        title_nonce,
        summary_ct,
        summary_nonce,
    )
    return conv_id


async def test_extraction_summary_audit_row_written(
    app_pool: asyncpg.Pool,
    clean_rls_db: asyncpg.Pool,
) -> None:
    """Successful extraction produces a conversation.extracted audit row.

    The worker writes the audit row with actor_type='user' via
    user_scoped_connection, so ctx['pool'] MUST be the journal_app pool: under
    journal_admin the cross-attribution trigger would block the write. Seed and
    read-back use the admin pool (clean_rls_db) -- journal_app has no SELECT on
    audit_log (append-only).
    """
    # --- Seed data via admin (BYPASSRLS) ---
    async with clean_rls_db.acquire() as conn:
        conv_id = await _seed_user_and_conversation(conn)

    # --- Build context: worker runs under journal_app for the user-actor audit ---
    mock_extraction_service = AsyncMock()

    # Stub out the LLM calls
    mock_extraction_service.categorize_conversation.return_value = CategorizationResult(
        topic_path="test/extraction-audit",
        topic_title="Extraction Audit Test",
        summary="Test summary",
        confidence=0.95,
    )
    mock_extraction_service.extract_entries.return_value = ExtractionEntriesResult(
        entries=[
            ExtractedEntry(
                content="Test entry",
                reasoning="Test reasoning",
                tags=[],
                entry_date="2026-05-02",
            ),
        ],
        input_tokens=100,
        output_tokens=50,
    )

    mock_redis = AsyncMock()
    mock_redis.publish = AsyncMock()

    ctx: dict = {
        "pool": app_pool,
        "cipher": _CIPHER,
        "extraction_service": mock_extraction_service,
        "redis": mock_redis,
    }

    # --- Run worker ---
    result = await extract_conversation(ctx, conv_id, _USER_ID_STR)

    # --- Assert worker result ---
    assert result["skipped"] is False
    assert result["entries_created"] == 1
    assert result["topic_path"] == "test/extraction-audit"

    # --- Assert audit row (read via admin -- journal_app cannot SELECT audit_log) ---
    async with clean_rls_db.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT actor_type, actor_id, action, target_kind, target_id, metadata "
            "FROM audit_log "
            "WHERE action = 'conversation.extracted' "
            "ORDER BY id DESC LIMIT 1"
        )
        assert row is not None, "No conversation.extracted audit row found"

        assert row["actor_type"] == "user"
        assert row["actor_id"] == _USER_ID_STR
        assert row["target_kind"] == "conversation"
        assert row["target_id"] == str(conv_id)

        raw_meta = row["metadata"]
        meta = json.loads(raw_meta) if isinstance(raw_meta, str) else dict(raw_meta)
        assert meta.get("via") == "extraction-worker"
        assert meta.get("entries_created") == 1
        assert meta.get("topics_touched") == 1


async def test_extraction_summary_audit_not_written_on_skip(
    app_pool: asyncpg.Pool,
    clean_rls_db: asyncpg.Pool,
) -> None:
    """When the idempotency check short-circuits, no audit row is written."""
    async with clean_rls_db.acquire() as conn:
        conv_id = await _seed_user_and_conversation(conn)
        # Mark as processed to trigger skip
        await conn.execute(
            "UPDATE conversations SET processed_at = now() WHERE id = $1",
            conv_id,
        )

    mock_cipher = MagicMock()
    mock_extraction_service = AsyncMock()
    mock_redis = AsyncMock()
    mock_redis.publish = AsyncMock()

    ctx: dict = {
        "pool": app_pool,
        "cipher": mock_cipher,
        "extraction_service": mock_extraction_service,
        "redis": mock_redis,
    }

    result = await extract_conversation(ctx, conv_id, _USER_ID_STR)
    assert result["skipped"] is True

    # Verify no audit row was written for this conversation.
    async with clean_rls_db.acquire() as conn:
        count: int = await conn.fetchval(
            "SELECT count(*) FROM audit_log "
            "WHERE action = 'conversation.extracted' AND target_id = $1",
            str(conv_id),
        )
        assert count == 0
