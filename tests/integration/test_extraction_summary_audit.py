"""Integration test: extraction worker writes a summary audit row (H-3/D4).

Verifies that after a successful extraction run, a
``conversation.extracted`` audit row appears with the correct
actor_type, actor_id, target_kind, target_id, and metadata.

Requires a running PostgreSQL instance with migrations applied through
0020.  Uses mock LLM service so no API key is needed.

Run with:
    pytest tests/integration/test_extraction_summary_audit.py -v
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import asyncpg
import pytest

from gubbi.extraction.jobs.extract_conversation import extract_conversation
from gubbi.extraction.service import CategorizationResult, ExtractedEntry

pytestmark = [
    pytest.mark.asyncio(loop_scope="session"),
    pytest.mark.integration,
]

_USER_UUID = UUID("11111111-2222-3333-4444-555555555555")
_USER_ID_STR = str(_USER_UUID)


async def _seed_conversation(conn: asyncpg.Connection) -> int:
    """Insert a minimal conversation row and return its id."""
    conv_id: int = await conn.fetchval(
        """
        INSERT INTO conversations (user_id, title, topic, summary, platform_id)
        VALUES ($1, 'Test conv', 'test', 'test summary', 'test-platform')
        RETURNING id
        """,
        _USER_ID_STR,
    )
    return conv_id


async def test_extraction_summary_audit_row_written(
    clean_rls_db: asyncpg.Pool,
) -> None:
    """Successful extraction produces a conversation.extracted audit row."""
    # NB: clean_rls_db yields the admin_pool.  We use a direct connection to
    # seed data and read back, but the worker function uses
    # user_scoped_connection internally (which uses the journal_app pool).
    # For this test we work exclusively with the admin pool so we can
    # bypass RLS.

    # --- Seed data ---
    async with clean_rls_db.acquire() as conn:
        conv_id = await _seed_conversation(conn)

    # --- Build mock context ---
    mock_pool = clean_rls_db  # reuse the admin pool for simplicity
    mock_cipher = MagicMock()
    mock_extraction_service = AsyncMock()

    # Stub out the LLM calls
    mock_extraction_service.categorize_conversation.return_value = CategorizationResult(
        topic_path="test/extraction-audit",
        topic_title="Extraction Audit Test",
        summary="Test summary",
        confidence=0.95,
    )
    mock_extraction_service.extract_entries.return_value = [
        ExtractedEntry(
            content="Test entry",
            reasoning="Test reasoning",
            tags=[],
            entry_date="2026-05-02",
        ),
    ]

    mock_redis = AsyncMock()
    mock_redis.publish = AsyncMock()

    ctx: dict = {
        "pool": mock_pool,
        "cipher": mock_cipher,
        "extraction_service": mock_extraction_service,
        "redis": mock_redis,
    }

    # --- Run worker ---
    result = await extract_conversation(ctx, conv_id, _USER_ID_STR)

    # --- Assert worker result ---
    assert result["skipped"] is False
    assert result["entries_created"] == 1
    assert result["topic_path"] == "test/extraction-audit"

    # --- Assert audit row ---
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

        meta = dict(row["metadata"])
        assert meta.get("via") == "extraction-worker"
        assert meta.get("entries_created") == 1
        assert meta.get("topics_touched") == 1


async def test_extraction_summary_audit_not_written_on_skip(
    clean_rls_db: asyncpg.Pool,
) -> None:
    """When the idempotency check short-circuits, no audit row is written."""
    async with clean_rls_db.acquire() as conn:
        conv_id = await _seed_conversation(conn)
        # Mark as processed to trigger skip
        await conn.execute(
            "UPDATE conversations SET processed_at = now() WHERE id = $1",
            conv_id,
        )

    mock_pool = clean_rls_db
    mock_cipher = MagicMock()
    mock_extraction_service = AsyncMock()
    mock_redis = AsyncMock()
    mock_redis.publish = AsyncMock()

    ctx: dict = {
        "pool": mock_pool,
        "cipher": mock_cipher,
        "extraction_service": mock_extraction_service,
        "redis": mock_redis,
    }

    result = await extract_conversation(ctx, conv_id, _USER_ID_STR)
    assert result["skipped"] is True

    # Verify no audit row was written for this action
    async with clean_rls_db.acquire() as conn:
        count: int = await conn.fetchval(
            "SELECT count(*) FROM audit_log WHERE action = 'conversation.extracted'"
        )
        assert count == 0
