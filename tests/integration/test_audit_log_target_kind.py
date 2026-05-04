"""Integration test: audit_log target_kind column roundtrip (H-3).

Verifies that an audit row written with target_kind set survives a
roundtrip read-back with the correct value.  Requires a running PostgreSQL
instance with migrations applied through 0020.

Run with:
    pytest tests/integration/test_audit_log_target_kind.py -v
"""

from __future__ import annotations

import asyncpg
import pytest

from gubbi.audit import Action, record_audit

pytestmark = [
    pytest.mark.asyncio(loop_scope="session"),
    pytest.mark.integration,
]


async def test_record_audit_with_target_kind_roundtrip(
    clean_rls_db: asyncpg.Pool,
) -> None:
    """Write an audit row with target_kind, read it back, verify the column."""
    async with clean_rls_db.acquire() as conn:
        before_count: int = await conn.fetchval("SELECT count(*) FROM audit_log")

        await record_audit(
            conn,
            actor_type="user",
            actor_id="11111111-2222-3333-4444-555555555555",
            action="conversation.extracted",
            target_type="conversation",
            target_id="42",
            target_kind="conversation",
            reason="integration test",
            metadata={"via": "test"},
        )

        after_count: int = await conn.fetchval("SELECT count(*) FROM audit_log")
        assert after_count == before_count + 1

        row = await conn.fetchrow(
            "SELECT * FROM audit_log WHERE target_id = '42' ORDER BY id DESC LIMIT 1"
        )
        assert row is not None
        assert row["target_kind"] == "conversation"
        assert row["target_type"] == "conversation"
        assert row["target_id"] == "42"
        assert row["action"] == "conversation.extracted"
        assert row["actor_type"] == "user"


async def test_record_audit_without_target_kind_still_works(
    clean_rls_db: asyncpg.Pool,
) -> None:
    """Omitting target_kind (and target_id) should still produce a valid row."""
    async with clean_rls_db.acquire() as conn:
        before_count: int = await conn.fetchval("SELECT count(*) FROM audit_log")

        await record_audit(
            conn,
            actor_type="system",
            actor_id="test-worker",
            action=Action.SECRET_ROTATED,
        )

        after_count: int = await conn.fetchval("SELECT count(*) FROM audit_log")
        assert after_count == before_count + 1

        row = await conn.fetchrow(
            "SELECT * FROM audit_log WHERE actor_id = 'test-worker' ORDER BY id DESC LIMIT 1"
        )
        assert row is not None
        assert row["target_kind"] is None
        assert row["target_id"] is None


async def test_target_kind_index_works(clean_rls_db: asyncpg.Pool) -> None:
    """The idx_audit_log_target_kind_target_id index should not reject inserts."""
    async with clean_rls_db.acquire() as conn:
        # Insert two rows with same target_id but different target_kind
        await record_audit(
            conn,
            actor_type="user",
            actor_id="a",
            action="entry.created",
            target_id="42",
            target_kind="entry",
            metadata={"test": "a"},
        )
        await record_audit(
            conn,
            actor_type="user",
            actor_id="b",
            action="topic.created",
            target_id="42",
            target_kind="topic",
            metadata={"test": "b"},
        )

        rows = await conn.fetch(
            "SELECT target_kind, target_id, actor_id FROM audit_log "
            "WHERE target_id = '42' ORDER BY id"
        )
        assert len(rows) == 2
        assert rows[0]["target_kind"] == "entry"
        assert rows[1]["target_kind"] == "topic"
