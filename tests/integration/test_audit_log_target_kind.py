"""Integration test: audit_log target_kind column roundtrip.

Verifies that an audit row written with target_kind set survives a
roundtrip read-back with the correct value.  Requires a running PostgreSQL
instance with migrations applied through 0020.

Run with:
    pytest tests/integration/test_audit_log_target_kind.py -v
"""

from __future__ import annotations

from uuid import UUID

import asyncpg
import pytest
from gubbi_common.db.user_scoped import user_scoped_connection

from gubbi.audit import Action, record_audit

pytestmark = [
    pytest.mark.asyncio(loop_scope="session"),
    pytest.mark.integration,
]

# actor_type='user' audit rows must be written via the journal_app role under
# user_scoped_connection: the cross-attribution trigger blocks journal_admin
# from inserting actor_type='user', and the RLS WITH CHECK policy requires
# actor_id == app.current_user_id. journal_app has no SELECT on audit_log
# (append-only), so read-backs go through the admin pool.
_ACTOR_A = UUID("11111111-2222-3333-4444-555555555555")
_ACTOR_B = UUID("00000000-0000-0000-0000-0000000000aa")
_ACTOR_C = UUID("00000000-0000-0000-0000-0000000000bb")


async def test_record_audit_with_target_kind_roundtrip(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
) -> None:
    """Write an audit row with target_kind, read it back, verify the column."""
    async with admin_pool.acquire() as conn:
        before_count: int = await conn.fetchval("SELECT count(*) FROM audit_log")

    async with user_scoped_connection(app_pool, user_id=_ACTOR_A) as conn:
        await record_audit(
            conn,
            actor_type="user",
            actor_id=str(_ACTOR_A),
            action="conversation.extracted",
            target_type="conversation",
            target_id="00000000-0000-0000-0000-000000000042",
            target_kind="conversation",
            reason="integration test",
            metadata={"via": "test"},
        )

    async with admin_pool.acquire() as conn:
        after_count: int = await conn.fetchval("SELECT count(*) FROM audit_log")
        assert after_count == before_count + 1

        row = await conn.fetchrow(
            "SELECT * FROM audit_log WHERE target_id = '00000000-0000-0000-0000-000000000042' "
            "ORDER BY id DESC LIMIT 1"
        )
        assert row is not None
        assert row["target_kind"] == "conversation"
        assert row["target_type"] == "conversation"
        assert row["target_id"] == "00000000-0000-0000-0000-000000000042"
        assert row["action"] == "conversation.extracted"
        assert row["actor_type"] == "user"


async def test_record_audit_without_target_kind_still_works(
    admin_pool: asyncpg.Pool,
) -> None:
    """Omitting target_kind (and target_id) should still produce a valid row.

    Uses actor_type='system' written through the admin pool -- the
    cross-attribution trigger only blocks actor_type='user' from journal_admin.
    """
    async with admin_pool.acquire() as conn:
        before_count: int = await conn.fetchval("SELECT count(*) FROM audit_log")

        await record_audit(
            conn,
            actor_type="system",
            actor_id="system:test-worker",
            action=Action.SECRET_ROTATED,
        )

        after_count: int = await conn.fetchval("SELECT count(*) FROM audit_log")
        assert after_count == before_count + 1

        row = await conn.fetchrow(
            "SELECT * FROM audit_log WHERE actor_id = 'system:test-worker' "
            "ORDER BY id DESC LIMIT 1"
        )
        assert row is not None
        assert row["target_kind"] is None
        assert row["target_id"] is None


async def test_target_kind_index_works(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
) -> None:
    """The idx_audit_log_target_kind_target_id index should not reject inserts."""
    shared_target = "00000000-0000-0000-0000-0000000000ff"
    async with user_scoped_connection(app_pool, user_id=_ACTOR_B) as conn:
        await record_audit(
            conn,
            actor_type="user",
            actor_id=str(_ACTOR_B),
            action="entry.created",
            target_id=shared_target,
            target_kind="entry",
            metadata={"test": "a"},
        )
    async with user_scoped_connection(app_pool, user_id=_ACTOR_C) as conn:
        await record_audit(
            conn,
            actor_type="user",
            actor_id=str(_ACTOR_C),
            action="topic.created",
            target_id=shared_target,
            target_kind="topic",
            metadata={"test": "b"},
        )

    async with admin_pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT target_kind, target_id, actor_id FROM audit_log "
            "WHERE target_id = $1 ORDER BY id",
            shared_target,
        )
    assert len(rows) == 2
    assert rows[0]["target_kind"] == "entry"
    assert rows[1]["target_kind"] == "topic"
