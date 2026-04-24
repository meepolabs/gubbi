"""Integration test for audit_log trigger immutability (TASK-02.20).

Verifies that the trg_audit_log_no_update and trg_audit_log_no_delete
triggers raise 'audit_log rows are append-only' for any mutation attempt,
even when executed via journal_admin (BYPASSRLS).

Requires a running PostgreSQL instance with migrations applied through
0010_audit_log. Use the RLS test database provisioned by _rls_provisioned.

Run with:
    pytest tests/integration/test_audit_immutability.py -v
"""

from __future__ import annotations

import asyncpg
import pytest

from journalctl.audit import Action, record_audit

pytestmark = [
    pytest.mark.asyncio(loop_scope="session"),
    pytest.mark.integration,
]

_APPEND_ONLY_MSG = "append-only"


async def _insert_test_row(conn: asyncpg.Connection) -> int:
    """Insert a minimal audit row and return its id."""
    row_id: int = await conn.fetchval(
        """
        INSERT INTO audit_log (actor_type, actor_id, action)
        VALUES ('system', 'test-worker', $1)
        RETURNING id
        """,
        Action.ADMIN_QUERY_EXECUTED,
    )
    return row_id


# ---------------------------------------------------------------------------
# Trigger blocks UPDATE
# ---------------------------------------------------------------------------


async def test_update_raises_append_only(admin_pool: asyncpg.Pool) -> None:
    """UPDATE on any audit_log row must raise via trigger."""
    async with admin_pool.acquire() as conn:
        row_id = await _insert_test_row(conn)

        with pytest.raises(asyncpg.RaiseError, match=_APPEND_ONLY_MSG):
            await conn.execute(
                "UPDATE audit_log SET reason = 'tampered' WHERE id = $1",
                row_id,
            )


# ---------------------------------------------------------------------------
# Trigger blocks DELETE
# ---------------------------------------------------------------------------


async def test_delete_raises_append_only(admin_pool: asyncpg.Pool) -> None:
    """DELETE on any audit_log row must raise via trigger."""
    async with admin_pool.acquire() as conn:
        row_id = await _insert_test_row(conn)

        with pytest.raises(asyncpg.RaiseError, match=_APPEND_ONLY_MSG):
            await conn.execute(
                "DELETE FROM audit_log WHERE id = $1",
                row_id,
            )


# ---------------------------------------------------------------------------
# record_audit() helper inserts correctly via admin connection
# ---------------------------------------------------------------------------


async def test_record_audit_inserts_row(admin_pool: asyncpg.Pool) -> None:
    """record_audit() produces a readable row via journal_admin SELECT."""
    async with admin_pool.acquire() as conn:
        before_count: int = await conn.fetchval("SELECT count(*) FROM audit_log")

        await record_audit(
            conn,
            actor_type="admin",
            actor_id="test-admin",
            action=Action.SECRET_ROTATED,
            target_type="secret",
            target_id="encryption_master_key",
            reason="scheduled rotation",
            metadata={"version": "v2"},
        )

        after_count: int = await conn.fetchval("SELECT count(*) FROM audit_log")
        assert after_count == before_count + 1

        row = await conn.fetchrow(
            "SELECT * FROM audit_log WHERE actor_id = 'test-admin' ORDER BY id DESC LIMIT 1"
        )
        assert row is not None
        assert row["actor_type"] == "admin"
        assert row["action"] == Action.SECRET_ROTATED
        assert row["target_type"] == "secret"
        assert row["target_id"] == "encryption_master_key"
        assert row["reason"] == "scheduled rotation"


# ---------------------------------------------------------------------------
# Trigger fires even for journal_admin (BYPASSRLS does not bypass triggers)
# ---------------------------------------------------------------------------


async def test_trigger_fires_for_journal_admin(admin_pool: asyncpg.Pool) -> None:
    """BYPASSRLS does not bypass triggers -- admin_pool must still be blocked."""
    async with admin_pool.acquire() as conn:
        row_id = await _insert_test_row(conn)

        # UPDATE attempt
        with pytest.raises(asyncpg.RaiseError, match=_APPEND_ONLY_MSG):
            await conn.execute(
                "UPDATE audit_log SET actor_id = 'modified' WHERE id = $1",
                row_id,
            )

        # DELETE attempt
        with pytest.raises(asyncpg.RaiseError, match=_APPEND_ONLY_MSG):
            await conn.execute(
                "DELETE FROM audit_log WHERE id = $1",
                row_id,
            )


# ---------------------------------------------------------------------------
# record_audit() then update raises
# ---------------------------------------------------------------------------


async def test_record_audit_then_update_raises(admin_pool: asyncpg.Pool) -> None:
    """record_audit() inserts a row, then an UPDATE on that row must raise."""
    async with admin_pool.acquire() as conn:
        row_before: int = await conn.fetchval("SELECT count(*) FROM audit_log")

        await record_audit(
            conn,
            actor_type="system",
            actor_id="test-fix-worker",
            action=Action.ADMIN_QUERY_EXECUTED,
        )

        row_id: int = await conn.fetchval("SELECT max(id) FROM audit_log")

        row_after: int = await conn.fetchval("SELECT count(*) FROM audit_log")
        assert row_after == row_before + 1

        with pytest.raises(asyncpg.RaiseError, match=_APPEND_ONLY_MSG):
            await conn.execute(
                "UPDATE audit_log SET reason = 'tampered' WHERE id = $1",
                row_id,
            )
