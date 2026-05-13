"""Integration test for migration 0029 ``audit_log_target_kind_check``.

Asserts the CHECK constraint
``audit_log_target_kind_invariant`` (CHECK (target_id IS NULL OR
target_kind IS NOT NULL)) rejects a row with ``target_id`` set and
``target_kind`` NULL. The Python-level helper enforces the same
invariant; this test guards the DB layer so a writer that bypasses the
helper (raw SQL, BYPASSRLS admin scripts) still fails at the boundary.
"""

from __future__ import annotations

import asyncpg
import pytest

pytestmark = [
    pytest.mark.asyncio(loop_scope="session"),
    pytest.mark.integration,
]


_CONSTRAINT_NAME = "audit_log_target_kind_invariant"


async def test_constraint_exists_and_is_validated(admin_pool: asyncpg.Pool) -> None:
    """The CHECK constraint must be present AND validated (not NOT VALID)."""
    async with admin_pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT convalidated
            FROM pg_constraint
            WHERE conname = $1
              AND conrelid = 'audit_log'::regclass
            """,
            _CONSTRAINT_NAME,
        )
    assert row is not None, f"CHECK constraint {_CONSTRAINT_NAME} missing from audit_log"
    assert row["convalidated"] is True, (
        f"{_CONSTRAINT_NAME} exists but is NOT VALID -- migration 0029's "
        "VALIDATE CONSTRAINT step did not run"
    )


async def test_target_id_without_target_kind_rejected_at_db(admin_pool: asyncpg.Pool) -> None:
    """A direct INSERT with target_id but no target_kind MUST raise CheckViolation.

    The Python helper already rejects this shape via
    ``record_audit_async`` -- this test exercises the DB-level guard
    in case a future writer bypasses the helper.
    """
    async with admin_pool.acquire() as conn:
        with pytest.raises(asyncpg.CheckViolationError):
            await conn.execute(
                """
                INSERT INTO audit_log
                    (actor_type, actor_id, action, target_type, target_id, target_kind,
                     reason, metadata, ip_address, user_agent)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9::inet, $10)
                """,
                "system",
                "system:check-test",
                "secret.rotated",
                "secret",
                "00000000-0000-0000-0000-0000000000ff",
                None,  # target_kind deliberately NULL while target_id is non-NULL
                None,
                "{}",
                None,
                None,
            )


async def test_target_id_with_target_kind_accepted_at_db(admin_pool: asyncpg.Pool) -> None:
    """The happy path -- both set -- must still succeed."""
    async with admin_pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO audit_log
                (actor_type, actor_id, action, target_type, target_id, target_kind,
                 reason, metadata, ip_address, user_agent)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9::inet, $10)
            """,
            "system",
            "system:check-test",
            "secret.rotated",
            "secret",
            "00000000-0000-0000-0000-0000000000ff",
            "secret",
            None,
            "{}",
            None,
            None,
        )


async def test_both_null_accepted_at_db(admin_pool: asyncpg.Pool) -> None:
    """target_id NULL with target_kind NULL is allowed (system-level events)."""
    async with admin_pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO audit_log
                (actor_type, actor_id, action, target_type, target_id, target_kind,
                 reason, metadata, ip_address, user_agent)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9::inet, $10)
            """,
            "system",
            "system:check-test",
            "admin.query_executed",
            None,
            None,  # target_id NULL
            None,  # target_kind NULL -- allowed
            None,
            "{}",
            None,
            None,
        )
