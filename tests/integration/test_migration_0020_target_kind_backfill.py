"""Integration test: migration 0020 backfills target_kind for pre-0020 rows.

Regression guard for the gap where pre-0020 audit rows have target_id set but
target_kind NULL.  Migration 0029 VALIDATE CONSTRAINT aborts with a
CheckViolation on any such row.  This test seeds a synthetic row at the 0019
schema level, upgrades through head, and asserts both that the upgrade
succeeds (no CheckViolation) and that the backfilled target_kind value is
correct.
"""

from __future__ import annotations

import uuid

import asyncpg
import pytest

from tests.integration._alembic_runner import run_alembic

pytestmark = [
    pytest.mark.asyncio(loop_scope="session"),
    pytest.mark.integration,
]


async def test_0020_backfills_target_kind_for_pre_migration_rows(
    admin_pool: asyncpg.Pool,
) -> None:
    """0019 seed row with target_id gets target_kind backfilled by 0020 upgrade.

    Steps wrapped in try/finally so the session DB is always restored to head
    regardless of assertion failures.
    """
    from tests.conftest import RLS_BOOTSTRAP_URL

    # Precondition: ensure DB is at head before mutating.
    up_pre = run_alembic(RLS_BOOTSTRAP_URL, "upgrade", "head")
    assert up_pre.returncode == 0, (
        f"precondition upgrade head failed:\n"
        f"stdout:\n{up_pre.stdout}\n"
        f"stderr:\n{up_pre.stderr}"
    )

    # Unique actor_id used to locate the row after upgrade.
    seed_actor_id = f"system:0020-backfill-test-{uuid.uuid4()}"
    seed_target_id = str(uuid.uuid4())

    try:
        # Downgrade to the revision before 0020 so target_kind column does not
        # exist yet and we can insert a synthetic pre-0020-style row.
        down = run_alembic(RLS_BOOTSTRAP_URL, "downgrade", "0019_rls_users")
        assert down.returncode == 0, (
            f"downgrade to 0019_rls_users failed:\n"
            f"stdout:\n{down.stdout}\n"
            f"stderr:\n{down.stderr}"
        )

        # Insert a synthetic pre-0020 row: target_id is set, target_kind column
        # does not exist at 0019 schema level, so it is omitted from the INSERT.
        # admin_pool authenticates as journal_admin (BYPASSRLS).  At 0019 the
        # only immutability triggers are BEFORE UPDATE and BEFORE DELETE, so a
        # plain INSERT requires no session_replication_role bypass.
        async with admin_pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO audit_log (actor_type, actor_id, action, target_type, target_id)
                VALUES ('system', $1, 'identity.created', 'user', $2)
                """,
                seed_actor_id,
                seed_target_id,
            )

        # Upgrade through head.  Migration 0020 must backfill the row; 0029
        # VALIDATE CONSTRAINT must not raise CheckViolation.
        up = run_alembic(RLS_BOOTSTRAP_URL, "upgrade", "head")
        assert up.returncode == 0, (
            f"upgrade head failed (expected 0020 backfill to prevent CheckViolation):\n"
            f"stdout:\n{up.stdout}\n"
            f"stderr:\n{up.stderr}"
        )

        # Verify the backfill: target_kind must equal target_type ('user').
        async with admin_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT target_kind FROM audit_log WHERE actor_id = $1",
                seed_actor_id,
            )
        assert (
            row is not None
        ), f"seeded audit row not found after upgrade (actor_id={seed_actor_id!r})"
        assert (
            row["target_kind"] == "user"
        ), f"expected target_kind='user' after backfill, got: {row['target_kind']!r}"

    finally:
        # Restore session DB to head unconditionally.
        up_restore = run_alembic(RLS_BOOTSTRAP_URL, "upgrade", "head")
        if up_restore.returncode != 0:
            pytest.fail(
                f"restore upgrade head failed -- session DB may be at 0019:\n"
                f"stdout:\n{up_restore.stdout}\n"
                f"stderr:\n{up_restore.stderr}"
            )
