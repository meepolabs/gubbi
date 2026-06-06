"""Schema-shape proofs for migration 0028 (audit_log cross-attribution guards).

Behavioural assertions for G1 (RLS WITH CHECK) and G2 (BEFORE INSERT trigger)
already live in ``tests/integration/test_audit_log_cross_attribution.py``.
This file complements that with metadata-shape proofs queried directly
from the system catalogs:

- ``pg_policies`` has ``audit_log_app_insert_self_only`` for
  ``cmd='INSERT'`` with the ``app.current_user_id`` GUC referenced in its
  WITH CHECK expression.
- ``pg_trigger`` has ``trg_audit_log_admin_no_user_actor`` as a
  before-insert trigger (``tgtype`` lower bits == ``2 | 4``) and is
  enabled (``tgenabled = 'O'``).
- The 0020 ``target_kind`` column also lives on ``audit_log``, proving
  both branches of the historical fork landed sequentially through the
  0028 chain.

A downgrade-safety test exercises the ``alembic downgrade -1`` path
end-to-end and re-runs ``alembic upgrade head`` so the session-scoped
test DB is left in the same shape it was found.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import asyncpg
import pytest

pytestmark = [
    pytest.mark.asyncio(loop_scope="session"),
    pytest.mark.integration,
]


_POLICY_NAME = "audit_log_app_insert_self_only"
_TRIGGER_NAME = "trg_audit_log_admin_no_user_actor"
# pg_trigger.tgtype low bits: 0x01=ROW, 0x02=BEFORE, 0x04=INSERT.
_BEFORE_INSERT_ROW_BITS = 0x01 | 0x02 | 0x04


# ---------------------------------------------------------------------------
# Schema-shape proofs (run on the session-scoped admin pool; no mutation)
# ---------------------------------------------------------------------------


async def test_app_insert_self_only_policy_present(
    admin_pool: asyncpg.Pool,
) -> None:
    """pg_policies row for the WITH CHECK INSERT policy lands as designed."""
    async with admin_pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT polname, polcmd, pg_get_expr(polqual, polrelid) AS using_expr,
                   pg_get_expr(polwithcheck, polrelid) AS check_expr
            FROM pg_policy
            WHERE polname = $1
              AND polrelid = 'audit_log'::regclass
            """,
            _POLICY_NAME,
        )
    assert row is not None, f"missing policy {_POLICY_NAME!r} on audit_log"
    # polcmd: 'a' = INSERT (Postgres internal char codes; asyncpg returns
    # the "char" type as a single-byte ``bytes`` value).
    assert row["polcmd"] == b"a", f"expected INSERT cmd, got: {row['polcmd']!r}"
    check_expr = row["check_expr"] or ""
    assert (
        "app.current_user_id" in check_expr
    ), f"expected app.current_user_id in WITH CHECK, got: {check_expr!r}"


async def test_admin_no_user_actor_trigger_present(
    admin_pool: asyncpg.Pool,
) -> None:
    """pg_trigger row exists, BEFORE INSERT, ROW level, enabled."""
    async with admin_pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT tgname, tgtype, tgenabled
            FROM pg_trigger
            WHERE tgname = $1
              AND tgrelid = 'audit_log'::regclass
              AND NOT tgisinternal
            """,
            _TRIGGER_NAME,
        )
    assert row is not None, f"missing trigger {_TRIGGER_NAME!r} on audit_log"
    # tgenabled 'O' = origin/local (i.e. enabled in normal replication mode).
    # asyncpg returns the "char" type as a single-byte ``bytes`` value.
    assert row["tgenabled"] == b"O", f"expected trigger enabled (b'O'), got: {row['tgenabled']!r}"
    # Mask off the high bits and assert BEFORE INSERT ROW.
    tg_low = int(row["tgtype"]) & 0x1F
    assert (
        tg_low == _BEFORE_INSERT_ROW_BITS
    ), f"expected BEFORE INSERT ROW (0x{_BEFORE_INSERT_ROW_BITS:02x}), got: 0x{tg_low:02x}"


async def test_audit_log_target_kind_column_present(
    admin_pool: asyncpg.Pool,
) -> None:
    """``target_kind`` from migration 0020 must coexist with 0028 guards.

    Proves the formerly-orphaned migration was reattached as 0028 *after*
    the live 0020 chain ran, rather than replacing it.
    """
    async with admin_pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT data_type
            FROM information_schema.columns
            WHERE table_name = 'audit_log' AND column_name = 'target_kind'
            """
        )
    assert row is not None, "missing audit_log.target_kind column from migration 0020"
    assert row["data_type"] == "text", f"expected TEXT type, got: {row['data_type']!r}"


# ---------------------------------------------------------------------------
# Downgrade safety (mutates schema; restores via re-upgrade)
# ---------------------------------------------------------------------------


def _run_alembic(bootstrap_dsn: str, *args: str) -> subprocess.CompletedProcess[str]:
    """Invoke alembic as a subprocess against ``bootstrap_dsn``.

    Mirrors ``tests/conftest.py::_run_alembic_upgrade`` so env vars are
    isolated to the migration run and global alembic state cannot leak
    into the test session.
    """
    project_root = Path(__file__).resolve().parents[2]
    env = {
        **os.environ,
        "JOURNAL_DB_MIGRATION_URL": bootstrap_dsn,
        "JOURNAL_OPERATOR_EMAIL": "operator@test.local",
    }
    return subprocess.run(  # noqa: S603 -- args are literals, no shell
        [sys.executable, "-m", "alembic", *args],
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.skip(
    reason="PERMANENT (by design, not a quarantine): asserts `alembic downgrade -1` "
    "removes just the cross-attribution policy + trigger, but the 0001 squashed "
    "baseline is the floor of the chain and has no per-step downgrade -- downgrade() "
    "raises NotImplementedError on purpose. A granular downgrade path will not be "
    "reintroduced, so this test stays skipped."
)
async def test_downgrade_removes_policy_and_trigger(
    admin_pool: asyncpg.Pool,
) -> None:
    """``alembic downgrade -1`` removes the policy + trigger and disables RLS.

    Restores via ``alembic upgrade head`` so the session-scoped DB is
    left at head for any subsequent tests in this run.
    """
    from tests.conftest import RLS_BOOTSTRAP_URL

    # Sanity: precondition is at-head.
    async with admin_pool.acquire() as conn:
        pre_policy = await conn.fetchval(
            "SELECT 1 FROM pg_policy WHERE polname = $1 AND polrelid = 'audit_log'::regclass",
            _POLICY_NAME,
        )
        pre_trigger = await conn.fetchval(
            """
            SELECT 1 FROM pg_trigger
            WHERE tgname = $1 AND tgrelid = 'audit_log'::regclass
              AND NOT tgisinternal
            """,
            _TRIGGER_NAME,
        )
        pre_rls = await conn.fetchval(
            "SELECT relrowsecurity FROM pg_class WHERE oid = 'audit_log'::regclass"
        )
    assert pre_policy == 1, "precondition failed: policy missing before downgrade"
    assert pre_trigger == 1, "precondition failed: trigger missing before downgrade"
    assert pre_rls is True, "precondition failed: RLS not enabled before downgrade"

    # Act: downgrade -1.
    down = _run_alembic(RLS_BOOTSTRAP_URL, "downgrade", "-1")
    if down.returncode != 0:
        pytest.fail(
            f"alembic downgrade -1 failed:\n" f"stdout:\n{down.stdout}\nstderr:\n{down.stderr}"
        )

    try:
        async with admin_pool.acquire() as conn:
            post_policy = await conn.fetchval(
                "SELECT 1 FROM pg_policy WHERE polname = $1 "
                "AND polrelid = 'audit_log'::regclass",
                _POLICY_NAME,
            )
            post_trigger = await conn.fetchval(
                """
                SELECT 1 FROM pg_trigger
                WHERE tgname = $1 AND tgrelid = 'audit_log'::regclass
                  AND NOT tgisinternal
                """,
                _TRIGGER_NAME,
            )
            post_rls = await conn.fetchval(
                "SELECT relrowsecurity FROM pg_class WHERE oid = 'audit_log'::regclass"
            )
        assert post_policy is None, "downgrade did not drop the WITH CHECK policy"
        assert post_trigger is None, "downgrade did not drop the BEFORE INSERT trigger"
        assert post_rls is False, "downgrade did not DISABLE ROW LEVEL SECURITY"
    finally:
        # Restore session DB to head no matter what (assertion failure or otherwise).
        up = _run_alembic(RLS_BOOTSTRAP_URL, "upgrade", "head")
        if up.returncode != 0:
            pytest.fail(
                f"alembic upgrade head (restore) failed -- session DB left "
                f"in pre-0028 state:\nstdout:\n{up.stdout}\nstderr:\n{up.stderr}"
            )
