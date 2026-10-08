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
"""

from __future__ import annotations

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
    assert "app.current_user_id" in check_expr, (
        f"expected app.current_user_id in WITH CHECK, got: {check_expr!r}"
    )


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
    assert tg_low == _BEFORE_INSERT_ROW_BITS, (
        f"expected BEFORE INSERT ROW (0x{_BEFORE_INSERT_ROW_BITS:02x}), got: 0x{tg_low:02x}"
    )


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
