"""Integration tests for migration 0020 -- cross-attribution guards on audit_log.

G1: RLS WITH CHECK policy ensures app_pool INSERTs have actor_id matching
    the session GUC (app.current_user_id).
G2: BEFORE INSERT trigger prevents admin_pool from inserting actor_type='user'.
"""

from __future__ import annotations

import re
from uuid import UUID

import asyncpg
import pytest
from gubbi_common.db.user_scoped import user_scoped_connection

pytestmark = [
    pytest.mark.asyncio(loop_scope="session"),
    pytest.mark.integration,
]

_RLS_ERROR_RE = re.compile(
    r"new row violates row-level security policy|row-level security|violates.*policy",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# G1: RLS WITH CHECK on journal_app INSERTs
# ---------------------------------------------------------------------------


async def test_app_pool_matching_actor_id_succeeds(
    app_pool: asyncpg.Pool,
    tenant_a: UUID,
) -> None:
    """App pool with GUC set to tenant_a can INSERT an audit_log row for tenant_a."""
    async with user_scoped_connection(app_pool, user_id=tenant_a) as conn:
        row_id = await conn.fetchval(
            """
            INSERT INTO audit_log (actor_type, actor_id, action)
            VALUES ('user', $1, 'test')
            RETURNING id
            """,
            str(tenant_a),
        )
    assert row_id is not None
    assert row_id > 0


async def test_app_pool_mismatching_actor_id_blocked(
    app_pool: asyncpg.Pool,
    tenant_a: UUID,
    tenant_b: UUID,
) -> None:
    """App pool with GUC set to tenant_a cannot INSERT a row with tenant_b's actor_id."""
    async with user_scoped_connection(app_pool, user_id=tenant_a) as conn:
        with pytest.raises(asyncpg.PostgresError, match=_RLS_ERROR_RE):
            await conn.execute(
                """
                INSERT INTO audit_log (actor_type, actor_id, action)
                VALUES ('user', $1, 'test')
                """,
                str(tenant_b),
            )


async def test_app_pool_empty_actor_id_blocked(
    app_pool: asyncpg.Pool,
) -> None:
    """App pool with no GUC (empty session) cannot INSERT with empty actor_id."""
    async with app_pool.acquire() as conn:
        with pytest.raises(asyncpg.PostgresError, match=_RLS_ERROR_RE):
            await conn.execute(
                """
                INSERT INTO audit_log (actor_type, actor_id, action)
                VALUES ('user', '', 'test')
                """
            )


# ---------------------------------------------------------------------------
# G2: BEFORE INSERT trigger on journal_admin
# ---------------------------------------------------------------------------


async def test_admin_pool_system_actor_succeeds(
    admin_pool: asyncpg.Pool,
) -> None:
    """Admin pool can INSERT with actor_type='system'."""
    async with admin_pool.acquire() as conn:
        row_id = await conn.fetchval(
            """
            INSERT INTO audit_log (actor_type, actor_id, action)
            VALUES ('system', 'test-cron', 'test')
            RETURNING id
            """,
        )
    assert row_id is not None
    assert row_id > 0


@pytest.mark.parametrize("actor_type", ["admin", "founder"])
async def test_admin_pool_non_user_actor_succeeds(
    admin_pool: asyncpg.Pool,
    actor_type: str,
) -> None:
    """Admin pool can INSERT with actor_type='admin' or 'founder'."""
    async with admin_pool.acquire() as conn:
        row_id = await conn.fetchval(
            "INSERT INTO audit_log (actor_type, actor_id, action) "
            "VALUES ($1, 'test-worker', 'test') RETURNING id",
            actor_type,
        )
    assert row_id is not None
    assert row_id > 0


async def test_admin_pool_user_actor_blocked(
    admin_pool: asyncpg.Pool,
    tenant_a: UUID,
) -> None:
    """Admin pool cannot INSERT with actor_type='user' -- trigger raises."""
    async with admin_pool.acquire() as conn:
        with pytest.raises(
            asyncpg.PostgresError,
            match="journal_admin cannot insert audit_log row with actor_type=user",
        ):
            await conn.execute(
                """
                INSERT INTO audit_log (actor_type, actor_id, action)
                VALUES ('user', $1, 'test')
                """,
                str(tenant_a),
            )
