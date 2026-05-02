"""Integration tests for migration 0019_rls_users -- RLS on users table.

Each test is self-contained: inserts its own users via admin_pool (BYPASSRLS)
and asserts RLS behaviour via app_pool. Cleanup is done in try/finally.
"""

from __future__ import annotations

from uuid import uuid4

import asyncpg
import pytest

pytestmark = pytest.mark.asyncio(loop_scope="session")


@pytest.mark.integration
async def test_users_rls_cross_user_isolation(
    admin_pool: asyncpg.Pool,
    app_pool: asyncpg.Pool,
) -> None:
    """Users A and B are visible only when GUC is set to their respective id.

    ``set_config(..., true)`` is transaction-scoped, so each set + fetch
    pair is wrapped in ``conn.transaction()``.
    """
    user_a = uuid4()
    user_b = uuid4()
    try:
        async with admin_pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO users (id, email, timezone, created_at, updated_at) "
                "VALUES ($1, 'user-a@test.local', 'UTC', now(), now())",
                user_a,
            )
            await conn.execute(
                "INSERT INTO users (id, email, timezone, created_at, updated_at) "
                "VALUES ($1, 'user-b@test.local', 'UTC', now(), now())",
                user_b,
            )

        async with app_pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "SELECT set_config('app.current_user_id', $1, true)", str(user_a)
                )
                rows = await conn.fetch("SELECT id FROM users WHERE deleted_at IS NULL")
            assert len(rows) == 1
            assert rows[0]["id"] == user_a

        async with app_pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "SELECT set_config('app.current_user_id', $1, true)", str(user_b)
                )
                rows = await conn.fetch("SELECT id FROM users WHERE deleted_at IS NULL")
            assert len(rows) == 1
            assert rows[0]["id"] == user_b
    finally:
        async with admin_pool.acquire() as conn:
            await conn.execute("DELETE FROM users WHERE id = ANY($1::uuid[])", [user_a, user_b])


@pytest.mark.integration
async def test_users_rls_wrong_guc_returns_zero_rows(
    admin_pool: asyncpg.Pool,
    app_pool: asyncpg.Pool,
) -> None:
    """Setting GUC to a UUID that matches neither user returns zero rows."""
    user_a = uuid4()
    user_b = uuid4()
    wrong_id = uuid4()
    try:
        async with admin_pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO users (id, email, timezone, created_at, updated_at) "
                "VALUES ($1, 'user-a@test.local', 'UTC', now(), now())",
                user_a,
            )
            await conn.execute(
                "INSERT INTO users (id, email, timezone, created_at, updated_at) "
                "VALUES ($1, 'user-b@test.local', 'UTC', now(), now())",
                user_b,
            )

        async with app_pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "SELECT set_config('app.current_user_id', $1, true)", str(wrong_id)
                )
                rows = await conn.fetch("SELECT id FROM users WHERE deleted_at IS NULL")
            assert len(rows) == 0
    finally:
        async with admin_pool.acquire() as conn:
            await conn.execute("DELETE FROM users WHERE id = ANY($1::uuid[])", [user_a, user_b])


@pytest.mark.integration
async def test_users_rls_null_guc_returns_zero_rows(
    admin_pool: asyncpg.Pool,
    app_pool: asyncpg.Pool,
) -> None:
    """Bare app_pool connection (no GUC set) sees zero rows (default-deny)."""
    user_a = uuid4()
    try:
        async with admin_pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO users (id, email, timezone, created_at, updated_at) "
                "VALUES ($1, 'user-a@test.local', 'UTC', now(), now())",
                user_a,
            )

        async with app_pool.acquire() as conn:
            rows = await conn.fetch("SELECT id FROM users WHERE deleted_at IS NULL")
            assert len(rows) == 0
    finally:
        async with admin_pool.acquire() as conn:
            await conn.execute("DELETE FROM users WHERE id = $1", user_a)


@pytest.mark.integration
async def test_users_rls_soft_deleted_user_invisible(
    admin_pool: asyncpg.Pool,
    app_pool: asyncpg.Pool,
) -> None:
    """A soft-deleted user is invisible even when GUC is set to their id.

    ``set_config(..., true)`` is transaction-scoped, so each set + fetch
    pair is wrapped in ``conn.transaction()``.
    """
    user_a = uuid4()
    try:
        async with admin_pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO users (id, email, timezone, created_at, updated_at) "
                "VALUES ($1, 'user-a@test.local', 'UTC', now(), now())",
                user_a,
            )

        async with app_pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "SELECT set_config('app.current_user_id', $1, true)", str(user_a)
                )
                rows = await conn.fetch("SELECT id FROM users WHERE deleted_at IS NULL")
            assert len(rows) == 1
            assert rows[0]["id"] == user_a

        async with admin_pool.acquire() as conn:
            await conn.execute("UPDATE users SET deleted_at = now() WHERE id = $1", user_a)

        async with app_pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "SELECT set_config('app.current_user_id', $1, true)", str(user_a)
                )
                rows = await conn.fetch("SELECT id FROM users WHERE deleted_at IS NULL")
            assert len(rows) == 0
    finally:
        async with admin_pool.acquire() as conn:
            await conn.execute("DELETE FROM users WHERE id = $1", user_a)
