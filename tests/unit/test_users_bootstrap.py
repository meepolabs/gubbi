"""Unit tests for journalctl.users.bootstrap -- function-level only."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

pytestmark = [pytest.mark.asyncio(loop_scope="function")]

_FAKE_UUID = "550e8400-e29b-41d4-a716-446655440000"
_EMAIL = "op@test.local"
_TIMEZONE = "UTC"


def _make_pool(
    execute_ret: str | None = "INSERT 0 1", fetchval_ret: str | None = _FAKE_UUID
) -> MagicMock:
    """Return a mock asyncpg.Pool with pre-configured acquire()."""
    conn = MagicMock()
    conn.execute = AsyncMock(return_value=execute_ret)
    conn.fetchval = AsyncMock(return_value=fetchval_ret)

    context_mgr = MagicMock()
    context_mgr.__aenter__ = AsyncMock(return_value=conn)
    context_mgr.__aexit__ = AsyncMock(return_value=False)

    pool = MagicMock()
    pool.acquire = MagicMock(return_value=context_mgr)
    return pool


async def test_scaffold_insert_when_absent() -> None:
    """When no user row exists, INSERT creates one and returns."""
    pool = _make_pool(execute_ret="INSERT 0 1", fetchval_ret=_FAKE_UUID)
    from journalctl.users.bootstrap import scaffold_operator

    await scaffold_operator(pool, _EMAIL, _TIMEZONE)
    pool.acquire.assert_called_once()


async def test_scaffold_noop_when_present() -> None:
    """When INSERT is a no-op (conflict), verify row still exists."""
    pool = _make_pool(execute_ret="INSERT 0 0", fetchval_ret=_FAKE_UUID)
    from journalctl.users.bootstrap import scaffold_operator

    await scaffold_operator(pool, _EMAIL, _TIMEZONE)
    pool.acquire.assert_called_once()


async def test_scaffold_error_on_no_row_after_insert() -> None:
    """When SELECT finds no row after INSERT, RuntimeError is raised."""
    pool = _make_pool(execute_ret="INSERT 0 1", fetchval_ret=None)
    from journalctl.users.bootstrap import scaffold_operator

    with pytest.raises(RuntimeError, match="No active user row found after provisioning"):
        await scaffold_operator(pool, _EMAIL, _TIMEZONE)


async def test_scaffold_postgres_error_wrapped() -> None:
    """Postgres errors are wrapped in RuntimeError with original cause."""
    import asyncpg

    pool = MagicMock()
    context_mgr = AsyncMock()
    conn = MagicMock()
    # The INSERT path uses fetchval (with RETURNING) to detect insert vs noop.
    conn.fetchval = AsyncMock(side_effect=asyncpg.PostgresError("connection refused"))
    context_mgr.__aenter__ = AsyncMock(return_value=conn)
    context_mgr.__aexit__ = AsyncMock(return_value=False)
    pool.acquire = MagicMock(return_value=context_mgr)

    from journalctl.users.bootstrap import scaffold_operator

    with pytest.raises(RuntimeError, match="PostgreSQL error during scaffold: connection refused"):
        await scaffold_operator(pool, _EMAIL, _TIMEZONE)
