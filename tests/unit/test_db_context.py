"""Tests for gubbi_common.db.user_scoped user_scoped_connection GUC wiring (TASK-02.06).

These tests verify the transaction-scoped contract of `user_scoped_connection`:
``app.current_user_id`` and ``hnsw.ef_search`` are set inside the yielded
transaction, and both are cleared when the transaction commits or rolls back —
so nothing leaks back into the pool when the connection is released.

The ``pool`` fixture lives in ``tests/conftest.py`` and skips the session if
PostgreSQL is not reachable at ``TEST_DATABASE_URL``.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator

import asyncpg
import pytest
from gubbi_common.db.user_scoped import MissingUserIdError, user_scoped_connection

from journalctl.core.auth_context import current_user_id

# The ``pool`` fixture is session-scoped. pytest-asyncio 0.25+ requires every
# test using session-scoped async fixtures to explicitly pin the loop scope,
# otherwise each test gets a fresh loop and the shared pool raises
# "cannot perform operation: another operation is in progress" at teardown.
pytestmark = pytest.mark.asyncio(loop_scope="session")


@pytest.fixture
def reset_user_ctxvar() -> Iterator[None]:
    """Snapshot the current_user_id ContextVar and restore it after the test."""
    token = current_user_id.set(None)
    try:
        yield
    finally:
        current_user_id.reset(token)


async def test_sets_user_id_inside_transaction(pool: asyncpg.Pool) -> None:
    """Both GUCs are bound inside the yielded connection."""
    user_a = uuid.uuid4()
    async with user_scoped_connection(pool, user_id=user_a) as conn:
        bound_user = await conn.fetchval("SELECT current_setting('app.current_user_id', true)")
        assert bound_user == str(user_a)
        bound_ef = await conn.fetchval("SELECT current_setting('hnsw.ef_search', true)")
        assert bound_ef == "100"


async def test_custom_hnsw_ef_search(pool: asyncpg.Pool) -> None:
    """hnsw_ef_search kwarg overrides the default."""
    user_a = uuid.uuid4()
    async with user_scoped_connection(pool, user_id=user_a, hnsw_ef_search=250) as conn:
        bound_ef = await conn.fetchval("SELECT current_setting('hnsw.ef_search', true)")
        assert bound_ef == "250"


async def test_cleared_after_commit(pool: asyncpg.Pool) -> None:
    """SET LOCAL does not leak — new transactions start clean after commit."""
    user_a = uuid.uuid4()
    async with user_scoped_connection(pool, user_id=user_a) as conn:
        assert await conn.fetchval("SELECT current_setting('app.current_user_id', true)") == str(
            user_a
        )

    async with pool.acquire() as conn:
        assert await conn.fetchval("SELECT current_setting('app.current_user_id', true)") == ""
        assert await conn.fetchval("SELECT current_setting('hnsw.ef_search', true)") == ""


async def test_cleared_after_rollback(pool: asyncpg.Pool) -> None:
    """An exception inside the scoped block rolls back the txn and clears GUCs too."""

    class _Boom(RuntimeError):
        pass

    user_a = uuid.uuid4()

    async def _run() -> None:
        async with user_scoped_connection(pool, user_id=user_a) as conn:
            bound = await conn.fetchval("SELECT current_setting('app.current_user_id', true)")
            assert bound == str(user_a)
            raise _Boom

    with pytest.raises(_Boom):
        await _run()

    async with pool.acquire() as conn:
        assert await conn.fetchval("SELECT current_setting('app.current_user_id', true)") == ""


async def test_explicit_user_id_from_contextvar(
    pool: asyncpg.Pool,
    reset_user_ctxvar: None,
) -> None:
    """Resolve user_id from ContextVar first, then pass it explicitly."""
    user_b = uuid.uuid4()
    current_user_id.set(user_b)
    user_id = current_user_id.get()
    assert user_id is not None
    async with user_scoped_connection(pool, user_id=user_id) as conn:
        bound_user = await conn.fetchval("SELECT current_setting('app.current_user_id', true)")
        assert bound_user == str(user_b)


async def test_raises_when_no_user(pool: asyncpg.Pool, reset_user_ctxvar: None) -> None:
    """MissingUserIdError when neither user_id arg nor ContextVar is set."""
    with pytest.raises(MissingUserIdError):
        async with user_scoped_connection(pool, user_id=None):  # type: ignore[arg-type]
            pytest.fail("should not reach body")


async def test_rejects_non_uuid(pool: asyncpg.Pool) -> None:
    """Passing a plain string for user_id raises TypeError at entry."""
    with pytest.raises(TypeError, match="user_id must be UUID"):
        async with user_scoped_connection(pool, user_id="not-a-uuid"):  # type: ignore[arg-type]
            pytest.fail("should not reach body")


async def test_nested_contextvar_does_not_leak_across_calls(
    pool: asyncpg.Pool,
    reset_user_ctxvar: None,
) -> None:
    """Two successive scoped connections for different users stay independent."""
    user_a = uuid.uuid4()
    user_b = uuid.uuid4()
    async with user_scoped_connection(pool, user_id=user_a) as conn:
        assert await conn.fetchval("SELECT current_setting('app.current_user_id', true)") == str(
            user_a
        )
    async with user_scoped_connection(pool, user_id=user_b) as conn:
        assert await conn.fetchval("SELECT current_setting('app.current_user_id', true)") == str(
            user_b
        )


async def test_concurrent_tasks_stay_isolated(pool: asyncpg.Pool) -> None:
    """Two concurrent scoped connections with different users do not interleave GUCs."""
    import asyncio as _asyncio

    user_a = uuid.uuid4()
    user_b = uuid.uuid4()

    async def _worker(user_id: uuid.UUID) -> str:
        async with user_scoped_connection(pool, user_id=user_id) as conn:
            # Hold the transaction open briefly to create overlap.
            await _asyncio.sleep(0.01)
            return await conn.fetchval("SELECT current_setting('app.current_user_id', true)")

    a_bound, b_bound = await _asyncio.gather(_worker(user_a), _worker(user_b))
    assert a_bound == str(user_a)
    assert b_bound == str(user_b)
