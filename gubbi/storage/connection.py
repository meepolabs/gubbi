"""Connection-acquisition helper that translates transient DB errors.

Use ``safe_acquire(pool)`` instead of ``pool.acquire()`` at request-path sites.
Use ``safe_user_scoped_connection(pool, user_id)`` instead of
``user_scoped_connection(pool, user_id)`` at request-path sites that need RLS.

Translates asynchronous connectivity errors to ``DatabaseUnavailable`` so the FastAPI
app boundary handler can map it to HTTP 503 instead of leaking a raw 500.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import UUID

import asyncpg
from gubbi_common.db.user_scoped import (
    DEFAULT_HNSW_EF_SEARCH,
    user_scoped_connection,
)

from gubbi.storage.exceptions import DatabaseUnavailable

# Transient errors that map to HTTP 503 (DatabaseUnavailable) at the API boundary.
# ``asyncio.TimeoutError`` covers pool-acquire budget exhaustion: gubbi-common
# 0.13.1 calls ``pool.acquire(timeout=5.0)``, and asyncpg raises
# ``asyncio.TimeoutError`` (not ``asyncpg.PostgresConnectionError``) when the
# acquire budget expires. Translating it here gives callers a clean
# ``DatabaseUnavailable`` instead of an unhandled ``TimeoutError``.
_TRANSIENT_ERRORS: tuple[type[Exception], ...] = (
    asyncpg.PostgresConnectionError,
    asyncpg.CannotConnectNowError,
    OSError,
    asyncio.TimeoutError,
)


@asynccontextmanager
async def safe_acquire(pool: asyncpg.Pool) -> AsyncIterator[asyncpg.Connection]:
    """Acquire a connection; translate transient connectivity errors."""
    try:
        async with pool.acquire() as conn:
            yield conn
    except _TRANSIENT_ERRORS as exc:
        raise DatabaseUnavailable(str(exc)) from exc


@asynccontextmanager
async def safe_user_scoped_connection(
    pool: asyncpg.Pool,
    user_id: UUID,
    *,
    hnsw_ef_search: int = DEFAULT_HNSW_EF_SEARCH,
) -> AsyncIterator[asyncpg.Connection]:
    """Acquire a user-scoped connection; translate transient connectivity errors.

    Wraps ``gubbi_common.db.user_scoped.user_scoped_connection`` so that
    transient errors during connection acquisition are translated to
    ``DatabaseUnavailable`` (HTTP 503) instead of leaking as raw 500s.
    """
    try:
        async with user_scoped_connection(pool, user_id, hnsw_ef_search=hnsw_ef_search) as conn:
            yield conn
    except _TRANSIENT_ERRORS as exc:
        raise DatabaseUnavailable(str(exc)) from exc
