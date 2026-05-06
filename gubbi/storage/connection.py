"""Connection-acquisition helper that translates transient DB errors.

Use ``safe_acquire(pool)`` instead of ``pool.acquire()`` at request-path sites.
Use ``safe_user_scoped_connection(pool, user_id)`` instead of
``user_scoped_connection(pool, user_id)`` at request-path sites that need RLS.

Translates asynchronous connectivity errors to ``DatabaseUnavailable`` so the FastAPI
app boundary handler can map it to HTTP 503 instead of leaking a raw 500.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import UUID

import asyncpg
from gubbi_common.db.user_scoped import (
    DEFAULT_HNSW_EF_SEARCH,
    user_scoped_connection,
)

from gubbi.storage.exceptions import DatabaseUnavailable

_TRANSIENT_ERRORS = (
    asyncpg.PostgresConnectionError,
    asyncpg.CannotConnectNowError,
    OSError,
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
