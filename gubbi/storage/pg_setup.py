"""PostgreSQL pool initialisation and advisory-lock helpers."""

import asyncpg

from gubbi.constants import (
    APP_POOL_SIZE_MAX,
    APP_POOL_SIZE_MIN,
    DB_COMMAND_TIMEOUT_SECS,
)

__all__: list[str] = [
    "advisory_unlock",
    "init_pool",
    "try_advisory_lock",
]


async def _init_connection(conn: asyncpg.Connection) -> None:
    """Register the pgvector codec on every new pool connection.

    Called by asyncpg for each connection it creates. Must be registered
    here (via `init=`) rather than on a single acquired connection --
    `register_vector(conn)` only applies to that one connection object,
    not to others in the pool.
    """
    from pgvector.asyncpg import register_vector

    await register_vector(conn)


async def init_pool(
    database_url: str,
    min_size: int = APP_POOL_SIZE_MIN,
    max_size: int = APP_POOL_SIZE_MAX,
) -> asyncpg.Pool:
    """Create an asyncpg connection pool and register the pgvector codec.

    statement_cache_size=0 keeps the pool compatible with pgbouncer in
    transaction-pooling mode (prepared statements are connection-scoped
    and break when connections are re-assigned between requests).
    """
    pool: asyncpg.Pool = await asyncpg.create_pool(
        database_url,
        min_size=min_size,
        max_size=max_size,
        command_timeout=DB_COMMAND_TIMEOUT_SECS,
        statement_cache_size=0,
        init=_init_connection,
        server_settings={"application_name": "gubbi"},
    )
    return pool


async def try_advisory_lock(conn: asyncpg.Connection, key: int) -> bool:
    """Attempt to acquire a session-level advisory lock. Returns True if acquired."""
    return bool(await conn.fetchval("SELECT pg_try_advisory_lock($1)", key))


async def advisory_unlock(conn: asyncpg.Connection, key: int) -> None:
    """Release a session-level advisory lock."""
    await conn.execute("SELECT pg_advisory_unlock($1)", key)
