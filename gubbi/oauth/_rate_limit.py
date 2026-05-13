"""Rate-limit event storage for OAuth login/register endpoints.

Extracted from `gubbi/oauth/storage.py` so the rate-limit concern can evolve
independently of the OAuth token/code/client persistence concerns. Both
classes share the same SQLite db file (WAL mode permits concurrent readers
and one writer); the `rate_limit_events` table DDL still lives in
`OAuthStorage._init_schema` so schema management stays centralized.

`RateLimitStorage` owns its own `asyncio.Lock` and aiosqlite connection -- it
does NOT share `OAuthStorage._lock` because the lock scope is per-connection.

Prerequisite: the `rate_limit_events` table is created by
`OAuthStorage._init_schema`. Callers MUST construct (and trigger schema
init on) an `OAuthStorage` against the same `db_path` before exercising
`RateLimitStorage` methods. To make the failure mode obvious instead of
surfacing a raw `sqlite3.OperationalError: no such table`, the first
method call lazily verifies the table exists and raises `RuntimeError`
with a clear remediation hint when it does not.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import aiosqlite

from gubbi.storage.constants import DB_BUSY_TIMEOUT_MS

__all__: list[str] = ["RateLimitStorage"]


class RateLimitStorage:
    """SQLite-backed counter for rate-limit events.

    Reads and writes the `rate_limit_events` table inside the OAuth db.
    Schema for that table is created by `OAuthStorage._init_schema`. On
    first method invocation this class verifies the table exists; if not,
    it raises a `RuntimeError` with a clear remediation hint instead of
    leaking a raw `sqlite3.OperationalError` to the caller.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._conn: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()
        self._schema_verified = False

    async def _get_conn(self) -> aiosqlite.Connection:
        """Return the lazily-initialized aiosqlite connection."""
        async with self._lock:
            if self._conn is None:
                self._conn = await aiosqlite.connect(str(self.db_path))
                self._conn.row_factory = aiosqlite.Row
                await self._conn.execute(f"PRAGMA busy_timeout={DB_BUSY_TIMEOUT_MS}")
            if not self._schema_verified:
                await self._verify_schema(self._conn)
                self._schema_verified = True
        return self._conn

    @staticmethod
    async def _verify_schema(conn: aiosqlite.Connection) -> None:
        """Raise RuntimeError if the `rate_limit_events` table is absent.

        The table is owned by `OAuthStorage._init_schema`; if it's missing,
        a caller has constructed `RateLimitStorage` standalone without first
        initializing `OAuthStorage` on the same db_path. Surface a clear
        remediation hint instead of a raw `sqlite3.OperationalError`.
        """
        cur = await conn.execute(
            "SELECT name FROM sqlite_master " "WHERE type = 'table' AND name = 'rate_limit_events'"
        )
        row = await cur.fetchone()
        if row is None:
            raise RuntimeError(
                "rate_limit_events table not initialised; construct "
                "OAuthStorage on the same db_path first"
            )

    def close(self) -> None:
        """Synchronously mark connection for closure.

        NOTE: aiosqlite connections are closed asynchronously; callers that
        hold a running event loop should use ``await storage.aclose()`` instead.
        This sync form is kept for teardown contexts (e.g. pytest fixtures)
        where the loop may no longer be running. It resets the internal state
        so the next ``_get_conn()`` call opens a fresh connection.

        Re-verify schema if the connection is later reopened against a
        potentially-different db file state.
        """
        # Reset the connection reference so next _get_conn re-opens.
        # The underlying aiosqlite worker thread will be reaped when the
        # Connection object is garbage-collected.
        self._conn = None
        self._schema_verified = False

    async def aclose(self) -> None:
        """Async close: flush and terminate the aiosqlite worker thread."""
        async with self._lock:
            if self._conn is not None:
                await self._conn.close()
                self._conn = None
            self._schema_verified = False

    async def record_event(self, event_key: str) -> None:
        """Record a single rate-limit event (e.g. 'login_failure:1.2.3.4')."""
        conn = await self._get_conn()
        async with self._lock:
            await conn.execute(
                "INSERT INTO rate_limit_events (event_key, occurred_at) VALUES (?, ?)",
                (event_key, int(time.time())),
            )
            await conn.commit()

    async def count_events(self, event_key: str, window_secs: int) -> int:
        """Count events for a key that occurred within the last window_secs seconds."""
        conn = await self._get_conn()
        async with self._lock:
            cutoff = int(time.time()) - window_secs
            cur = await conn.execute(
                "SELECT COUNT(*) AS c FROM rate_limit_events "
                "WHERE event_key = ? AND occurred_at >= ?",
                (event_key, cutoff),
            )
            row = await cur.fetchone()
        return int(row["c"]) if row else 0

    async def prune(self, retention_secs: int) -> int:
        """Delete events older than retention_secs. Returns rows deleted."""
        conn = await self._get_conn()
        async with self._lock:
            cutoff = int(time.time()) - retention_secs
            cur = await conn.execute(
                "DELETE FROM rate_limit_events WHERE occurred_at < ?",
                (cutoff,),
            )
            await conn.commit()
            return int(cur.rowcount)
