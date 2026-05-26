"""SQLite storage for OAuth 2.0 data.

Stores clients, authorization codes, access tokens, and refresh tokens.
Follows patterns from storage/index.py: WAL mode, busy_timeout for
multi-worker safety, lazy connection initialization.

This database is independent from the journal FTS5 index and can be
deleted/recreated without affecting journal data.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3  # Imported for exception type only; no sync sqlite3 usage.
import time
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import aiosqlite
from mcp.server.auth.provider import AccessToken, AuthorizationCode, RefreshToken
from mcp.shared.auth import OAuthClientInformationFull

from gubbi.oauth._rate_limit import RateLimitStorage
from gubbi.oauth.constants import RATE_LIMIT_EVENT_RETENTION_SECS
from gubbi.storage.constants import DB_BUSY_TIMEOUT_MS

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

__all__: list[str] = ["SCHEMA", "OAuthStorage"]

# structlog.AsyncBoundLogger emits return coroutines that must be awaited;
# this module's public API is fully async, but stdlib logging is kept for
# simplicity (no await needed in the backfill helper).
logger = logging.getLogger(__name__)

SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS clients (
    client_id   TEXT PRIMARY KEY,
    client_info TEXT NOT NULL,
    created_at  INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS auth_codes (
    code        TEXT PRIMARY KEY,
    data        TEXT NOT NULL,
    created_at  INTEGER NOT NULL,
    expires_at  INTEGER
);

CREATE TABLE IF NOT EXISTS access_tokens (
    token       TEXT PRIMARY KEY,
    data        TEXT NOT NULL,
    created_at  INTEGER NOT NULL,
    expires_at  INTEGER
);

CREATE TABLE IF NOT EXISTS refresh_tokens (
    token       TEXT PRIMARY KEY,
    data        TEXT NOT NULL,
    created_at  INTEGER NOT NULL,
    expires_at  INTEGER
);

CREATE TABLE IF NOT EXISTS token_pairs (
    access_token  TEXT NOT NULL,
    refresh_token TEXT NOT NULL,
    created_at    INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS rate_limit_events (
    event_key   TEXT NOT NULL,
    occurred_at INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_token_pairs_access
    ON token_pairs(access_token);
CREATE INDEX IF NOT EXISTS idx_token_pairs_refresh
    ON token_pairs(refresh_token);
CREATE INDEX IF NOT EXISTS idx_rate_limit_events_key_time
    ON rate_limit_events(event_key, occurred_at);
"""

# Columns added post-initial-schema. ALTER TABLE ADD COLUMN cannot be inside
# a transaction that also uses IF NOT EXISTS semantics, so run these out-of-band
# and swallow the "duplicate column" OperationalError for idempotency.
_ADD_COLUMN_MIGRATIONS: tuple[tuple[str, str], ...] = (
    ("auth_codes", "expires_at INTEGER"),
    ("access_tokens", "expires_at INTEGER"),
    ("refresh_tokens", "expires_at INTEGER"),
)

# Indexes that depend on columns added via _ADD_COLUMN_MIGRATIONS.
# Must be created AFTER those migrations run -- separated from SCHEMA to avoid
# "no such column: expires_at" errors on legacy databases.
_POST_MIGRATION_INDEXES: tuple[str, ...] = (
    "CREATE INDEX IF NOT EXISTS idx_auth_codes_expires_at ON auth_codes(expires_at)",
    "CREATE INDEX IF NOT EXISTS idx_access_tokens_expires_at ON access_tokens(expires_at)",
    "CREATE INDEX IF NOT EXISTS idx_refresh_tokens_expires_at ON refresh_tokens(expires_at)",
)


class OAuthStorage:
    """SQLite storage layer for OAuth 2.0 entities."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._conn: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()
        self._initialized: bool = False
        self._rl = RateLimitStorage(db_path)

    async def _get_conn(self) -> aiosqlite.Connection:
        """Return the lazily-initialized aiosqlite connection (schema applied on first access)."""
        async with self._lock:
            if self._conn is None:
                self._conn = await aiosqlite.connect(str(self.db_path))
                self._conn.row_factory = aiosqlite.Row
                await self._conn.execute(f"PRAGMA busy_timeout={DB_BUSY_TIMEOUT_MS}")
                if not self._initialized:
                    await self._init_schema()
                    self._initialized = True
            return self._conn

    async def initialize(self) -> None:
        """Lazily connect and apply schema."""
        await self._get_conn()

    @asynccontextmanager
    async def _atomic(self) -> AsyncIterator[aiosqlite.Connection]:
        """Yield a connection inside ``BEGIN IMMEDIATE`` -> commit/rollback.

        Non-reentrancy invariant: callers MUST NOT, while inside the yield,
        call back into any storage method that re-acquires ``self._lock``
        (notably ``_get_conn`` and ``close``). ``asyncio.Lock`` is not
        reentrant, so a same-task re-acquire would deadlock. ``close()``
        also takes this lock, so a self-call from inside ``_atomic`` would
        hang shutdown indefinitely. Today no caller violates this.
        """
        conn = await self._get_conn()
        async with self._lock:
            await conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
                await conn.commit()
            except BaseException:
                # NOTE: body exceptions raised inside _atomic are logged via
                # rollback_err.__context__ on the rollback-failure path below
                # (and propagated to the caller normally). Callers MUST NOT
                # raise exceptions whose __repr__ embeds token material or
                # other secrets -- aiosqlite/sqlite3 exceptions are safe today
                # (CPython strips parameter values), but custom exception
                # types added to this surface should respect the same rule.
                try:
                    await conn.rollback()
                except BaseException as rollback_err:
                    # rollback_err.__context__ is the original body exception
                    # (Python's exception-chaining sets it automatically when
                    # one exception is raised during handling of another).
                    # Capture it explicitly in the message text -- gubbi's
                    # production JSON logger does not walk __context__ chains
                    # by default, so the original auth-context (e.g. "invalid
                    # client_secret") would be lost from operator logs without
                    # this. exc_info=True still gives chain-walking formatters
                    # (stdlib, structlog with format_exc_info) the full trace.
                    original = rollback_err.__context__
                    logger.warning(
                        "OAuth storage rollback failed (%s); original body "
                        "exception: %r. SQLite will discard uncommitted state "
                        "on connection close.",
                        rollback_err,
                        original,
                        exc_info=True,
                    )
                raise

    async def _init_schema(self) -> None:
        assert self._conn is not None, "connection must be available when _init_schema runs"  # noqa: S101
        await self._conn.executescript(SCHEMA)
        await self._run_add_column_migrations()
        await self._run_post_migration_indexes()
        await self._backfill_expires_at()

    async def _run_add_column_migrations(self) -> None:
        assert self._conn is not None  # noqa: S101
        for table, column_def in _ADD_COLUMN_MIGRATIONS:
            try:
                await self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column_def}")
                await self._conn.commit()
            except sqlite3.OperationalError as e:
                msg = str(e)
                if not msg.startswith("duplicate column name:"):
                    raise

    async def _run_post_migration_indexes(self) -> None:
        """Create indexes that depend on columns added by _run_add_column_migrations."""
        assert self._conn is not None  # noqa: S101
        for stmt in _POST_MIGRATION_INDEXES:
            await self._conn.execute(stmt)
        await self._conn.commit()

    async def _backfill_expires_at(self) -> None:
        """One-time backfill of expires_at column from JSON blob data.

        Safe to run on every startup -- only updates rows where expires_at IS NULL.
        Limited to 10 000 rows per startup; excess rows are picked up
        on subsequent startups.
        """
        assert self._conn is not None  # noqa: S101
        for table, key_col in (
            ("auth_codes", "code"),
            ("access_tokens", "token"),
            ("refresh_tokens", "token"),
        ):
            cur = await self._conn.execute(
                f"SELECT {key_col}, data FROM {table} WHERE expires_at IS NULL LIMIT 10000"  # noqa: S608
            )
            rows = list(await cur.fetchall())
            if len(rows) >= 10000:
                logger.warning(
                    "oauth backfill: %s has >=10000 rows missing expires_at; "
                    "processing first 10000 this startup, remainder on next restart",
                    table,
                )
            for row in rows:
                try:
                    expires_at = json.loads(row["data"]).get("expires_at")
                except (json.JSONDecodeError, TypeError):
                    continue
                if expires_at is None:
                    continue
                await self._conn.execute(
                    f"UPDATE {table} SET expires_at = ? WHERE {key_col} = ?",  # noqa: S608
                    (int(float(expires_at)), row[key_col]),
                )
            await self._conn.commit()

    async def close(self) -> None:
        """Release the SQLite connection and the rate-limit storage handle.

        Acquires self._lock so any in-flight _atomic block can finish
        (commit or rollback) before the connection is torn down. Without
        the lock, a lifespan teardown racing an active transaction could
        close the connection out from under the open _atomic context.

        Non-reentrancy invariant: must NOT be called from inside an
        ``_atomic`` block on the same task (asyncio.Lock is not reentrant
        -- self-acquire would deadlock). Today every call site is the
        lifespan teardown handler, which never enters from inside a
        transaction.
        """
        async with self._lock:
            if self._conn:
                await self._conn.close()
                self._conn = None
            self._initialized = False
        if self._rl is not None:
            await self._rl.aclose()

    # ------------------------------------------------------------------
    # Clients
    # ------------------------------------------------------------------

    async def save_client(self, client_info: OAuthClientInformationFull) -> None:
        """Insert or replace an OAuth client row keyed by client_id."""
        conn = await self._get_conn()
        async with self._lock:
            await conn.execute(
                "INSERT OR REPLACE INTO clients (client_id, client_info, created_at) "
                "VALUES (?, ?, strftime('%s', 'now'))",
                (client_info.client_id, client_info.model_dump_json()),
            )
            await conn.commit()

    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        """Load a client row by client_id, or return None if not registered."""
        conn = await self._get_conn()
        async with self._lock:
            cur = await conn.execute(
                "SELECT client_info FROM clients WHERE client_id = ?",
                (client_id,),
            )
            row = await cur.fetchone()
        if row is None:
            return None
        return OAuthClientInformationFull.model_validate_json(row["client_info"])

    # ------------------------------------------------------------------
    # Authorization codes
    # ------------------------------------------------------------------

    async def save_auth_code(self, code: str, auth_code: AuthorizationCode) -> None:
        """Persist an authorization code with its expires_at for indexed cleanup."""
        conn = await self._get_conn()
        async with self._lock:
            await conn.execute(
                "INSERT OR REPLACE INTO auth_codes (code, data, created_at, expires_at) "
                "VALUES (?, ?, strftime('%s', 'now'), ?)",
                (
                    code,
                    auth_code.model_dump_json(),
                    int(float(auth_code.expires_at)) if auth_code.expires_at is not None else None,
                ),
            )
            await conn.commit()

    async def get_auth_code(self, code: str) -> AuthorizationCode | None:
        """Load an authorization code by code string, or return None if absent/expired."""
        conn = await self._get_conn()
        async with self._lock:
            cur = await conn.execute(
                "SELECT data FROM auth_codes WHERE code = ?",
                (code,),
            )
            row = await cur.fetchone()
        if row is None:
            return None
        return AuthorizationCode.model_validate_json(row["data"])

    async def delete_auth_code(self, code: str) -> None:
        """Single-use code: remove from storage after redemption (or on revocation)."""
        conn = await self._get_conn()
        async with self._lock:
            await conn.execute("DELETE FROM auth_codes WHERE code = ?", (code,))
            await conn.commit()

    # ------------------------------------------------------------------
    # Access tokens
    # ------------------------------------------------------------------

    async def save_access_token(self, token: str, access_token: AccessToken) -> None:
        """Persist an access token with its expires_at for indexed cleanup."""
        conn = await self._get_conn()
        async with self._lock:
            await conn.execute(
                "INSERT OR REPLACE INTO access_tokens (token, data, created_at, expires_at) "
                "VALUES (?, ?, strftime('%s', 'now'), ?)",
                (
                    token,
                    access_token.model_dump_json(),
                    int(access_token.expires_at) if access_token.expires_at is not None else None,
                ),
            )
            await conn.commit()

    async def get_access_token(self, token: str) -> AccessToken | None:
        """Load an access token by token string, or return None if absent."""
        conn = await self._get_conn()
        async with self._lock:
            cur = await conn.execute(
                "SELECT data FROM access_tokens WHERE token = ?",
                (token,),
            )
            row = await cur.fetchone()
        if row is None:
            return None
        return AccessToken.model_validate_json(row["data"])

    async def delete_access_token(self, token: str) -> None:
        """Revoke an access token by deleting its row."""
        conn = await self._get_conn()
        async with self._lock:
            await conn.execute("DELETE FROM access_tokens WHERE token = ?", (token,))
            await conn.commit()

    # ------------------------------------------------------------------
    # Refresh tokens
    # ------------------------------------------------------------------

    async def save_refresh_token(self, token: str, refresh_token: RefreshToken) -> None:
        """Persist a refresh token with its expires_at for indexed cleanup."""
        conn = await self._get_conn()
        async with self._lock:
            await conn.execute(
                "INSERT OR REPLACE INTO refresh_tokens (token, data, created_at, expires_at) "
                "VALUES (?, ?, strftime('%s', 'now'), ?)",
                (
                    token,
                    refresh_token.model_dump_json(),
                    int(refresh_token.expires_at) if refresh_token.expires_at is not None else None,
                ),
            )
            await conn.commit()

    async def get_refresh_token(self, token: str) -> RefreshToken | None:
        """Load a refresh token by token string, or return None if absent."""
        conn = await self._get_conn()
        async with self._lock:
            cur = await conn.execute(
                "SELECT data FROM refresh_tokens WHERE token = ?",
                (token,),
            )
            row = await cur.fetchone()
        if row is None:
            return None
        return RefreshToken.model_validate_json(row["data"])

    async def delete_refresh_token(self, token: str) -> None:
        """Revoke a refresh token by deleting its row (paired access tokens cleared separately)."""
        conn = await self._get_conn()
        async with self._lock:
            await conn.execute("DELETE FROM refresh_tokens WHERE token = ?", (token,))
            await conn.commit()

    # ------------------------------------------------------------------
    # Token pairs (access <-> refresh mapping for selective revocation)
    # ------------------------------------------------------------------

    async def save_token_pair(self, access_token: str, refresh_token: str) -> None:
        """Link an access token to its paired refresh token."""
        conn = await self._get_conn()
        async with self._lock:
            await conn.execute(
                "INSERT INTO token_pairs (access_token, refresh_token, created_at) "
                "VALUES (?, ?, ?)",
                (access_token, refresh_token, int(time.time())),
            )
            await conn.commit()

    async def save_issued_token_pair(
        self,
        access_token_str: str,
        access_token: AccessToken,
        refresh_token_str: str,
        refresh_token: RefreshToken,
    ) -> None:
        """Atomically persist access token, refresh token, and their pairing.

        All three inserts are wrapped in a single transaction so a crash
        between saves cannot leave partial state.
        """
        async with self._atomic() as conn:
            await conn.execute(
                "INSERT OR REPLACE INTO access_tokens (token, data, created_at, expires_at) "
                "VALUES (?, ?, strftime('%s', 'now'), ?)",
                (
                    access_token_str,
                    access_token.model_dump_json(),
                    int(access_token.expires_at) if access_token.expires_at is not None else None,
                ),
            )
            await conn.execute(
                "INSERT OR REPLACE INTO refresh_tokens (token, data, created_at, expires_at) "
                "VALUES (?, ?, strftime('%s', 'now'), ?)",
                (
                    refresh_token_str,
                    refresh_token.model_dump_json(),
                    int(refresh_token.expires_at) if refresh_token.expires_at is not None else None,
                ),
            )
            await conn.execute(
                "INSERT INTO token_pairs (access_token, refresh_token, created_at) "
                "VALUES (?, ?, ?)",
                (access_token_str, refresh_token_str, int(time.time())),
            )

    async def get_paired_refresh_token(self, access_token: str) -> str | None:
        """Get the refresh token paired with an access token."""
        conn = await self._get_conn()
        async with self._lock:
            cur = await conn.execute(
                "SELECT refresh_token FROM token_pairs WHERE access_token = ?",
                (access_token,),
            )
            row = await cur.fetchone()
        return row["refresh_token"] if row else None

    async def get_paired_access_tokens(self, refresh_token: str) -> list[str]:
        """Get all access tokens paired with a refresh token."""
        conn = await self._get_conn()
        async with self._lock:
            cur = await conn.execute(
                "SELECT access_token FROM token_pairs WHERE refresh_token = ?",
                (refresh_token,),
            )
            rows = await cur.fetchall()
        return [row["access_token"] for row in rows]

    async def delete_token_pair_by_access(self, access_token: str) -> None:
        """Drop the (access, refresh) pairing rows referencing this access token."""
        conn = await self._get_conn()
        async with self._lock:
            await conn.execute(
                "DELETE FROM token_pairs WHERE access_token = ?",
                (access_token,),
            )
            await conn.commit()

    async def delete_token_pair_by_refresh(self, refresh_token: str) -> None:
        """Drop the (access, refresh) pairing rows referencing this refresh token."""
        conn = await self._get_conn()
        async with self._lock:
            await conn.execute(
                "DELETE FROM token_pairs WHERE refresh_token = ?",
                (refresh_token,),
            )
            await conn.commit()

    # ------------------------------------------------------------------
    # Atomic refresh token rotation
    # ------------------------------------------------------------------

    async def rotate_refresh_token(
        self,
        old_refresh_token_str: str,
        new_access_token_str: str,
        new_access_token: AccessToken,
        new_refresh_token_str: str,
        new_refresh_token: RefreshToken,
    ) -> None:
        """Atomically revoke old refresh + its paired access tokens and issue new pair.

        Wraps all operations in a single SQLite transaction + asyncio.Lock so
        a crash or coroutine interleaving cannot leave partial state (closes the
        refresh rotation atomicity gap).
        """
        async with self._atomic() as conn:
            # 1. Collect access tokens paired with old refresh (so we can revoke them)
            cur = await conn.execute(
                "SELECT access_token FROM token_pairs WHERE refresh_token = ?",
                (old_refresh_token_str,),
            )
            paired_rows = await cur.fetchall()
            paired_access = [r["access_token"] for r in paired_rows]

            # 2. Delete old access tokens
            for at in paired_access:
                await conn.execute("DELETE FROM access_tokens WHERE token = ?", (at,))

            # 3. Delete old pair rows
            await conn.execute(
                "DELETE FROM token_pairs WHERE refresh_token = ?",
                (old_refresh_token_str,),
            )

            # 4. Delete old refresh token
            await conn.execute(
                "DELETE FROM refresh_tokens WHERE token = ?",
                (old_refresh_token_str,),
            )

            # 5. Insert new access token with indexed expires_at
            await conn.execute(
                "INSERT OR REPLACE INTO access_tokens "
                "(token, data, created_at, expires_at) "
                "VALUES (?, ?, strftime('%s','now'), ?)",
                (
                    new_access_token_str,
                    new_access_token.model_dump_json(),
                    int(new_access_token.expires_at)
                    if new_access_token.expires_at is not None
                    else None,
                ),
            )

            # 6. Insert new refresh token with indexed expires_at
            await conn.execute(
                "INSERT OR REPLACE INTO refresh_tokens "
                "(token, data, created_at, expires_at) "
                "VALUES (?, ?, strftime('%s','now'), ?)",
                (
                    new_refresh_token_str,
                    new_refresh_token.model_dump_json(),
                    int(new_refresh_token.expires_at)
                    if new_refresh_token.expires_at is not None
                    else None,
                ),
            )

            # 7. Insert new pair row
            await conn.execute(
                "INSERT INTO token_pairs (access_token, refresh_token, created_at) "
                "VALUES (?, ?, ?)",
                (new_access_token_str, new_refresh_token_str, int(time.time())),
            )

    # ------------------------------------------------------------------
    # Rate limit events (login failures, register attempts, etc.)
    # ------------------------------------------------------------------
    # Delegated to RateLimitStorage. Schema for the rate_limit_events table
    # is still owned by _init_schema above so all DDL stays centralized.

    async def record_rate_limit_event(self, event_key: str) -> None:
        """Record a single rate-limit event (e.g. 'login_failure:1.2.3.4')."""
        await self._rl.record_event(event_key)

    async def count_rate_limit_events(self, event_key: str, window_secs: int) -> int:
        """Count events for a key that occurred within the last window_secs seconds."""
        return await self._rl.count_events(event_key, window_secs)

    async def prune_rate_limit_events(self, retention_secs: int) -> int:
        """Delete events older than retention_secs. Returns rows deleted."""
        return await self._rl.prune(retention_secs)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def cleanup_expired(self) -> int:
        """Delete expired tokens/codes and prune stale rate-limit events.

        Uses indexed expires_at for O(log n) per-table deletion.
        Also cascades: access tokens paired with an expired refresh token are
        removed before the refresh token itself. Returns total rows deleted.
        """
        conn = await self._get_conn()
        async with self._lock:
            now = int(time.time())
            deleted = 0

            # 1. Expired auth codes -- expires_at is NOT NULL for new rows; legacy NULLs
            #    are treated as expired (same as old default_expired=True behavior).
            cur = await conn.execute(
                "DELETE FROM auth_codes " "WHERE expires_at IS NULL OR expires_at < ?",
                (now,),
            )
            deleted += cur.rowcount

            # 2. Expired access tokens (+ their pair rows)
            await conn.execute(
                "DELETE FROM token_pairs "
                "WHERE access_token IN (SELECT token FROM access_tokens "
                "                       WHERE expires_at IS NOT NULL AND expires_at < ?)",
                (now,),
            )
            cur = await conn.execute(
                "DELETE FROM access_tokens " "WHERE expires_at IS NOT NULL AND expires_at < ?",
                (now,),
            )
            deleted += cur.rowcount

            # 3. Cascade: access tokens paired with an expired refresh token
            cur = await conn.execute(
                "DELETE FROM access_tokens "
                "WHERE token IN ("
                "    SELECT access_token FROM token_pairs "
                "    WHERE refresh_token IN ("
                "        SELECT token FROM refresh_tokens "
                "        WHERE expires_at IS NOT NULL AND expires_at < ?"
                "    )"
                ")",
                (now,),
            )
            deleted += cur.rowcount

            # 4. Expired refresh tokens (+ their pair rows)
            await conn.execute(
                "DELETE FROM token_pairs "
                "WHERE refresh_token IN (SELECT token FROM refresh_tokens "
                "                        WHERE expires_at IS NOT NULL AND expires_at < ?)",
                (now,),
            )
            cur = await conn.execute(
                "DELETE FROM refresh_tokens " "WHERE expires_at IS NOT NULL AND expires_at < ?",
                (now,),
            )
            deleted += cur.rowcount

            await conn.commit()

        # 5. Prune stale rate-limit events (own lock inside)
        deleted += await self.prune_rate_limit_events(RATE_LIMIT_EVENT_RETENTION_SECS)
        return deleted
