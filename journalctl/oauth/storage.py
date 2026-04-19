"""SQLite storage for OAuth 2.0 data.

Stores clients, authorization codes, access tokens, and refresh tokens.
Follows patterns from storage/index.py: WAL mode, busy_timeout for
multi-worker safety, lazy connection initialization.

This database is independent from the journal FTS5 index and can be
deleted/recreated without affecting journal data.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

from mcp.server.auth.provider import AccessToken, AuthorizationCode, RefreshToken
from mcp.shared.auth import OAuthClientInformationFull

from journalctl.oauth.constants import RATE_LIMIT_EVENT_RETENTION_SECS
from journalctl.storage.constants import DB_BUSY_TIMEOUT_MS

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
# Must be created AFTER those migrations run — separated from SCHEMA to avoid
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
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
            )
            self._conn.row_factory = sqlite3.Row
            self._conn.execute(f"PRAGMA busy_timeout={DB_BUSY_TIMEOUT_MS}")
            self._init_schema()
        return self._conn

    def _init_schema(self) -> None:
        self._conn.executescript(SCHEMA)  # type: ignore[union-attr]
        self._run_add_column_migrations()
        self._run_post_migration_indexes()
        self._backfill_expires_at()

    def _run_add_column_migrations(self) -> None:
        for table, column_def in _ADD_COLUMN_MIGRATIONS:
            try:
                self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column_def}")  # type: ignore[union-attr]
                self._conn.commit()  # type: ignore[union-attr]
            except sqlite3.OperationalError as e:
                if "duplicate column name" not in str(e):
                    raise

    def _run_post_migration_indexes(self) -> None:
        """Create indexes that depend on columns added by _run_add_column_migrations."""
        for stmt in _POST_MIGRATION_INDEXES:
            self._conn.execute(stmt)  # type: ignore[union-attr]
        self._conn.commit()  # type: ignore[union-attr]

    def _backfill_expires_at(self) -> None:
        """One-time backfill of expires_at column from JSON blob data.

        Safe to run on every startup — only updates rows where expires_at IS NULL.
        """
        # Table names and column names are internal constants — not user input.
        for table, key_col in (
            ("auth_codes", "code"),
            ("access_tokens", "token"),
            ("refresh_tokens", "token"),
        ):
            rows = self._conn.execute(  # type: ignore[union-attr]
                f"SELECT {key_col}, data FROM {table} WHERE expires_at IS NULL"  # noqa: S608
            ).fetchall()
            for row in rows:
                try:
                    expires_at = json.loads(row["data"]).get("expires_at")
                except (json.JSONDecodeError, TypeError):
                    continue
                if expires_at is None:
                    continue
                self._conn.execute(  # type: ignore[union-attr]
                    f"UPDATE {table} SET expires_at = ? WHERE {key_col} = ?",  # noqa: S608
                    (int(float(expires_at)), row[key_col]),
                )
            self._conn.commit()  # type: ignore[union-attr]

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # Clients
    # ------------------------------------------------------------------

    def save_client(self, client_info: OAuthClientInformationFull) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO clients (client_id, client_info, created_at) "
                "VALUES (?, ?, strftime('%s', 'now'))",
                (client_info.client_id, client_info.model_dump_json()),
            )
            self.conn.commit()

    def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT client_info FROM clients WHERE client_id = ?",
                (client_id,),
            ).fetchone()
        if row is None:
            return None
        return OAuthClientInformationFull.model_validate_json(row["client_info"])

    # ------------------------------------------------------------------
    # Authorization codes
    # ------------------------------------------------------------------

    def save_auth_code(self, code: str, auth_code: AuthorizationCode) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO auth_codes (code, data, created_at, expires_at) "
                "VALUES (?, ?, strftime('%s', 'now'), ?)",
                (
                    code,
                    auth_code.model_dump_json(),
                    int(float(auth_code.expires_at)) if auth_code.expires_at is not None else None,
                ),
            )
            self.conn.commit()

    def get_auth_code(self, code: str) -> AuthorizationCode | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT data FROM auth_codes WHERE code = ?",
                (code,),
            ).fetchone()
        if row is None:
            return None
        return AuthorizationCode.model_validate_json(row["data"])

    def delete_auth_code(self, code: str) -> None:
        with self._lock:
            self.conn.execute("DELETE FROM auth_codes WHERE code = ?", (code,))
            self.conn.commit()

    # ------------------------------------------------------------------
    # Access tokens
    # ------------------------------------------------------------------

    def save_access_token(self, token: str, access_token: AccessToken) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO access_tokens (token, data, created_at, expires_at) "
                "VALUES (?, ?, strftime('%s', 'now'), ?)",
                (
                    token,
                    access_token.model_dump_json(),
                    int(access_token.expires_at) if access_token.expires_at is not None else None,
                ),
            )
            self.conn.commit()

    def get_access_token(self, token: str) -> AccessToken | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT data FROM access_tokens WHERE token = ?",
                (token,),
            ).fetchone()
        if row is None:
            return None
        return AccessToken.model_validate_json(row["data"])

    def delete_access_token(self, token: str) -> None:
        with self._lock:
            self.conn.execute("DELETE FROM access_tokens WHERE token = ?", (token,))
            self.conn.commit()

    # ------------------------------------------------------------------
    # Refresh tokens
    # ------------------------------------------------------------------

    def save_refresh_token(self, token: str, refresh_token: RefreshToken) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO refresh_tokens (token, data, created_at, expires_at) "
                "VALUES (?, ?, strftime('%s', 'now'), ?)",
                (
                    token,
                    refresh_token.model_dump_json(),
                    int(refresh_token.expires_at) if refresh_token.expires_at is not None else None,
                ),
            )
            self.conn.commit()

    def get_refresh_token(self, token: str) -> RefreshToken | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT data FROM refresh_tokens WHERE token = ?",
                (token,),
            ).fetchone()
        if row is None:
            return None
        return RefreshToken.model_validate_json(row["data"])

    def delete_refresh_token(self, token: str) -> None:
        with self._lock:
            self.conn.execute("DELETE FROM refresh_tokens WHERE token = ?", (token,))
            self.conn.commit()

    # ------------------------------------------------------------------
    # Token pairs (access <-> refresh mapping for selective revocation)
    # ------------------------------------------------------------------

    def save_token_pair(self, access_token: str, refresh_token: str) -> None:
        """Link an access token to its paired refresh token."""
        with self._lock:
            self.conn.execute(
                "INSERT INTO token_pairs (access_token, refresh_token, created_at) "
                "VALUES (?, ?, ?)",
                (access_token, refresh_token, int(time.time())),
            )
            self.conn.commit()

    def save_issued_token_pair(
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
        with self._lock, self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO access_tokens (token, data, created_at, expires_at) "
                "VALUES (?, ?, strftime('%s', 'now'), ?)",
                (
                    access_token_str,
                    access_token.model_dump_json(),
                    int(access_token.expires_at) if access_token.expires_at is not None else None,
                ),
            )
            self.conn.execute(
                "INSERT OR REPLACE INTO refresh_tokens (token, data, created_at, expires_at) "
                "VALUES (?, ?, strftime('%s', 'now'), ?)",
                (
                    refresh_token_str,
                    refresh_token.model_dump_json(),
                    int(refresh_token.expires_at) if refresh_token.expires_at is not None else None,
                ),
            )
            self.conn.execute(
                "INSERT INTO token_pairs (access_token, refresh_token, created_at) "
                "VALUES (?, ?, ?)",
                (access_token_str, refresh_token_str, int(time.time())),
            )

    def get_paired_refresh_token(self, access_token: str) -> str | None:
        """Get the refresh token paired with an access token."""
        with self._lock:
            row = self.conn.execute(
                "SELECT refresh_token FROM token_pairs WHERE access_token = ?",
                (access_token,),
            ).fetchone()
        return row["refresh_token"] if row else None

    def get_paired_access_tokens(self, refresh_token: str) -> list[str]:
        """Get all access tokens paired with a refresh token."""
        with self._lock:
            rows = self.conn.execute(
                "SELECT access_token FROM token_pairs WHERE refresh_token = ?",
                (refresh_token,),
            ).fetchall()
        return [row["access_token"] for row in rows]

    def delete_token_pair_by_access(self, access_token: str) -> None:
        with self._lock:
            self.conn.execute(
                "DELETE FROM token_pairs WHERE access_token = ?",
                (access_token,),
            )
            self.conn.commit()

    def delete_token_pair_by_refresh(self, refresh_token: str) -> None:
        with self._lock:
            self.conn.execute(
                "DELETE FROM token_pairs WHERE refresh_token = ?",
                (refresh_token,),
            )
            self.conn.commit()

    # ------------------------------------------------------------------
    # Atomic refresh token rotation (HIGH-1)
    # ------------------------------------------------------------------

    def rotate_refresh_token(
        self,
        old_refresh_token_str: str,
        new_access_token_str: str,
        new_access_token: AccessToken,
        new_refresh_token_str: str,
        new_refresh_token: RefreshToken,
    ) -> None:
        """Atomically revoke old refresh + its paired access tokens and issue new pair.

        Wraps all operations in a single SQLite transaction + threading.Lock so
        a crash or coroutine interleaving cannot leave partial state (closes HIGH-1
        refresh rotation atomicity gap).
        """
        with self._lock, self.conn:
            # 1. Collect access tokens paired with old refresh (so we can revoke them)
            paired_rows = self.conn.execute(
                "SELECT access_token FROM token_pairs WHERE refresh_token = ?",
                (old_refresh_token_str,),
            ).fetchall()
            paired_access = [r["access_token"] for r in paired_rows]

            # 2. Delete old access tokens
            for at in paired_access:
                self.conn.execute("DELETE FROM access_tokens WHERE token = ?", (at,))

            # 3. Delete old pair rows
            self.conn.execute(
                "DELETE FROM token_pairs WHERE refresh_token = ?",
                (old_refresh_token_str,),
            )

            # 4. Delete old refresh token
            self.conn.execute(
                "DELETE FROM refresh_tokens WHERE token = ?",
                (old_refresh_token_str,),
            )

            # 5. Insert new access token with indexed expires_at
            self.conn.execute(
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
            self.conn.execute(
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
            self.conn.execute(
                "INSERT INTO token_pairs (access_token, refresh_token, created_at) "
                "VALUES (?, ?, ?)",
                (new_access_token_str, new_refresh_token_str, int(time.time())),
            )

    # ------------------------------------------------------------------
    # Rate limit events (login failures, register attempts, etc.)
    # ------------------------------------------------------------------

    def record_rate_limit_event(self, event_key: str) -> None:
        """Record a single rate-limit event (e.g. 'login_failure:1.2.3.4')."""
        with self._lock:
            self.conn.execute(
                "INSERT INTO rate_limit_events (event_key, occurred_at) VALUES (?, ?)",
                (event_key, int(time.time())),
            )
            self.conn.commit()

    def count_rate_limit_events(self, event_key: str, window_secs: int) -> int:
        """Count events for a key that occurred within the last window_secs seconds."""
        with self._lock:
            cutoff = int(time.time()) - window_secs
            row = self.conn.execute(
                "SELECT COUNT(*) AS c FROM rate_limit_events "
                "WHERE event_key = ? AND occurred_at >= ?",
                (event_key, cutoff),
            ).fetchone()
            return int(row["c"]) if row else 0

    def prune_rate_limit_events(self, retention_secs: int) -> int:
        """Delete events older than retention_secs. Returns rows deleted."""
        with self._lock:
            cutoff = int(time.time()) - retention_secs
            cur = self.conn.execute(
                "DELETE FROM rate_limit_events WHERE occurred_at < ?",
                (cutoff,),
            )
            self.conn.commit()
            return cur.rowcount

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup_expired(self) -> int:
        """Delete expired auth codes, access tokens, refresh tokens, and prune
        stale rate-limit events.

        Uses indexed expires_at for O(log n) per-table deletion.
        Also cascades: access tokens paired with an expired refresh token are
        removed before the refresh token itself. Returns total rows deleted.
        """
        with self._lock:
            now = int(time.time())
            deleted = 0

            # 1. Expired auth codes — expires_at is NOT NULL for new rows; legacy NULLs
            #    are treated as expired (same as old default_expired=True behavior).
            cur = self.conn.execute(
                "DELETE FROM auth_codes " "WHERE expires_at IS NULL OR expires_at < ?",
                (now,),
            )
            deleted += cur.rowcount

            # 2. Expired access tokens (+ their pair rows)
            self.conn.execute(
                "DELETE FROM token_pairs "
                "WHERE access_token IN (SELECT token FROM access_tokens "
                "                       WHERE expires_at IS NOT NULL AND expires_at < ?)",
                (now,),
            )
            cur = self.conn.execute(
                "DELETE FROM access_tokens " "WHERE expires_at IS NOT NULL AND expires_at < ?",
                (now,),
            )
            deleted += cur.rowcount

            # 3. Cascade: access tokens paired with an expired refresh token
            cur = self.conn.execute(
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
            self.conn.execute(
                "DELETE FROM token_pairs "
                "WHERE refresh_token IN (SELECT token FROM refresh_tokens "
                "                        WHERE expires_at IS NOT NULL AND expires_at < ?)",
                (now,),
            )
            cur = self.conn.execute(
                "DELETE FROM refresh_tokens " "WHERE expires_at IS NOT NULL AND expires_at < ?",
                (now,),
            )
            deleted += cur.rowcount

            self.conn.commit()

        # 5. Prune stale rate-limit events (own lock inside)
        deleted += self.prune_rate_limit_events(RATE_LIMIT_EVENT_RETENTION_SECS)
        return deleted
