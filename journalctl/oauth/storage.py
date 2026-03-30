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
import time
from pathlib import Path

from mcp.server.auth.provider import AccessToken, AuthorizationCode, RefreshToken
from mcp.shared.auth import OAuthClientInformationFull

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
    created_at  INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS access_tokens (
    token       TEXT PRIMARY KEY,
    data        TEXT NOT NULL,
    created_at  INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS refresh_tokens (
    token       TEXT PRIMARY KEY,
    data        TEXT NOT NULL,
    created_at  INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS token_pairs (
    access_token  TEXT NOT NULL,
    refresh_token TEXT NOT NULL,
    created_at    INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_token_pairs_access
    ON token_pairs(access_token);
CREATE INDEX IF NOT EXISTS idx_token_pairs_refresh
    ON token_pairs(refresh_token);
"""


class OAuthStorage:
    """SQLite storage layer for OAuth 2.0 entities."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None

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
        assert self._conn is not None  # noqa: S101
        self._conn.executescript(SCHEMA)

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # Clients
    # ------------------------------------------------------------------

    def save_client(self, client_info: OAuthClientInformationFull) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO clients (client_id, client_info, created_at) "
            "VALUES (?, ?, strftime('%s', 'now'))",
            (client_info.client_id, client_info.model_dump_json()),
        )
        self.conn.commit()

    def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
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
        self.conn.execute(
            "INSERT OR REPLACE INTO auth_codes (code, data, created_at) "
            "VALUES (?, ?, strftime('%s', 'now'))",
            (code, auth_code.model_dump_json()),
        )
        self.conn.commit()

    def get_auth_code(self, code: str) -> AuthorizationCode | None:
        row = self.conn.execute(
            "SELECT data FROM auth_codes WHERE code = ?",
            (code,),
        ).fetchone()
        if row is None:
            return None
        return AuthorizationCode.model_validate_json(row["data"])

    def delete_auth_code(self, code: str) -> None:
        self.conn.execute("DELETE FROM auth_codes WHERE code = ?", (code,))
        self.conn.commit()

    # ------------------------------------------------------------------
    # Access tokens
    # ------------------------------------------------------------------

    def save_access_token(self, token: str, access_token: AccessToken) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO access_tokens (token, data, created_at) "
            "VALUES (?, ?, strftime('%s', 'now'))",
            (token, access_token.model_dump_json()),
        )
        self.conn.commit()

    def get_access_token(self, token: str) -> AccessToken | None:
        row = self.conn.execute(
            "SELECT data FROM access_tokens WHERE token = ?",
            (token,),
        ).fetchone()
        if row is None:
            return None
        return AccessToken.model_validate_json(row["data"])

    def delete_access_token(self, token: str) -> None:
        self.conn.execute("DELETE FROM access_tokens WHERE token = ?", (token,))
        self.conn.commit()

    # ------------------------------------------------------------------
    # Refresh tokens
    # ------------------------------------------------------------------

    def save_refresh_token(self, token: str, refresh_token: RefreshToken) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO refresh_tokens (token, data, created_at) "
            "VALUES (?, ?, strftime('%s', 'now'))",
            (token, refresh_token.model_dump_json()),
        )
        self.conn.commit()

    def get_refresh_token(self, token: str) -> RefreshToken | None:
        row = self.conn.execute(
            "SELECT data FROM refresh_tokens WHERE token = ?",
            (token,),
        ).fetchone()
        if row is None:
            return None
        return RefreshToken.model_validate_json(row["data"])

    def delete_refresh_token(self, token: str) -> None:
        self.conn.execute("DELETE FROM refresh_tokens WHERE token = ?", (token,))
        self.conn.commit()

    # ------------------------------------------------------------------
    # Token pairs (access <-> refresh mapping for selective revocation)
    # ------------------------------------------------------------------

    def save_token_pair(self, access_token: str, refresh_token: str) -> None:
        """Link an access token to its paired refresh token."""
        self.conn.execute(
            "INSERT INTO token_pairs (access_token, refresh_token, created_at) " "VALUES (?, ?, ?)",
            (access_token, refresh_token, int(time.time())),
        )
        self.conn.commit()

    def get_paired_refresh_token(self, access_token: str) -> str | None:
        """Get the refresh token paired with an access token."""
        row = self.conn.execute(
            "SELECT refresh_token FROM token_pairs WHERE access_token = ?",
            (access_token,),
        ).fetchone()
        return row["refresh_token"] if row else None

    def get_paired_access_tokens(self, refresh_token: str) -> list[str]:
        """Get all access tokens paired with a refresh token."""
        rows = self.conn.execute(
            "SELECT access_token FROM token_pairs WHERE refresh_token = ?",
            (refresh_token,),
        ).fetchall()
        return [row["access_token"] for row in rows]

    def delete_token_pair_by_access(self, access_token: str) -> None:
        self.conn.execute(
            "DELETE FROM token_pairs WHERE access_token = ?",
            (access_token,),
        )
        self.conn.commit()

    def delete_token_pair_by_refresh(self, refresh_token: str) -> None:
        self.conn.execute(
            "DELETE FROM token_pairs WHERE refresh_token = ?",
            (refresh_token,),
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    # Pre-built queries keyed by table name — no f-string interpolation needed.
    # Each entry: (key_column_name, SELECT sql, DELETE sql)
    _CLEANUP_QUERIES: dict[str, tuple[str, str, str]] = {
        "auth_codes": (
            "code",
            "SELECT code, data FROM auth_codes",
            "DELETE FROM auth_codes WHERE code = ?",
        ),
        "access_tokens": (
            "token",
            "SELECT token, data FROM access_tokens",
            "DELETE FROM access_tokens WHERE token = ?",
        ),
        "refresh_tokens": (
            "token",
            "SELECT token, data FROM refresh_tokens",
            "DELETE FROM refresh_tokens WHERE token = ?",
        ),
    }

    _CLEANUP_PAIR_DELETE: dict[str, str] = {
        "access_token": "DELETE FROM token_pairs WHERE access_token = ?",
        "refresh_token": "DELETE FROM token_pairs WHERE refresh_token = ?",
    }

    def _cleanup_table(
        self,
        table: str,
        paired_col: str | None = None,
        *,
        default_expired: bool = False,
    ) -> int:
        """Delete rows where JSON data.expires_at < now.

        Args:
            table: Table name — must be a key in _CLEANUP_QUERIES.
            paired_col: If set, also deletes from token_pairs on this column.
            default_expired: If True, treat missing/None expires_at as expired
                (used for auth_codes which must always have an expiry).
        """
        if table not in self._CLEANUP_QUERIES:
            raise ValueError(f"Invalid table: {table!r}")
        if paired_col is not None and paired_col not in self._CLEANUP_PAIR_DELETE:
            raise ValueError(f"Invalid paired column: {paired_col!r}")

        key_col, select_sql, delete_sql = self._CLEANUP_QUERIES[table]
        now = int(time.time())
        deleted = 0
        rows = self.conn.execute(select_sql).fetchall()
        for row in rows:
            expires_at = json.loads(row["data"]).get("expires_at")
            is_expired = (
                (expires_at or 0) < now
                if default_expired
                else expires_at is not None and expires_at < now
            )
            if is_expired:
                self.conn.execute(delete_sql, (row[key_col],))
                if paired_col:
                    self.conn.execute(self._CLEANUP_PAIR_DELETE[paired_col], (row[key_col],))
                deleted += 1
        return deleted

    def _cascade_expired_refresh_to_access(self) -> int:
        """Delete access tokens whose paired refresh token has expired."""
        now = int(time.time())
        deleted = 0
        rows = self.conn.execute("SELECT token, data FROM refresh_tokens").fetchall()
        for row in rows:
            expires_at = json.loads(row["data"]).get("expires_at")
            if expires_at is not None and expires_at < now:
                for at in self.get_paired_access_tokens(row["token"]):
                    self.conn.execute(
                        "DELETE FROM access_tokens WHERE token = ?",
                        (at,),
                    )
                    deleted += 1
        return deleted

    def cleanup_expired(self) -> int:
        """Delete expired auth codes, access tokens, and refresh tokens.

        Also deletes access tokens paired with expired refresh tokens
        to prevent orphaned tokens.

        Returns the number of records deleted.
        """
        deleted = 0
        deleted += self._cleanup_table("auth_codes", default_expired=True)
        deleted += self._cleanup_table("access_tokens", "access_token")
        deleted += self._cascade_expired_refresh_to_access()
        deleted += self._cleanup_table("refresh_tokens", "refresh_token")
        self.conn.commit()
        return deleted
