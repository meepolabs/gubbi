"""Rate-limit event storage for OAuth login/register endpoints.

Extracted from `gubbi/oauth/storage.py` so the rate-limit concern can evolve
independently of the OAuth token/code/client persistence concerns. Both
classes share the same SQLite db file (WAL mode permits concurrent readers
and one writer); the `rate_limit_events` table DDL still lives in
`OAuthStorage._init_schema` so schema management stays centralized.

`RateLimitStorage` owns its own `threading.Lock` and connection -- it does
NOT share `OAuthStorage._lock` because the lock scope is per-connection.

Prerequisite: the `rate_limit_events` table is created by
`OAuthStorage._init_schema`. Callers MUST construct (and trigger schema
init on) an `OAuthStorage` against the same `db_path` before exercising
`RateLimitStorage` methods. To make the failure mode obvious instead of
surfacing a raw `sqlite3.OperationalError: no such table`, the first
method call lazily verifies the table exists and raises `RuntimeError`
with a clear remediation hint when it does not.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

from gubbi.storage.constants import DB_BUSY_TIMEOUT_MS


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
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()
        self._schema_verified = False

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
            )
            self._conn.row_factory = sqlite3.Row
            self._conn.execute(f"PRAGMA busy_timeout={DB_BUSY_TIMEOUT_MS}")
        if not self._schema_verified:
            self._verify_schema(self._conn)
            self._schema_verified = True
        return self._conn

    @staticmethod
    def _verify_schema(conn: sqlite3.Connection) -> None:
        """Raise RuntimeError if the `rate_limit_events` table is absent.

        The table is owned by `OAuthStorage._init_schema`; if it's missing,
        a caller has constructed `RateLimitStorage` standalone without first
        initializing `OAuthStorage` on the same db_path. Surface a clear
        remediation hint instead of a raw `sqlite3.OperationalError`.
        """
        row = conn.execute(
            "SELECT name FROM sqlite_master " "WHERE type = 'table' AND name = 'rate_limit_events'"
        ).fetchone()
        if row is None:
            raise RuntimeError(
                "rate_limit_events table not initialised; construct "
                "OAuthStorage on the same db_path first"
            )

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
        # Re-verify schema if the connection is later reopened against a
        # potentially-different db file state.
        self._schema_verified = False

    def record_event(self, event_key: str) -> None:
        """Record a single rate-limit event (e.g. 'login_failure:1.2.3.4')."""
        with self._lock:
            self.conn.execute(
                "INSERT INTO rate_limit_events (event_key, occurred_at) VALUES (?, ?)",
                (event_key, int(time.time())),
            )
            self.conn.commit()

    def count_events(self, event_key: str, window_secs: int) -> int:
        """Count events for a key that occurred within the last window_secs seconds."""
        with self._lock:
            cutoff = int(time.time()) - window_secs
            row = self.conn.execute(
                "SELECT COUNT(*) AS c FROM rate_limit_events "
                "WHERE event_key = ? AND occurred_at >= ?",
                (event_key, cutoff),
            ).fetchone()
            return int(row["c"]) if row else 0

    def prune(self, retention_secs: int) -> int:
        """Delete events older than retention_secs. Returns rows deleted."""
        with self._lock:
            cutoff = int(time.time()) - retention_secs
            cur = self.conn.execute(
                "DELETE FROM rate_limit_events WHERE occurred_at < ?",
                (cutoff,),
            )
            self.conn.commit()
            return int(cur.rowcount)
