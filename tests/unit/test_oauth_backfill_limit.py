"""Test OAuth backfill caps rows at 10000 per startup (M-9.11)."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path


def test_oauth_backfill_caps_at_10000(tmp_path: Path) -> None:
    """Only 10 000 NULL expires_at rows are backfilled per startup."""
    from journalctl.oauth.storage import OAuthStorage

    db_path = tmp_path / "backfill_test.db"
    conn = sqlite3.connect(str(db_path))
    # Create table without expires_at column (legacy schema) and insert rows.
    conn.execute("""
        CREATE TABLE access_tokens (
            token TEXT PRIMARY KEY,
            data TEXT NOT NULL,
            created_at INTEGER NOT NULL
        )
    """)

    # Insert exactly 15000 legacy rows with no expires_at column.
    base_time = int(time.time())
    for i in range(15000):
        token = f"legacy-token-{i:05d}"
        data = json.dumps(
            {
                "token": token,
                "expires_at": base_time + 3600,
                "client_id": "test",
                "scopes": [],
            }
        )
        conn.execute(
            "INSERT INTO access_tokens (token, data, created_at) VALUES (?, ?, ?)",
            (token, data, i),
        )
    conn.commit()
    conn.close()

    # Re-open via OAuthStorage — this triggers backfill.
    storage = OAuthStorage(db_path)
    _ = storage.conn  # forces lazy init -> migrations + backfill

    # Count how many rows now have expires_at set.
    result = storage.conn.execute(
        "SELECT COUNT(*) AS c FROM access_tokens WHERE expires_at IS NOT NULL"
    )
    row = result.fetchone()
    count_with_expires = row["c"] if row else 0

    # Should be capped at 10000 (excess rows remain NULL).
    assert (
        count_with_expires == 10000
    ), f"Expected exactly 10000 backfilled rows, got {count_with_expires}"

    storage.close()
