"""Verify migration 0004 runs as a no-op in Mode 3 (JOURNAL_OPERATOR_EMAIL unset).

In Mode 3 / fresh-DB deployments, JOURNAL_OPERATOR_EMAIL is not set.
Migration 0004 should run all phases unconditionally and succeed because
there are no pre-existing tenant rows to violate the null-count guard.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import asyncpg
import pytest

pytestmark = [pytest.mark.integration]


def test_mode3_skip_logic(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify alembic upgrade head succeeds when JOURNAL_OPERATOR_EMAIL is unset.

    Distinguishes between infrastructure failure (PG not reachable) and
    real DDL errors. Only infrastructure failures produce a skip.
    """
    monkeypatch.delenv("JOURNAL_OPERATOR_EMAIL", raising=False)

    project_root = Path(__file__).resolve().parents[2]
    test_db_url = os.environ.get(
        "TEST_DATABASE_URL",
        "postgresql://journal:testpass@localhost:5433/journal_test",
    )
    env = {
        **os.environ,
        "JOURNAL_DB_MIGRATION_URL": test_db_url,
    }

    result = subprocess.run(  # noqa: S603 -- sys.executable is trusted, args are literals
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    stderr_lower = result.stderr.lower()
    if result.returncode != 0:
        if "could not connect to server" in stderr_lower or "connection refused" in stderr_lower:
            pytest.skip("PostgreSQL not reachable -- infrastructure missing")
        pytest.fail(
            "alembic upgrade head failed:\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )


@pytest.mark.asyncio(loop_scope="session")
async def test_mode3_noop_no_email(pool: asyncpg.Pool) -> None:
    """Verify schema state after alembic upgrade head with no operator email.

    Asserts:
    1. users table is empty (no operator row seeded).
    2. entries.user_id column exists (Phase 1 ran).
    3. entries.user_id is NOT NULL (Phase 5 ran).
    """
    async with pool.acquire() as conn:
        count = await conn.fetchval("SELECT COUNT(*) FROM users")
    assert count == 0, "Mode 3 fresh DB must have zero users rows"

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT column_name, is_nullable "
            "FROM information_schema.columns "
            "WHERE table_name = 'entries' AND column_name = 'user_id'"
        )
    assert row is not None, "entries.user_id column not found -- Phase 1 did not run"

    assert (
        row["is_nullable"] == "NO"
    ), "entries.user_id is nullable but must be NOT NULL -- Phase 5 did not run"
