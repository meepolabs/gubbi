"""Verify the migration chain runs as a no-op in Mode 3 (JOURNAL_OPERATOR_EMAIL unset).

In Mode 3 / fresh-DB deployments, JOURNAL_OPERATOR_EMAIL is not set.
The chain should run all phases unconditionally and succeed because
there are no pre-existing tenant rows to violate the null-count guard.

The claim is about a FRESH database, so the module migrates its own throwaway
database rather than reading the shared ``journal_test``: other suites seed
``users`` there, and ``clean_pool`` does not truncate it. The scratch database
lives on the working cluster because the migration grants to journal_app and
journal_admin, which exist there; the disposable cluster must stay free of them.
"""

from __future__ import annotations

import os
import subprocess
import sys
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import asyncpg
import pytest
import pytest_asyncio

from tests.conftest import TEST_DATABASE_URL
from tests.fixtures.cluster_roles import maintenance_dsn

pytestmark = [pytest.mark.integration]

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DATABASE_PREFIX = "journal_mode3_"
_OPERATOR_EMAIL_ENV = "JOURNAL_OPERATOR_EMAIL"


@dataclass(frozen=True)
class Mode3Database:
    """A freshly created database and the result of migrating it with no operator email."""

    dsn: str
    upgrade: subprocess.CompletedProcess[str]


def _with_database(dsn: str, name: str) -> str:
    return urlunparse(urlparse(dsn)._replace(path=f"/{name}"))


def _upgrade_head_without_operator_email(dsn: str) -> subprocess.CompletedProcess[str]:
    env = {key: value for key, value in os.environ.items() if key != _OPERATOR_EMAIL_ENV}
    env["JOURNAL_DB_MIGRATION_URL"] = dsn
    return subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=_PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest_asyncio.fixture(scope="module", loop_scope="session")
async def mode3_database() -> AsyncIterator[Mode3Database]:
    """A per-module database migrated to head in Mode 3, dropped on teardown.

    Reaching Postgres is the one prerequisite whose absence skips; it is probed
    before anything is created.
    """
    name = f"{_DATABASE_PREFIX}{uuid.uuid4().hex[:12]}"
    dsn = _with_database(TEST_DATABASE_URL, name)
    try:
        admin = await asyncpg.connect(maintenance_dsn(TEST_DATABASE_URL), timeout=5)
    except (OSError, asyncpg.PostgresError, TimeoutError) as exc:
        pytest.skip(f"PostgreSQL not reachable -- infrastructure missing: {exc}")

    try:
        await admin.execute(f'CREATE DATABASE "{name}"')
        try:
            setup = await asyncpg.connect(dsn, timeout=5)
            try:
                await setup.execute("CREATE EXTENSION IF NOT EXISTS vector")
            finally:
                await setup.close()
            yield Mode3Database(dsn=dsn, upgrade=_upgrade_head_without_operator_email(dsn))
        finally:
            await admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
    finally:
        await admin.close()


def test_mode3_skip_logic(mode3_database: Mode3Database) -> None:
    """Verify alembic upgrade head succeeds on a fresh database with no operator email."""
    result = mode3_database.upgrade
    if result.returncode != 0:
        pytest.fail(
            f"alembic upgrade head failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


@pytest.mark.asyncio(loop_scope="session")
async def test_mode3_noop_no_email(mode3_database: Mode3Database) -> None:
    """Verify schema state after alembic upgrade head with no operator email.

    Asserts:
    1. users table is empty (no operator row seeded).
    2. entries.user_id column exists (Phase 1 ran).
    3. entries.user_id is NOT NULL (Phase 5 ran).
    """
    assert mode3_database.upgrade.returncode == 0, mode3_database.upgrade.stderr

    conn = await asyncpg.connect(mode3_database.dsn, timeout=5)
    try:
        count = await conn.fetchval("SELECT COUNT(*) FROM users")
        row = await conn.fetchrow(
            "SELECT column_name, is_nullable "
            "FROM information_schema.columns "
            "WHERE table_name = 'entries' AND column_name = 'user_id'"
        )
    finally:
        await conn.close()

    assert count == 0, "Mode 3 fresh DB must have zero users rows"
    assert row is not None, "entries.user_id column not found -- Phase 1 did not run"
    assert row["is_nullable"] == "NO", (
        "entries.user_id is nullable but must be NOT NULL -- Phase 5 did not run"
    )
