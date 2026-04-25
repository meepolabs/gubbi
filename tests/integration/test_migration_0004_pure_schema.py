"""Verify migration 0004 is pure DDL: no INSERT/UPDATE into users columns.

This test has two assertions:

a) The migration file source contains zero INSERT/UPDATE/DELETE statements that
   mutate the ``users`` table (checked via regex on the file at runtime).
b) After alembic upgrade head, every tenant table has ``user_id`` with a NOT
   NULL constraint, confirmed by querying ``information_schema.columns``.

Uses the shared ``pool`` fixture from conftest.py (session-scoped). The pool
is already migrated at this point so no subprocess invocation is needed.
"""

from __future__ import annotations

import pathlib
import re

import asyncpg
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.asyncio(loop_scope="session")]

_TENANT_TABLES = ("topics", "entries", "conversations", "messages", "entry_embeddings")

_MIGRATION_FILE = (
    pathlib.Path(__file__).resolve().parents[2]
    / "journalctl"
    / "alembic"
    / "versions"
    / "20260419_0004_add_user_id_to_tenants.py"
)  # noqa: E501


async def test_migration_0004_has_no_user_mutations() -> None:
    """Migration file source must not contain INSERT/UPDATE/DELETE targeting users."""
    src = _MIGRATION_FILE.read_text(encoding="utf-8")

    insert_users = re.search(r"INSERT\s+INTO\s+users\b", src, re.IGNORECASE)
    assert insert_users is None, (
        "Migration 0004 must not contain INSERT INTO users. "
        "Operator provisioning is a separate script."
    )

    update_users = re.search(r"UPDATE\s+users\s+SET\b", src, re.IGNORECASE)
    assert update_users is None, (
        "Migration 0004 must not contain UPDATE users SET. "
        "Operator provisioning is a separate script."
    )

    delete_users = re.search(r"DELETE\s+FROM\s+users\b", src, re.IGNORECASE)
    assert delete_users is None, "Migration 0004 must not contain DELETE FROM users."


async def test_tenant_tables_have_not_null_user_id(pool: asyncpg.Pool) -> None:
    """Every tenant table's user_id column must have is_nullable = 'NO'."""
    query = (
        "SELECT is_nullable, column_name "
        "FROM information_schema.columns "
        "WHERE table_name = $1 AND column_name = 'user_id'"
    )

    for table in _TENANT_TABLES:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(query, table)
        assert row is not None, f"user_id column not found on {table}"
        assert row["is_nullable"] == "NO", f"user_id on {table} is nullable but must be NOT NULL"
