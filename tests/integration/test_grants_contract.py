"""Contract test: post-migration grant state matches deployment/grants.sql.

Asserts that after ``alembic upgrade head`` the privileges held by
journal_app and journal_admin match exactly what deployment/grants.sql
declares. Adding a new table-grant to grants.sql requires updating
EXPECTED_GRANTS below -- that coupling is the drift-detection signal.

The test uses the journal_rls_test DB provisioned by ``_rls_provisioned``
(tests/conftest.py). If the DB is unavailable the test skips cleanly.
"""

from __future__ import annotations

import asyncpg
import pytest  # noqa: F401

# ---------------------------------------------------------------------------
# Canonical privilege expectations -- mirrors deployment/grants.sql exactly.
#
# Format: (role, table, privilege) -> expected bool
#
# Tenant tables + users: journal_app has full CRUD, journal_admin has ALL.
# audit_log: journal_app INSERT only, journal_admin SELECT + INSERT only.
# ---------------------------------------------------------------------------

_TENANT_TABLES = ("topics", "entries", "conversations", "messages", "entry_embeddings")
_CRUD = ("SELECT", "INSERT", "UPDATE", "DELETE")

EXPECTED_GRANTS: dict[tuple[str, str, str], bool] = {}

# Tenant tables: journal_app -- full CRUD
for _tbl in _TENANT_TABLES:
    for _priv in _CRUD:
        EXPECTED_GRANTS[("journal_app", _tbl, _priv)] = True

# Tenant tables: journal_admin -- INSERT, UPDATE, DELETE, REFERENCES, TRIGGER
for _tbl in _TENANT_TABLES:
    for _priv in ("INSERT", "UPDATE", "DELETE", "REFERENCES", "TRIGGER"):
        EXPECTED_GRANTS[("journal_admin", _tbl, _priv)] = True

# users table: same as tenant tables
for _priv in _CRUD:
    EXPECTED_GRANTS[("journal_app", "users", _priv)] = True
for _priv in ("INSERT", "UPDATE", "DELETE", "REFERENCES", "TRIGGER"):
    EXPECTED_GRANTS[("journal_admin", "users", _priv)] = True

# audit_log: append-only least-privilege (migration 0010)
EXPECTED_GRANTS[("journal_app", "audit_log", "SELECT")] = False
EXPECTED_GRANTS[("journal_app", "audit_log", "INSERT")] = True
EXPECTED_GRANTS[("journal_app", "audit_log", "UPDATE")] = False
EXPECTED_GRANTS[("journal_app", "audit_log", "DELETE")] = False

EXPECTED_GRANTS[("journal_admin", "audit_log", "SELECT")] = True
EXPECTED_GRANTS[("journal_admin", "audit_log", "INSERT")] = True
EXPECTED_GRANTS[("journal_admin", "audit_log", "UPDATE")] = False
EXPECTED_GRANTS[("journal_admin", "audit_log", "DELETE")] = False


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_grants_match_migrations(admin_pool: asyncpg.Pool) -> None:
    """Post-alembic-upgrade-head grant state matches deployment/grants.sql.

    Failures here mean grants.sql and the migrations have drifted.
    Fix by either correcting grants.sql or adding a new migration to
    restore the expected privilege state.
    """
    failures: list[str] = []

    async with admin_pool.acquire() as conn:
        for (role, table, priv), expected in EXPECTED_GRANTS.items():
            actual: bool = await conn.fetchval(
                "SELECT has_table_privilege($1, $2, $3)",
                role,
                table,
                priv,
            )
            if actual != expected:
                direction = "True" if expected else "False"
                failures.append(f"({role}, {table}, {priv}): expected={direction} got={actual}")

    assert not failures, "Grant state diverges from deployment/grants.sql:\n" + "\n".join(
        f"  {f}" for f in failures
    )
