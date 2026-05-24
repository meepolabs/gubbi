"""Contract test: post-migration grant state matches the audited squashed baseline.

Asserts that after ``alembic upgrade head`` the privileges held by
journal_app and journal_admin on the gubbi-owned subset of the schema
match the audited end-state captured in
``deployment/scripts/baseline_schema.sql`` and
``deployment/scripts/grants.sql``.

Coverage: gubbi tables only (audit_log, conversations, entries,
entry_embeddings, extraction_jobs, messages, topics, users) plus the
otel_ro role posture. Cloud-side tables (tenants, subscriptions,
llm_budgets, outbox_events, stripe_events) are covered by gubbi-cloud's
lifespan probes -- this test does not assume they exist (gubbi-cloud's
chain may not have run in the gubbi-only test fixtures).

Failures here mean either:
  - grants.sql / baseline_schema.sql drifted
  - a future migration changed grants without updating EXPECTED_GRANTS
Fix by correcting whichever side is wrong; the test mirrors
deployment/scripts/grants.sql exactly.
"""

from __future__ import annotations

import asyncpg
import pytest

# ---------------------------------------------------------------------------
# Canonical privilege expectations -- mirrors the audited baseline state.
#
# Format: (role, table, privilege) -> expected bool
# ---------------------------------------------------------------------------

# gubbi tables where journal_app has full CRUD.
_FULL_CRUD_TABLES = (
    "topics",
    "entries",
    "conversations",
    "messages",
    "entry_embeddings",
    "extraction_jobs",
)
_CRUD = ("SELECT", "INSERT", "UPDATE", "DELETE")

EXPECTED_GRANTS: dict[tuple[str, str, str], bool] = {}

# Full-CRUD gubbi tables: journal_app has all four
for _tbl in _FULL_CRUD_TABLES:
    for _priv in _CRUD:
        EXPECTED_GRANTS[("journal_app", _tbl, _priv)] = True

# Full-CRUD gubbi tables: journal_admin has all four (and more, but we only
# assert the CRUD subset here -- has_table_privilege returns True even if
# only a wider grant is held)
for _tbl in _FULL_CRUD_TABLES:
    for _priv in _CRUD:
        EXPECTED_GRANTS[("journal_admin", _tbl, _priv)] = True

# users: narrowed -- journal_app gets SELECT + UPDATE only (gubbi 0019
# REVOKE'd INSERT and DELETE; preserved in the squashed baseline).
EXPECTED_GRANTS[("journal_app", "users", "SELECT")] = True
EXPECTED_GRANTS[("journal_app", "users", "UPDATE")] = True
EXPECTED_GRANTS[("journal_app", "users", "INSERT")] = False
EXPECTED_GRANTS[("journal_app", "users", "DELETE")] = False

EXPECTED_GRANTS[("journal_admin", "users", "SELECT")] = True
EXPECTED_GRANTS[("journal_admin", "users", "INSERT")] = True
EXPECTED_GRANTS[("journal_admin", "users", "UPDATE")] = True
EXPECTED_GRANTS[("journal_admin", "users", "DELETE")] = True

# audit_log: append-only least-privilege (gubbi migration 0010)
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
async def test_grants_match_audited_baseline(admin_pool: asyncpg.Pool) -> None:
    """Post-alembic-upgrade-head grant state matches the audited contract.

    Failures here mean grants.sql / baseline_schema.sql / migrations have
    drifted. Fix by aligning whichever side is wrong; this test mirrors
    deployment/scripts/grants.sql exactly for the gubbi-owned subset.
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

    assert not failures, "Grant state diverges from deployment/scripts/grants.sql:\n" + "\n".join(
        f"  {f}" for f in failures
    )


@pytest.mark.asyncio
async def test_otel_ro_posture(admin_pool: asyncpg.Pool) -> None:
    """otel_ro role exists, NOLOGIN, member of pg_monitor, no data-table SELECT."""
    failures: list[str] = []

    async with admin_pool.acquire() as conn:
        exists = await conn.fetchval(
            "SELECT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'otel_ro')"
        )
        if not exists:
            failures.append("otel_ro role missing")

        nologin = await conn.fetchval(
            "SELECT NOT rolcanlogin FROM pg_roles WHERE rolname = 'otel_ro'"
        )
        if not nologin:
            failures.append("otel_ro is LOGIN-enabled (should be NOLOGIN)")

        has_monitor = await conn.fetchval(
            """
            SELECT EXISTS (
                SELECT 1 FROM pg_auth_members am
                JOIN pg_roles r ON r.oid = am.member
                JOIN pg_roles g ON g.oid = am.roleid
                WHERE r.rolname = 'otel_ro' AND g.rolname = 'pg_monitor'
            )
            """
        )
        if not has_monitor:
            failures.append("otel_ro is not a member of pg_monitor")

        # otel_ro must NOT have SELECT on any gubbi data table.
        for tbl in (*_FULL_CRUD_TABLES, "users", "audit_log"):
            has_select = await conn.fetchval(
                "SELECT has_table_privilege('otel_ro', $1, 'SELECT')", tbl
            )
            if has_select:
                failures.append(f"otel_ro has unexpected SELECT on {tbl}")

    assert not failures, "otel_ro posture is wrong:\n" + "\n".join(f"  {f}" for f in failures)
