"""Regression test for migration 0027: extraction_jobs RLS NULLIF wrap.

Migration 0023 (extraction_jobs_relocate) created the
``extraction_jobs_user_isolation`` policy without the NULLIF wrap that
0007 had established as the canonical pattern for the five legacy
tenant tables. Without NULLIF, the policy casts
``current_setting('app.current_user_id', true)`` directly to ``::uuid``
and raises ``invalid input syntax for type uuid: ""`` whenever a pooled
connection has previously bound the GUC and then sees an empty-string
residue on the next checkout.

Migration 0027 ports the 0007 NULLIF wrap to ``extraction_jobs``. This
test pins that fix:

    a. Acquire a connection from ``app_pool`` (RLS enforced, NO BYPASSRLS).
    b. Set ``app.current_user_id`` to the empty string via ``set_config``.
    c. Run ``SELECT * FROM extraction_jobs`` and assert it returns zero
       rows -- crucially, that it does NOT raise
       ``InvalidTextRepresentationError``.
    d. Set the GUC to a real UUID and assert the query still works
       (positive control: the policy is not just universally denying).

The fixtures ``app_pool``, ``admin_pool``, and ``clean_rls_db`` are
session-scoped and provisioned in ``tests/conftest.py``; they migrate a
fresh PG (the ``journal_rls_test`` database) to head before yielding.
``seeded_a`` (auto-discovered via ``tests/integration/conftest.py``)
plants a user + topic + conversation + entries via ``seed_for`` so the
positive-control test gets a wire-correct ``conversations`` row to hang
the ``extraction_jobs`` FK off without re-implementing the schema.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import asyncpg
import pytest

# ``seed_for`` is a helper (not a fixture) imported at module scope --
# matches the pattern used by test_rls_isolation.py. The actual
# ``seeded_a`` fixture is auto-discovered from tests/integration/conftest.py.
from tests.fixtures.tenants import TenantSeed, seed_for  # noqa: F401 -- type hint + helper call

# Pin asyncio loop scope to "session" so tests share the same event loop the
# session-scoped pools were created in. Mirrors test_rls_isolation.py.
pytestmark = pytest.mark.asyncio(loop_scope="session")


@pytest.mark.integration
async def test_empty_string_guc_returns_zero_rows_not_cast_error(
    app_pool: asyncpg.Pool,
    clean_rls_db: asyncpg.Pool,  # noqa: ARG001 -- side effect: TRUNCATE before/after
) -> None:
    """Empty-string ``app.current_user_id`` must return zero rows, not raise.

    Reproduces the exact pooled-connection-residue path: a previous
    ``SET LOCAL`` cleared on commit leaves an empty-string registration
    on the session. The next query under the un-hardened 0023 policy
    raised ``InvalidTextRepresentationError`` on the cast. Migration
    0027 wraps with NULLIF so the empty string becomes NULL and the
    row is filtered out.
    """
    async with app_pool.acquire() as conn, conn.transaction():
        # ``set_config(name, value, true)`` is the transaction-local form --
        # matches user_scoped_connection's prologue but with an empty value
        # to simulate the post-commit residue state.
        await conn.execute("SELECT set_config('app.current_user_id', '', true)")
        rows = await conn.fetch("SELECT * FROM extraction_jobs")
    assert rows == [], (
        "Expected empty result set with empty-string GUC; the NULLIF wrap "
        "should make user_id = NULL evaluate to UNKNOWN. If this raises "
        "InvalidTextRepresentationError, migration 0027 has regressed."
    )


@pytest.mark.integration
async def test_real_uuid_guc_returns_owned_rows(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    seeded_a: TenantSeed,
) -> None:
    """Positive control: a real user_id GUC sees that user's own rows.

    Confirms the 0027 policy is not universally denying -- it correctly
    matches when ``app.current_user_id`` is set to the row's owning UUID.
    Uses ``seeded_a`` so we get a user + conversation already wired up
    by the canonical tenant fixture; we only need to plant one
    ``extraction_jobs`` row owned by that user via ``admin_pool``
    (BYPASSRLS), so the seed itself does not have to deal with the
    policy under test.
    """
    assert seeded_a.conversation_id is not None
    user_id = seeded_a.user_id
    conversation_id = seeded_a.conversation_id
    now = datetime.now(UTC)

    async with admin_pool.acquire() as conn:
        job_id = await conn.fetchval(
            """
            INSERT INTO extraction_jobs
                (user_id, conversation_id, source, status, period_start, created_at)
            VALUES ($1, $2, $3, $4,
                    date_trunc('month', $5::timestamptz)::date,
                    $5)
            RETURNING id
            """,
            user_id,
            conversation_id,
            "test-source",
            "queued",
            now,
        )

    # Sanity: the row exists from BYPASSRLS' perspective. Assert by id
    # match rather than absolute count -- the test does not bind
    # ``clean_rls_db``, so other tests sharing this session may have
    # planted additional rows we do not own. The owning-id check below
    # is the load-bearing assertion either way.
    async with admin_pool.acquire() as conn:
        admin_ids = {row["id"] for row in await conn.fetch("SELECT id FROM extraction_jobs")}
    assert job_id in admin_ids

    # Scoped read: the owning user must see that row. Assert by id
    # membership rather than count, for the same session-sharing reason
    # noted on the admin sanity check above.
    async with app_pool.acquire() as conn, conn.transaction():
        await conn.execute(
            "SELECT set_config('app.current_user_id', $1, true)",
            str(user_id),
        )
        rows = await conn.fetch("SELECT id, user_id FROM extraction_jobs")
    scoped_ids = {row["id"] for row in rows}
    assert job_id in scoped_ids
    # Every row visible to the user_id-scoped session must be owned by
    # that user (sanity check of the policy itself).
    assert all(row["user_id"] == user_id for row in rows)

    # Cross-tenant negative: a different user_id sees zero rows.
    other_user_id = uuid4()
    async with app_pool.acquire() as conn, conn.transaction():
        await conn.execute(
            "SELECT set_config('app.current_user_id', $1, true)",
            str(other_user_id),
        )
        cross_rows = await conn.fetch("SELECT id FROM extraction_jobs")
    assert cross_rows == []
