"""Integration tests for migration 0005 RLS policies.

Every test here asserts that RLS actually blocks cross-tenant access —
if any of these fails, the tenant-isolation invariant is broken and
nothing downstream is safe. The file is organised into five sections:

1. BASIC VISIBILITY        - seeded rows for A are visible only to A
2. LOOKUP BY PRIMARY KEY   - RLS hides rows even when the integer PK is known
3. WITH CHECK VIOLATIONS   - INSERT / UPDATE with foreign user_id are rejected
4. UPDATE / DELETE CROSS   - cross-tenant UPDATE / DELETE affect zero rows
5. DEFAULT DENY            - app pool without scope sees nothing; admin bypass sees all

Fixtures (admin_pool, app_pool, clean_rls_db, tenant_a, tenant_b,
seeded_a, seeded_b, seed_for) are imported from conftest / tests.fixtures.
Fixture imports at module scope are REQUIRED so pytest can resolve the
transitive dependency graph (e.g. seeded_a depends on tenant_a).
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from uuid import UUID

import asyncpg
import pytest
from gubbi_common.db.user_scoped import user_scoped_connection

from journalctl.core.crypto import ContentCipher
from journalctl.storage.exceptions import ConversationNotFoundError, TopicNotFoundError
from journalctl.storage.repositories import conversations as conv_repo
from journalctl.storage.repositories import entries as entry_repo
from journalctl.storage.repositories import topics as topic_repo

# NB: tenant fixtures (tenant_a, tenant_b, seeded_a, seeded_b, seed_for,
# TenantSeed) come from tests/integration/conftest.py — pytest auto-discovers
# them by name so they must NOT be imported at module scope here (that would
# shadow the fixture function with the same name and break resolution).
from tests.fixtures.tenants import TenantSeed, seed_for  # noqa: F401 — type hint + helper call

# Integration tests share session-scoped asyncpg pools (admin_pool, app_pool).
# Pin the test loop scope to "session" so tests run in the same event loop the
# pools were created in -- otherwise asyncpg's futures error with
# "attached to a different loop".
pytestmark = pytest.mark.asyncio(loop_scope="session")

# Error regex matching Postgres row-security violations (INSERT and UPDATE WITH CHECK).
_RLS_ERROR_RE = "row-level security|row security|violates.*policy|new row violates"


# ---------------------------------------------------------------------------
# SECTION 1 - BASIC VISIBILITY
# ---------------------------------------------------------------------------


async def test_topics_visible_only_to_owner(
    app_pool: asyncpg.Pool,
    seeded_a: TenantSeed,
    seeded_b: TenantSeed,
) -> None:
    """topic_repo.list_all under a user scope returns only that user's topics."""
    async with user_scoped_connection(app_pool, user_id=seeded_a.user_id) as conn:
        a_topics, a_total = await topic_repo.list_all(conn)
    assert a_total == 1
    assert a_topics[0].topic == seeded_a.topic_path
    assert all(t.topic != seeded_b.topic_path for t in a_topics)

    async with user_scoped_connection(app_pool, user_id=seeded_b.user_id) as conn:
        b_topics, b_total = await topic_repo.list_all(conn)
    assert b_total == 1
    assert b_topics[0].topic == seeded_b.topic_path


async def test_entries_visible_only_to_owner(
    app_pool: asyncpg.Pool,
    cipher: ContentCipher,
    seeded_a: TenantSeed,
    seeded_b: TenantSeed,
) -> None:
    """entry_repo.get_by_date_range under a user scope returns only that user's entries."""
    today = date.today()
    date_from = (today - timedelta(days=14)).isoformat()
    date_to = (today + timedelta(days=1)).isoformat()

    async with user_scoped_connection(app_pool, user_id=seeded_a.user_id) as conn:
        a_rows = await entry_repo.get_by_date_range(conn, cipher, date_from, date_to)
    a_entry_ids = {r["entry_id"] for r in a_rows if r["entry_id"] is not None}
    assert a_entry_ids == set(seeded_a.entry_ids)
    assert a_entry_ids.isdisjoint(set(seeded_b.entry_ids))

    async with user_scoped_connection(app_pool, user_id=seeded_b.user_id) as conn:
        b_rows = await entry_repo.get_by_date_range(conn, cipher, date_from, date_to)
    b_entry_ids = {r["entry_id"] for r in b_rows if r["entry_id"] is not None}
    assert b_entry_ids == set(seeded_b.entry_ids)


async def test_conversations_visible_only_to_owner(
    app_pool: asyncpg.Pool,
    cipher: ContentCipher,
    seeded_a: TenantSeed,
    seeded_b: TenantSeed,
) -> None:
    """conv_repo.list_conversations under a user scope returns only that user's conversations."""
    async with user_scoped_connection(app_pool, user_id=seeded_a.user_id) as conn:
        a_convs, a_total = await conv_repo.list_conversations(conn, cipher)
    assert a_total == 1
    assert a_convs[0].id == seeded_a.conversation_id
    assert a_convs[0].id != seeded_b.conversation_id


async def test_entries_get_stats_scoped_per_user(
    app_pool: asyncpg.Pool,
    seeded_a: TenantSeed,
    seeded_b: TenantSeed,
) -> None:
    """entry_repo.get_stats reflects only the scoped user's counts."""
    async with user_scoped_connection(app_pool, user_id=seeded_a.user_id) as conn:
        a_stats = await entry_repo.get_stats(conn)
    async with user_scoped_connection(app_pool, user_id=seeded_b.user_id) as conn:
        b_stats = await entry_repo.get_stats(conn)

    # get_stats returns total_documents = entry_count + conv_count, plus
    # conversations and topics separately. seed_for plants n_entries (default 3)
    # entries and include_conversation=True adds 1 conversation -> 3 + 1 = 4.
    expected_a = len(seeded_a.entry_ids) + (1 if seeded_a.conversation_id is not None else 0)
    expected_b = len(seeded_b.entry_ids) + (1 if seeded_b.conversation_id is not None else 0)
    assert a_stats["topics"] == 1
    assert a_stats["conversations"] == 1
    assert a_stats["total_documents"] == expected_a
    assert b_stats["topics"] == 1
    assert b_stats["conversations"] == 1
    assert b_stats["total_documents"] == expected_b


# ---------------------------------------------------------------------------
# SECTION 2 - LOOKUP BY PRIMARY KEY
# ---------------------------------------------------------------------------


async def test_entries_pk_lookup_cross_tenant_returns_none(
    app_pool: asyncpg.Pool,
    cipher: ContentCipher,
    seeded_a: TenantSeed,
    seeded_b: TenantSeed,
) -> None:
    """get_text for B's entry, scoped as A, returns None (RLS hides the row)."""
    async with user_scoped_connection(app_pool, user_id=seeded_a.user_id) as conn:
        result = await entry_repo.get_text(conn, cipher, seeded_b.entry_ids[0])
    assert result is None


async def test_conversations_pk_lookup_cross_tenant_raises(
    app_pool: asyncpg.Pool,
    cipher: ContentCipher,
    seeded_a: TenantSeed,
    seeded_b: TenantSeed,
) -> None:
    """read_conversation_by_id for B's conversation, scoped as A, raises."""
    assert seeded_b.conversation_id is not None
    async with user_scoped_connection(app_pool, user_id=seeded_a.user_id) as conn:
        with pytest.raises(ConversationNotFoundError):
            await conv_repo.read_conversation_by_id(conn, cipher, seeded_b.conversation_id)


async def test_topic_get_id_cross_tenant_raises(
    app_pool: asyncpg.Pool,
    seeded_a: TenantSeed,
    seeded_b: TenantSeed,
) -> None:
    """get_id for B's topic_path, scoped as A, raises TopicNotFoundError."""
    async with user_scoped_connection(app_pool, user_id=seeded_a.user_id) as conn:
        with pytest.raises(TopicNotFoundError):
            await topic_repo.get_id(conn, seeded_b.topic_path)


# ---------------------------------------------------------------------------
# SECTION 3 - WITH CHECK VIOLATIONS
# ---------------------------------------------------------------------------


async def test_insert_into_entries_with_foreign_user_id_rejected(
    app_pool: asyncpg.Pool,
    seeded_a: TenantSeed,
    tenant_b: UUID,
) -> None:
    """Scoped as A, INSERT INTO entries with user_id=B is rejected by WITH CHECK."""
    now = datetime.now(UTC)
    today = date.today()
    # Encrypted shape post-0008; plaintext "cross-tenant pwn attempt" is encrypted
    # with the seed cipher so the row is syntactically valid -- the RLS WITH CHECK
    # clause is what should reject it, not a missing-column error.
    from tests.fixtures.tenants import _SEED_CIPHER  # noqa: PLC0415

    ct, nonce = _SEED_CIPHER.encrypt("cross-tenant pwn attempt")
    async with user_scoped_connection(app_pool, user_id=seeded_a.user_id) as conn:
        with pytest.raises(asyncpg.PostgresError, match=_RLS_ERROR_RE):
            await conn.execute(
                """
                INSERT INTO entries
                    (topic_id, user_id, date,
                     content_encrypted, content_nonce, search_vector,
                     tags, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, to_tsvector('english', $6), $7, $8, $8)
                """,
                seeded_a.topic_id,
                tenant_b,
                today,
                ct,
                nonce,
                "cross-tenant pwn attempt",
                [],
                now,
            )


async def test_insert_into_topics_with_foreign_user_id_rejected(
    app_pool: asyncpg.Pool,
    tenant_a: UUID,
    tenant_b: UUID,
) -> None:
    """Scoped as A, INSERT INTO topics with user_id=B is rejected by WITH CHECK."""
    now = datetime.now(UTC)
    async with user_scoped_connection(app_pool, user_id=tenant_a) as conn:
        with pytest.raises(asyncpg.PostgresError, match=_RLS_ERROR_RE):
            await conn.execute(
                """
                INSERT INTO topics (path, title, description, user_id, created_at, updated_at)
                VALUES ($1, $2, '', $3, $4, $4)
                """,
                "attacker/topic",
                "Attacker topic",
                tenant_b,
                now,
            )


async def test_update_own_entry_to_foreign_user_id_rejected(
    app_pool: asyncpg.Pool,
    seeded_a: TenantSeed,
    tenant_b: UUID,
) -> None:
    """Scoped as A, UPDATE setting user_id=B on A's own entry fails WITH CHECK."""
    async with user_scoped_connection(app_pool, user_id=seeded_a.user_id) as conn:
        with pytest.raises(asyncpg.PostgresError, match=_RLS_ERROR_RE):
            await conn.execute(
                "UPDATE entries SET user_id = $1 WHERE id = $2",
                tenant_b,
                seeded_a.entry_ids[0],
            )


# ---------------------------------------------------------------------------
# SECTION 4 - UPDATE / DELETE CROSS-TENANT
# ---------------------------------------------------------------------------


async def test_delete_cross_tenant_affects_zero_rows(
    admin_pool: asyncpg.Pool,
    app_pool: asyncpg.Pool,
    seeded_a: TenantSeed,
    seeded_b: TenantSeed,
) -> None:
    """DELETE targeting B's entry_id, scoped as A, affects zero rows."""
    b_entry_id = seeded_b.entry_ids[0]
    async with user_scoped_connection(app_pool, user_id=seeded_a.user_id) as conn:
        status = await conn.execute("DELETE FROM entries WHERE id = $1", b_entry_id)
    assert status.endswith(" 0"), f"expected trailing ' 0' (status={status!r})"

    # Confirm B's row is still on disk via admin_pool (BYPASSRLS).
    async with admin_pool.acquire() as conn:
        still_present = await conn.fetchval(
            "SELECT 1 FROM entries WHERE id = $1 AND deleted_at IS NULL", b_entry_id
        )
    assert still_present == 1


async def test_update_cross_tenant_affects_zero_rows(
    admin_pool: asyncpg.Pool,
    app_pool: asyncpg.Pool,
    seeded_a: TenantSeed,
    seeded_b: TenantSeed,
) -> None:
    """UPDATE targeting B's entry_id, scoped as A, affects zero rows."""
    b_entry_id = seeded_b.entry_ids[0]
    async with user_scoped_connection(app_pool, user_id=seeded_a.user_id) as conn:
        status = await conn.execute(
            "UPDATE entries SET tags = ARRAY['pwned']::text[] WHERE id = $1", b_entry_id
        )
    assert status.endswith(" 0"), f"expected trailing ' 0' (status={status!r})"

    # Row should be untouched.
    async with admin_pool.acquire() as conn:
        tags = await conn.fetchval("SELECT tags FROM entries WHERE id = $1", b_entry_id)
    assert "pwned" not in set(tags or [])


# ---------------------------------------------------------------------------
# SECTION 5 - DEFAULT DENY / SANITY
# ---------------------------------------------------------------------------


async def test_app_pool_without_scope_sees_zero_rows(
    admin_pool: asyncpg.Pool,
    app_pool: asyncpg.Pool,
    tenant_a: UUID,
    tenant_b: UUID,
) -> None:
    """Without user_scoped_connection, app_pool sees zero rows (GUC unset -> user_id=NULL match)."""
    await seed_for(admin_pool, tenant_a, topic_path="tenant-a/default-deny")
    await seed_for(admin_pool, tenant_b, topic_path="tenant-b/default-deny")

    async with app_pool.acquire() as conn:
        entry_count = await conn.fetchval("SELECT COUNT(*) FROM entries")
        topic_count = await conn.fetchval("SELECT COUNT(*) FROM topics")
    assert entry_count == 0
    assert topic_count == 0


async def test_admin_pool_sees_everything(
    admin_pool: asyncpg.Pool,
    tenant_a: UUID,
    tenant_b: UUID,
) -> None:
    """admin_pool has BYPASSRLS so a bare SELECT returns both tenants' rows combined."""
    await seed_for(admin_pool, tenant_a, topic_path="tenant-a/admin-sees", n_entries=3)
    await seed_for(admin_pool, tenant_b, topic_path="tenant-b/admin-sees", n_entries=3)

    async with admin_pool.acquire() as conn:
        entry_count = await conn.fetchval("SELECT COUNT(*) FROM entries")
        topic_count = await conn.fetchval("SELECT COUNT(*) FROM topics")
    assert entry_count == 6  # noqa: PLR2004 — 3 entries per tenant x 2 tenants
    assert topic_count == 2  # noqa: PLR2004


# ---------------------------------------------------------------------------
# SECTION 6 - MIGRATION INVARIANTS (regression guard)
# ---------------------------------------------------------------------------


async def test_force_rls_is_enabled_on_every_tenant_table(admin_pool: asyncpg.Pool) -> None:
    """Both ``relrowsecurity`` AND ``relforcerowsecurity`` must be true on all 5 tables.

    If someone accidentally runs ``ALTER TABLE ... NO FORCE ROW LEVEL SECURITY``
    or ``DISABLE ROW LEVEL SECURITY`` via a stray migration or hotfix, every
    tenant's journal leaks to BYPASSRLS role owners. This test is the guard.
    """
    tenant_tables = ("topics", "entries", "conversations", "messages", "entry_embeddings")
    async with admin_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT relname, relrowsecurity, relforcerowsecurity
            FROM pg_class
            WHERE relname = ANY($1::text[])
              AND relnamespace = 'public'::regnamespace
            """,
            list(tenant_tables),
        )
    seen = {r["relname"]: (r["relrowsecurity"], r["relforcerowsecurity"]) for r in rows}
    assert set(seen.keys()) == set(
        tenant_tables
    ), f"missing tables: {set(tenant_tables) - set(seen)}"
    for table, (row_sec, force_sec) in seen.items():
        assert row_sec, f"ENABLE ROW LEVEL SECURITY missing on {table}"
        assert force_sec, f"FORCE ROW LEVEL SECURITY missing on {table}"


# ---------------------------------------------------------------------------
# SECTION 7 - entry_embeddings + HNSW under RLS (closes 02.05 follow-up)
# ---------------------------------------------------------------------------


async def test_embeddings_visible_only_to_owner(
    app_pool: asyncpg.Pool,
    seeded_a: TenantSeed,
    seeded_b: TenantSeed,
) -> None:
    """Scoped to A, entry_embeddings count reflects only A's seeded rows."""
    async with user_scoped_connection(app_pool, user_id=seeded_a.user_id) as conn:
        count_a = await conn.fetchval("SELECT COUNT(*) FROM entry_embeddings")
    assert count_a == len(seeded_a.entry_ids)

    async with user_scoped_connection(app_pool, user_id=seeded_b.user_id) as conn:
        count_b = await conn.fetchval("SELECT COUNT(*) FROM entry_embeddings")
    assert count_b == len(seeded_b.entry_ids)


async def test_vector_search_cross_tenant_returns_only_own_rows(
    app_pool: asyncpg.Pool,
    seeded_a: TenantSeed,
    seeded_b: TenantSeed,
) -> None:
    """HNSW-ranked query under user_scoped_connection (ef_search=100) filters via RLS post-scan.

    This exercises the exact concern surfaced in the 02.05 DB review: HNSW
    returns candidates without user_id awareness, then RLS drops everything
    not owned by app.current_user_id. Confirm no B rows leak into A's results.
    """
    # Use a non-zero unit vector so cosine distance is well-defined (zero vectors
    # would produce NaN). pgvector codec is registered on the pool, so a plain
    # Python list is accepted as the bound parameter — no SQL cast needed.
    query_vec = [1.0] + [0.0] * 383
    async with user_scoped_connection(app_pool, user_id=seeded_a.user_id) as conn:
        rows = await conn.fetch(
            "SELECT entry_id FROM entry_embeddings ORDER BY embedding <=> $1 LIMIT 10",
            query_vec,
        )
    result_ids = {r["entry_id"] for r in rows}
    assert result_ids == set(seeded_a.entry_ids)
    assert result_ids.isdisjoint(set(seeded_b.entry_ids))


async def test_messages_visible_only_to_owner(
    app_pool: asyncpg.Pool,
    seeded_a: TenantSeed,
    seeded_b: TenantSeed,
) -> None:
    """messages table — a direct scoped query confirms RLS hides B's messages from A."""
    async with user_scoped_connection(app_pool, user_id=seeded_a.user_id) as conn:
        ids = {r["id"] for r in await conn.fetch("SELECT id FROM messages")}
    assert ids == set(seeded_a.message_ids)
    assert ids.isdisjoint(set(seeded_b.message_ids))
