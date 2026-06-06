"""GIN index on ``messages.search_vector`` end-to-end.

Migration ``0025_messages_search_vector_gin`` adds
``idx_messages_fts`` so FTS scans against ``messages`` use the same
plan shape that ``entries`` and ``conversations`` already enjoy.

This test exercises the round-trip:
  1. Seed a message row with a known plaintext via the tenant fixture
     (``seed_for`` populates ``search_vector`` via
     ``to_tsvector('english', $1)`` in the same shape that
     ``append_messages`` uses in production).
  2. Issue a ``tsquery`` filter and assert the row comes back.
  3. ``EXPLAIN`` the same query and assert the plan picks up the
     GIN index (``idx_messages_fts``).

The EXPLAIN assertion is the load-bearing one -- it pins the index
to the FTS scan and would regress silently if a future migration
dropped the index without restoring it.
"""

from __future__ import annotations

from uuid import UUID

import asyncpg
import pytest

from tests.fixtures.tenants import seed_for

pytestmark = pytest.mark.asyncio(loop_scope="session")


async def test_messages_search_vector_uses_gin_index(
    admin_pool: asyncpg.Pool,
    tenant_a: UUID,
    clean_rls_db: asyncpg.Pool,
) -> None:
    """``to_tsquery`` against ``messages.search_vector`` plans against ``idx_messages_fts``."""
    seed = await seed_for(
        admin_pool,
        tenant_a,
        topic_path="messages-fts/notes",
        n_entries=0,
        n_messages=4,
    )
    assert len(seed.message_ids) == 4

    # The seed encodes the literal string ``"Message {i} content"`` into
    # ``search_vector`` via to_tsvector('english', ...).  After 'english'
    # tokenisation, ``Message`` -> ``messag`` -- query via ``plainto_tsquery``
    # so we match the stemmed tokens without hard-coding stem rules.
    async with admin_pool.acquire() as conn:
        # 1. The query returns the seeded rows.
        hits = await conn.fetch(
            """
            SELECT id FROM messages
            WHERE search_vector @@ plainto_tsquery('english', $1)
            """,
            "Message content",
        )
        assert len(hits) == len(seed.message_ids), (
            "FTS query must return every seeded message; "
            f"expected {len(seed.message_ids)}, got {len(hits)}"
        )

        # 2. The planner uses the GIN index.  With only a handful of seed
        # rows the planner often prefers a sequential scan because the
        # table is below ``seq_page_cost`` thresholds; force index usage
        # for the EXPLAIN so we are asserting on the index's existence
        # and applicability, not on the planner's cost model for tiny
        # test tables.  Production volumes will pick the index on their
        # own.  Reset the GUC after the EXPLAIN so unrelated assertions
        # in later test phases are not affected (``SET LOCAL`` would be
        # cleaner but EXPLAIN cannot run inside a multi-statement
        # transaction here without changing the fixture shape).
        await conn.execute("SET enable_seqscan = off")
        try:
            plan_rows = await conn.fetch(
                """
                EXPLAIN
                SELECT id FROM messages
                WHERE search_vector @@ plainto_tsquery('english', $1)
                """,
                "Message content",
            )
        finally:
            await conn.execute("SET enable_seqscan = on")

        plan_text = "\n".join(row[0] for row in plan_rows)
        assert "idx_messages_fts" in plan_text, (
            "Plan must reference idx_messages_fts (GIN index created by "
            f"migration 0025); got:\n{plan_text}"
        )
        # Sanity: bitmap-index-scan is the typical access for GIN.
        assert "Bitmap Index Scan" in plan_text or "Index Scan" in plan_text, (
            "Plan must use an index-scan node (Bitmap Index Scan typical "
            f"for GIN); got:\n{plan_text}"
        )


async def test_idx_messages_fts_exists(
    admin_pool: asyncpg.Pool,
    clean_rls_db: asyncpg.Pool,
) -> None:
    """``idx_messages_fts`` is present in ``pg_indexes`` after migration."""
    async with admin_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT indexdef FROM pg_indexes "
            "WHERE schemaname = 'public' AND indexname = 'idx_messages_fts'",
        )
    assert row is not None, "idx_messages_fts missing -- migration 0025 not applied"
    indexdef = row["indexdef"]
    assert "using gin" in indexdef.lower(), f"idx_messages_fts must be a GIN index; got: {indexdef}"
    assert (
        "search_vector" in indexdef
    ), f"idx_messages_fts must cover the search_vector column; got: {indexdef}"
