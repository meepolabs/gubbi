"""Integration tests for ``list_all`` / ``list_conversations`` pagination.

Sibling tests of ``test_entries_pagination.py`` -- same bug class, same
fixture style, two repositories. Both queries paginate by a non-unique
timestamp DESC; without an ``id DESC`` total-order tie-break, OFFSET
pagination is non-deterministic for tied rows: Postgres is free to
reorder ties between calls, so the same row can land in two pages or
be skipped entirely.

The seed deliberately gives every row in a window an IDENTICAL
timestamp so the lead ORDER BY key always ties; the only column that
can disambiguate is the PK. The (c) assertion -- concatenated pages
match ``sorted(seeded_ids, reverse=True)`` -- is the reliable RED
signal: any non-deterministic plan will violate it.

Docker-backed: skipped automatically when Postgres is not reachable.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

import asyncpg
import pytest
from gubbi_common.db.user_scoped import user_scoped_connection

from gubbi.crypto.cipher import ContentCipher
from gubbi.storage.repositories import conversations as conv_repo
from gubbi.storage.repositories import topics as topic_repo

# Session-scoped asyncpg pools require tests to run in the pools' event loop.
pytestmark = pytest.mark.asyncio(loop_scope="session")


N_ROWS = 15
PAGE_SIZE = 5


async def _seed_topics_same_updated_at(
    admin_pool: asyncpg.Pool,
    user_id: UUID,
    n: int,
) -> list[int]:
    """Insert ``n`` topics for ``user_id`` sharing one ``updated_at``.

    Single connection + single transaction + single ``now`` value, so
    every row truly ties on the lead ORDER BY key in ``list_all``.
    Distinct paths -> distinct serial ids; returns the ids in INSERT
    order.
    """
    fixed_now = datetime.now(UTC)
    ids: list[int] = []
    async with admin_pool.acquire() as conn, conn.transaction():
        for i in range(n):
            topic_id = await conn.fetchval(
                """
                INSERT INTO topics
                    (path, title, description, user_id, created_at, updated_at)
                VALUES ($1, $2, '', $3, $4, $4)
                RETURNING id
                """,
                f"pagination-topics/topic-{i:02d}",
                f"Topic {i:02d}",
                user_id,
                fixed_now,
            )
            ids.append(int(topic_id))
    return ids


async def _seed_conversations_same_created_at(
    admin_pool: asyncpg.Pool,
    cipher: ContentCipher,
    user_id: UUID,
    topic_id: int,
    n: int,
) -> list[int]:
    """Insert ``n`` conversations under ``topic_id`` sharing one ``created_at``.

    Mirrors the conversation insert in ``tests/fixtures/tenants.py``;
    distinct slugs + json_paths via uuid4 satisfy the
    (topic_id, slug) unique constraint and the platform-id dedup
    index. Returns the ids in INSERT order.
    """
    fixed_now = datetime.now(UTC)
    ids: list[int] = []
    async with admin_pool.acquire() as conn, conn.transaction():
        for i in range(n):
            title_plain = f"Conversation {i:02d}"
            summary_plain = f"Summary {i:02d}"
            title_ct, title_nonce = cipher.encrypt(title_plain)
            summary_ct, summary_nonce = cipher.encrypt(summary_plain)
            conv_id = await conn.fetchval(
                """
                INSERT INTO conversations
                    (topic_id, user_id, title_encrypted, title_nonce, slug, source,
                     summary_encrypted, summary_nonce, tags,
                     participants, message_count, created_at, updated_at, json_path,
                     search_vector)
                VALUES ($1, $2, $3, $4, $5, 'claude', $6, $7, $8, $9, $10, $11, $11, $12,
                        to_tsvector('english', $13))
                RETURNING id
                """,
                topic_id,
                user_id,
                title_ct,
                title_nonce,
                f"conv-{uuid4().hex[:12]}",
                summary_ct,
                summary_nonce,
                ["seed"],
                ["user", "assistant"],
                0,
                fixed_now,
                f"conversations_json/{uuid4()}.json",
                f"{title_plain} {summary_plain}",
            )
            ids.append(int(conv_id))
    return ids


async def _seed_topic_for_conversations(
    admin_pool: asyncpg.Pool,
    user_id: UUID,
    topic_path: str,
) -> int:
    """Insert one topic owned by ``user_id`` and return its id."""
    now = datetime.now(UTC)
    async with admin_pool.acquire() as conn:
        topic_id = await conn.fetchval(
            """
            INSERT INTO topics (path, title, description, user_id, created_at, updated_at)
            VALUES ($1, 'Convo host', '', $2, $3, $3)
            RETURNING id
            """,
            topic_path,
            user_id,
            now,
        )
    return int(topic_id)


async def test_list_all_pagination_yields_complete_disjoint_id_desc_pages(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    tenant_a: UUID,
) -> None:
    """``list_all`` paginates deterministically when ``updated_at`` ties.

    Seeds N_ROWS topics with identical ``updated_at`` so the lead
    ORDER BY key always ties. Pages through at offsets 0/PAGE_SIZE/
    2*PAGE_SIZE and asserts:

    (a) union of all page ids == seeded id set (no row missed),
    (b) pages pairwise disjoint (no row served twice),
    (c) concatenated pages match ``sorted(seeded_ids, reverse=True)``
        -- the deterministic newest-first tie-break order driven by
        the unique-PK ``id DESC`` tail of the ORDER BY.

    Without ``id DESC`` in the ORDER BY, (c) is the assertion that
    fails: the planner is free to reorder tied rows between calls,
    causing duplicates and gaps that (a) and (b) also catch.
    """
    # Arrange.
    seeded_ids = await _seed_topics_same_updated_at(admin_pool, tenant_a, N_ROWS)
    assert len(seeded_ids) == N_ROWS
    assert len(set(seeded_ids)) == N_ROWS, "seed produced duplicate ids -- bug in seed helper"

    # Act -- page through 0/PAGE_SIZE/2*PAGE_SIZE.
    pages: list[list[int]] = []
    totals: list[int] = []
    async with user_scoped_connection(app_pool, tenant_a) as conn:
        for page_offset in (0, PAGE_SIZE, 2 * PAGE_SIZE):
            metas, total = await topic_repo.list_all(conn, limit=PAGE_SIZE, offset=page_offset)
            pages.append([m.id for m in metas if m.id is not None])
            totals.append(total)

    # Page-size invariants (sanity-check the seed): three full pages.
    assert [len(p) for p in pages] == [PAGE_SIZE, PAGE_SIZE, PAGE_SIZE]
    assert totals == [N_ROWS, N_ROWS, N_ROWS]

    # Assert (a): union of pages == seeded ids.
    union = {tid for page in pages for tid in page}
    assert union == set(seeded_ids), (
        f"pages do not cover the seeded ids: missing={set(seeded_ids) - union}, "
        f"extra={union - set(seeded_ids)}"
    )

    # Assert (b): pairwise disjoint.
    page0, page1, page2 = pages
    assert not (set(page0) & set(page1)), f"page 0/1 overlap: {set(page0) & set(page1)}"
    assert not (set(page0) & set(page2)), f"page 0/2 overlap: {set(page0) & set(page2)}"
    assert not (set(page1) & set(page2)), f"page 1/2 overlap: {set(page1) & set(page2)}"

    # Assert (c): concatenated == seeded ids sorted by id DESC.
    concatenated = page0 + page1 + page2
    expected = sorted(seeded_ids, reverse=True)
    assert concatenated == expected, (
        f"concatenated pages do not match deterministic id-DESC order: "
        f"got {concatenated}, expected {expected}"
    )


async def test_list_conversations_pagination_yields_complete_disjoint_id_desc_pages(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    cipher: ContentCipher,
    tenant_a: UUID,
) -> None:
    """``list_conversations`` paginates deterministically when ``created_at`` ties.

    Seeds N_ROWS conversations with identical ``created_at`` so the
    lead ORDER BY key always ties. Pages through at offsets 0/
    PAGE_SIZE/2*PAGE_SIZE and asserts:

    (a) union of all page ids == seeded id set (no row missed),
    (b) pages pairwise disjoint (no row served twice),
    (c) concatenated pages match ``sorted(seeded_ids, reverse=True)``
        -- the deterministic newest-first tie-break order driven by
        the unique-PK ``id DESC`` tail of the ORDER BY.

    Same RED signal as the ``list_all`` sibling test: (c) falsifies
    any plan that lets tied rows shuffle between calls.
    """
    # Arrange.
    topic_id = await _seed_topic_for_conversations(admin_pool, tenant_a, "pagination-conv/host")
    seeded_ids = await _seed_conversations_same_created_at(
        admin_pool, cipher, tenant_a, topic_id, N_ROWS
    )
    assert len(seeded_ids) == N_ROWS
    assert len(set(seeded_ids)) == N_ROWS, "seed produced duplicate ids -- bug in seed helper"

    # Act -- page through 0/PAGE_SIZE/2*PAGE_SIZE.
    pages: list[list[int]] = []
    totals: list[int] = []
    async with user_scoped_connection(app_pool, tenant_a) as conn:
        for page_offset in (0, PAGE_SIZE, 2 * PAGE_SIZE):
            metas, total = await conv_repo.list_conversations(
                conn, cipher, limit=PAGE_SIZE, offset=page_offset
            )
            pages.append([m.id for m in metas if m.id is not None])
            totals.append(total)

    # Page-size invariants (sanity-check the seed): three full pages.
    assert [len(p) for p in pages] == [PAGE_SIZE, PAGE_SIZE, PAGE_SIZE]
    assert totals == [N_ROWS, N_ROWS, N_ROWS]

    # Assert (a): union of pages == seeded ids.
    union = {cid for page in pages for cid in page}
    assert union == set(seeded_ids), (
        f"pages do not cover the seeded ids: missing={set(seeded_ids) - union}, "
        f"extra={union - set(seeded_ids)}"
    )

    # Assert (b): pairwise disjoint.
    page0, page1, page2 = pages
    assert not (set(page0) & set(page1)), f"page 0/1 overlap: {set(page0) & set(page1)}"
    assert not (set(page0) & set(page2)), f"page 0/2 overlap: {set(page0) & set(page2)}"
    assert not (set(page1) & set(page2)), f"page 1/2 overlap: {set(page1) & set(page2)}"

    # Assert (c): concatenated == seeded ids sorted by id DESC.
    concatenated = page0 + page1 + page2
    expected = sorted(seeded_ids, reverse=True)
    assert concatenated == expected, (
        f"concatenated pages do not match deterministic id-DESC order: "
        f"got {concatenated}, expected {expected}"
    )


async def test_list_all_offset_past_end_reports_full_total(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    tenant_a: UUID,
) -> None:
    """``list_all`` reports the full total for an offset-past-end page, not 0.

    Regression guard: ``total`` is the full filtered count regardless of page.
    The pre-fix code read it from ``COUNT(*) OVER()`` on ``rows[0]`` and fell
    back to 0 on an empty page, so offset-past-end pages wrongly reported
    ``total=0``. ``list_all`` now runs a fallback COUNT on the empty page.
    """
    # Arrange.
    seeded_ids = await _seed_topics_same_updated_at(admin_pool, tenant_a, N_ROWS)

    # Act -- offset past the last row.
    async with user_scoped_connection(app_pool, tenant_a) as conn:
        metas, total = await topic_repo.list_all(conn, limit=PAGE_SIZE, offset=N_ROWS + PAGE_SIZE)

    # Assert.
    assert metas == []
    assert total == len(seeded_ids) == N_ROWS


async def test_list_conversations_offset_past_end_reports_full_total(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    cipher: ContentCipher,
    tenant_a: UUID,
) -> None:
    """``list_conversations`` reports the full total for an offset-past-end page.

    Same regression as the ``list_all`` sibling: an empty (past-end) page must
    still carry the full filtered total via the fallback COUNT.
    """
    # Arrange.
    topic_id = await _seed_topic_for_conversations(
        admin_pool, tenant_a, "pagination-conv-pastend/host"
    )
    seeded_ids = await _seed_conversations_same_created_at(
        admin_pool, cipher, tenant_a, topic_id, N_ROWS
    )

    # Act -- offset past the last row.
    async with user_scoped_connection(app_pool, tenant_a) as conn:
        metas, total = await conv_repo.list_conversations(
            conn, cipher, limit=PAGE_SIZE, offset=N_ROWS + PAGE_SIZE
        )

    # Assert.
    assert metas == []
    assert total == len(seeded_ids) == N_ROWS
