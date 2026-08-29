"""Integration tests for ``entry_repo.read`` pagination correctness.

Pins the ``journal_read_topic`` tool's documented contract:

    "Returns entries in reverse-chronological order (newest first)"
    "offset: Skip the N most-recent entries for pagination"

Each page must be a NEWEST-FIRST window, and pages stitched together
must reconstruct the full series with no gaps and no overlap.

A mock-based repo test cannot catch the bug these tests guard against
-- the failure mode is in the live ``ORDER BY ... LIMIT ... OFFSET``
plan -- so we exercise a real Postgres via the ``app_pool`` /
``admin_pool`` / ``cipher`` / ``tenant_a`` fixtures.

The seed deliberately uses ONE topic, ONE date, and ONE shared
``created_at`` for every row. That collapses the lead two ORDER BY
keys into ties and forces ``id DESC`` to be the only column that
disambiguates rows -- so OFFSET pagination is non-deterministic
unless the SQL carries an ``id DESC`` total-order tie-break.

Docker-backed: skipped automatically when Postgres is not reachable.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from uuid import UUID

import asyncpg
import pytest
from gubbi_common.db.user_scoped import user_scoped_connection

from gubbi.crypto.cipher import ContentCipher
from gubbi.storage.repositories import entries as entry_repo

# Session-scoped asyncpg pools require tests to run in the pools' event loop.
pytestmark = pytest.mark.asyncio(loop_scope="session")


N_ENTRIES = 25
PAGE_SIZE = 10


async def _seed_topic(admin_pool: asyncpg.Pool, user_id: UUID, topic_path: str) -> int:
    """Insert a topic owned by ``user_id``. Returns ``topic_id``."""
    now = datetime.now(UTC)
    async with admin_pool.acquire() as conn:
        topic_id = await conn.fetchval(
            """
            INSERT INTO topics (path, title, description, user_id, created_at, updated_at)
            VALUES ($1, 'Pagination test', '', $2, $3, $3)
            RETURNING id
            """,
            topic_path,
            user_id,
            now,
        )
    return int(topic_id)


async def _seed_entries_same_date_and_created_at(
    admin_pool: asyncpg.Pool,
    cipher: ContentCipher,
    user_id: UUID,
    topic_id: int,
    n: int,
) -> list[int]:
    """Insert ``n`` entries sharing one ``date`` AND one ``created_at``.

    Single connection, single transaction, single ``now`` value -- so
    every row truly ties on the lead two ORDER BY keys. Returns the
    inserted entry ids in INSERT order (which is the order they were
    handed out by the ``entries_id_seq`` serial). The caller can read
    "newest" as the tail of this list (highest id) and "oldest" as the
    head (lowest id).
    """
    fixed_now = datetime.now(UTC)
    fixed_date = date(2026, 1, 15)
    ids: list[int] = []
    async with admin_pool.acquire() as conn, conn.transaction():
        for i in range(n):
            content = f"entry-{i:02d} body"
            content_ct, content_nonce = cipher.encrypt(content)
            entry_id = await conn.fetchval(
                """
                INSERT INTO entries
                    (topic_id, user_id, date,
                     content_encrypted, content_nonce,
                     reasoning_encrypted, reasoning_nonce,
                     search_vector, tags, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, NULL, NULL,
                        to_tsvector('english', $6), '{}', $7, $7)
                RETURNING id
                """,
                topic_id,
                user_id,
                fixed_date,
                content_ct,
                content_nonce,
                content,
                fixed_now,
            )
            ids.append(int(entry_id))
    return ids


async def test_pagination_yields_complete_disjoint_newest_first_pages(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    cipher: ContentCipher,
    tenant_a: UUID,
) -> None:
    """Three pages of size 10 over 25 same-date entries cover every id once.

    Asserts the four invariants any well-behaved offset pagination
    must satisfy:

    (a) the union of all pages is EXACTLY the seeded id set (complete);
    (b) the pages are pairwise DISJOINT (no row served twice);
    (c) concatenated pages are non-INCREASING by id (newest-first
        across pages -- highest id first, walking back to lowest);
    (d) ``total`` is the full window count on every page.

    With same-date / same-created_at rows, only an ``id DESC``
    tie-break in the SQL ORDER BY can hold these invariants -- the
    test falsifies any plan that anchors page 0 at a different end of
    the sequence than later pages, or that lets row order drift
    between calls.
    """
    # Arrange.
    topic_path = "pagination/all-same-date"
    topic_id = await _seed_topic(admin_pool, tenant_a, topic_path)
    seeded_ids = await _seed_entries_same_date_and_created_at(
        admin_pool, cipher, tenant_a, topic_id, N_ENTRIES
    )
    assert len(seeded_ids) == N_ENTRIES
    assert len(set(seeded_ids)) == N_ENTRIES, "seed produced duplicate ids -- bug in seed helper"

    # Act -- page 0/10/20.
    pages: list[list[int]] = []
    totals: list[int] = []
    async with user_scoped_connection(app_pool, tenant_a) as conn:
        for page_offset in (0, PAGE_SIZE, 2 * PAGE_SIZE):
            _meta, entries, total = await entry_repo.read(
                conn,
                cipher,
                topic_path,
                limit=PAGE_SIZE,
                offset=page_offset,
            )
            pages.append([e.id for e in entries])
            totals.append(total)

    # Assert (d): total reports the full window on every page.
    assert totals == [N_ENTRIES, N_ENTRIES, N_ENTRIES]

    # Page-size invariants: 10 + 10 + 5.
    assert [len(p) for p in pages] == [PAGE_SIZE, PAGE_SIZE, N_ENTRIES - 2 * PAGE_SIZE]

    # Assert (a): union of pages == seeded ids (no row missing).
    union = {eid for page in pages for eid in page}
    assert union == set(seeded_ids), (
        f"pages do not cover the seeded ids: missing={set(seeded_ids) - union}, "
        f"extra={union - set(seeded_ids)}"
    )

    # Assert (b): pages are pairwise disjoint (no row served twice).
    page0, page1, page2 = pages
    assert not (set(page0) & set(page1)), f"page 0 and page 1 overlap: {set(page0) & set(page1)}"
    assert not (set(page0) & set(page2)), f"page 0 and page 2 overlap: {set(page0) & set(page2)}"
    assert not (set(page1) & set(page2)), f"page 1 and page 2 overlap: {set(page1) & set(page2)}"

    # Assert (c): concatenated pages are non-INCREASING by id -- newest-first
    # stream. Equivalent to "sorted in reverse on id".
    concatenated = page0 + page1 + page2
    assert concatenated == sorted(concatenated, reverse=True), (
        f"concatenated pages are not in non-increasing (newest-first) order: {concatenated}"
    )


async def test_offset_past_end_reports_full_total_not_zero(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    cipher: ContentCipher,
    tenant_a: UUID,
) -> None:
    """A page whose offset is past the end still reports the full filtered total.

    Regression guard for the pagination contract: ``total`` is the full
    filtered count, independent of the requested page. The pre-fix code read
    ``total`` from ``COUNT(*) OVER()`` on ``rows[0]`` and fell back to 0 when the
    page was empty -- so offset-past-end pages wrongly reported ``total=0`` even
    though rows exist. ``list_entries`` now runs a fallback COUNT on the empty
    page.
    """
    # Arrange -- seed N entries, then request a page well past the end.
    topic_path = "pagination/offset-past-end"
    topic_id = await _seed_topic(admin_pool, tenant_a, topic_path)
    seeded_ids = await _seed_entries_same_date_and_created_at(
        admin_pool, cipher, tenant_a, topic_id, N_ENTRIES
    )

    # Act -- offset beyond the last row.
    async with user_scoped_connection(app_pool, tenant_a) as conn:
        rows, total = await entry_repo.list_entries(
            conn,
            topic=topic_path,
            limit=PAGE_SIZE,
            offset=N_ENTRIES + PAGE_SIZE,
        )

    # Assert -- empty page, but total reflects the full filtered set.
    assert rows == []
    assert total == len(seeded_ids) == N_ENTRIES


async def test_offset_zero_page_returns_newest_entries_not_oldest(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    cipher: ContentCipher,
    tenant_a: UUID,
) -> None:
    """Page 0 must be the NEWEST PAGE_SIZE entries, not the oldest.

    Regression guard for the journal's reverse-chronological contract:
    a journal's read view shows today first and walks backward into
    history. The pre-fix implementation had two divergent code paths
    whose ordering was anchored at OPPOSITE ends of the sequence (the
    offset==0 branch ordered DESC then ``reversed()``, the offset>0
    branch ordered ASC), so paging limit/offset double-returned some
    rows and stranded others. This guard pins page 0 to the newest
    end of the stream and the in-page ordering to newest-first.
    """
    # Arrange.
    topic_path = "pagination/newest-first-guard"
    topic_id = await _seed_topic(admin_pool, tenant_a, topic_path)
    seeded_ids = await _seed_entries_same_date_and_created_at(
        admin_pool, cipher, tenant_a, topic_id, N_ENTRIES
    )

    # Act.
    async with user_scoped_connection(app_pool, tenant_a) as conn:
        _meta, entries, _total = await entry_repo.read(
            conn,
            cipher,
            topic_path,
            limit=PAGE_SIZE,
            offset=0,
        )

    # Assert: page 0 is the NEWEST PAGE_SIZE ids (the tail slice of the
    # seeded sequence in DESC order), NOT the oldest PAGE_SIZE ids.
    page_ids = [e.id for e in entries]
    expected_newest = sorted(seeded_ids, reverse=True)[:PAGE_SIZE]
    assert page_ids == expected_newest, (
        f"offset=0 page must be the newest {PAGE_SIZE} ids ({expected_newest}); got {page_ids}"
    )

    # And the page contents must match the newest contents in DESC
    # order -- belt-and-braces: id-only assertion would still pass if
    # a future change reshuffles ids vs. content.
    # ``entry-NN body`` index N runs 0..24 in INSERT order; the newest
    # PAGE_SIZE rows are the ones with the highest indices, returned in
    # descending order.
    contents = [e.content for e in entries]
    expected_contents = [
        f"entry-{i:02d} body" for i in range(N_ENTRIES - 1, N_ENTRIES - 1 - PAGE_SIZE, -1)
    ]
    assert contents == expected_contents
