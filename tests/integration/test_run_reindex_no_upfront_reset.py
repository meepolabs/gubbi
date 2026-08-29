"""``_run_reindex`` no longer resets already-indexed rows.

The pre-fix shape called ``reset_indexed_at`` upfront -- every reindex
implicitly wiped ``indexed_at`` on every row. Under the new claim-then-
process semantics (``FOR UPDATE SKIP LOCKED`` + per-batch claim), that
upfront reset became an active hazard: any row a concurrent worker had
just stamped would be re-cleared and re-claimed, producing duplicate
encodes.

This test seeds a mix of indexed + unindexed rows and asserts that
``_run_reindex`` only touches the unindexed half. The wipe-and-re-embed
path is now an explicit admin action (``_reset_all_indexed_at``).
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock
from uuid import UUID

import asyncpg
import pytest
import pytest_asyncio

from gubbi.app_context import AppContext
from gubbi.crypto.cipher import ContentCipher
from gubbi.tools.admin import _reset_all_indexed_at, _run_reindex
from tests.fixtures.tenants import seed_for

pytestmark = pytest.mark.asyncio(loop_scope="session")


class _CountingEmbeddingService:
    """Minimal stand-in for ``EmbeddingService`` recording every encode call."""

    def __init__(self) -> None:
        self.encode_calls: list[str] = []
        self.save_calls: list[int] = []
        self._lock = asyncio.Lock()

    def encode(self, text: str) -> list[float]:
        self.encode_calls.append(text)
        return [1.0] + [0.0] * 383

    async def save_by_vector(
        self,
        conn: asyncpg.Connection,
        entry_id: int,
        embedding: list[float],
    ) -> None:
        async with self._lock:
            self.save_calls.append(entry_id)
        await conn.execute(
            """
            INSERT INTO entry_embeddings (entry_id, user_id, embedding, indexed_at)
            SELECT $1, e.user_id, $2, now()
            FROM entries e WHERE e.id = $1
            ON CONFLICT (entry_id) DO UPDATE SET embedding = EXCLUDED.embedding,
                                                 indexed_at = EXCLUDED.indexed_at
            """,
            entry_id,
            embedding,
        )


def _make_app_ctx(embedding_service: Any, admin_pool: asyncpg.Pool) -> AppContext:
    app_ctx = MagicMock(spec=AppContext)
    app_ctx.embedding_service = embedding_service
    app_ctx.admin_pool = admin_pool
    return app_ctx


@pytest_asyncio.fixture
async def mixed_indexed_entries(
    admin_pool: asyncpg.Pool,
    tenant_a: UUID,
) -> tuple[UUID, list[int], list[int]]:
    """Seed entries split into already-indexed and unindexed halves.

    Returns ``(user_id, indexed_ids, unindexed_ids)``. ``indexed_ids``
    are stamped with a fixed ``indexed_at = now()``; ``unindexed_ids``
    have ``indexed_at = NULL``.
    """
    seed = await seed_for(admin_pool, tenant_a, topic_path="reindex-x3/notes", n_entries=12)
    all_ids = list(seed.entry_ids)
    indexed_ids = all_ids[:6]
    unindexed_ids = all_ids[6:]

    async with admin_pool.acquire() as conn:
        # Drop fixture-planted embeddings so the encode pipeline has work.
        await conn.execute("DELETE FROM entry_embeddings WHERE user_id = $1", tenant_a)
        # Wipe indexed_at across the seed, then stamp the indexed half explicitly.
        await conn.execute("UPDATE entries SET indexed_at = NULL WHERE user_id = $1", tenant_a)
        await conn.execute(
            "UPDATE entries SET indexed_at = now() WHERE id = ANY($1)",
            indexed_ids,
        )

    return tenant_a, indexed_ids, unindexed_ids


async def test_run_reindex_does_not_reset_already_indexed_rows(
    admin_pool: asyncpg.Pool,
    cipher: ContentCipher,
    mixed_indexed_entries: tuple[UUID, list[int], list[int]],
) -> None:
    """``_run_reindex`` must NOT wipe ``indexed_at`` on previously-indexed rows.

    Pre-fix behaviour: the function called ``reset_indexed_at`` at the
    head, clearing every non-deleted row's stamp. Post-fix: the queue
    is "rows where indexed_at IS NULL" and the indexed rows stay put.
    """
    _tenant_id, indexed_ids, unindexed_ids = mixed_indexed_entries

    # Capture the pre-run indexed_at stamps for the indexed half so we can
    # assert they are preserved (not just non-NULL but not re-stamped).
    async with admin_pool.acquire() as conn:
        before_rows = await conn.fetch(
            "SELECT id, indexed_at FROM entries WHERE id = ANY($1) ORDER BY id",
            indexed_ids,
        )
    before_stamps: dict[int, datetime] = {r["id"]: r["indexed_at"] for r in before_rows}
    assert all(ts is not None for ts in before_stamps.values()), (
        "Fixture invariant: indexed half must have non-NULL indexed_at before run"
    )

    svc = _CountingEmbeddingService()
    ctx = _make_app_ctx(svc, admin_pool)

    result = await _run_reindex(ctx, admin_pool, cipher)

    # Only the unindexed half should have been encoded.
    assert sorted(svc.save_calls) == sorted(unindexed_ids), (
        f"Reindex must process only unindexed rows; saved={sorted(svc.save_calls)}, "
        f"expected={sorted(unindexed_ids)}"
    )
    assert result["embeddings_generated"] == len(unindexed_ids)
    assert result["embeddings_failed"] == 0

    # The indexed half's stamps must be preserved exactly -- NOT reset to
    # NULL, NOT re-stamped to now() (which would be the symptom of the
    # old upfront-reset bug racing with mark_indexed_batch).
    async with admin_pool.acquire() as conn:
        after_rows = await conn.fetch(
            "SELECT id, indexed_at FROM entries WHERE id = ANY($1) ORDER BY id",
            indexed_ids,
        )
    after_stamps = {r["id"]: r["indexed_at"] for r in after_rows}
    for entry_id in indexed_ids:
        assert after_stamps[entry_id] is not None, (
            f"entry {entry_id} indexed_at was reset to NULL -- upfront reset regression"
        )
        # Bit-for-bit equality: the row must not have been touched.
        assert after_stamps[entry_id] == before_stamps[entry_id], (
            f"entry {entry_id} indexed_at changed: "
            f"before={before_stamps[entry_id]!r}, after={after_stamps[entry_id]!r}"
        )


async def test_reset_all_indexed_at_clears_every_indexed_row(
    admin_pool: asyncpg.Pool,
    mixed_indexed_entries: tuple[UUID, list[int], list[int]],
) -> None:
    """``_reset_all_indexed_at`` is the explicit wipe-and-re-embed admin action.

    Mirrors the behaviour ``_run_reindex`` used to do implicitly: clears
    ``indexed_at`` on every non-deleted row so a follow-up reindex
    re-embeds everything. Returns ``rows_reset`` so the caller can audit
    the impact.
    """
    tenant_id, indexed_ids, unindexed_ids = mixed_indexed_entries

    ctx = _make_app_ctx(_CountingEmbeddingService(), admin_pool)
    result = await _reset_all_indexed_at(admin_pool, ctx)

    assert result["status_locked"] == 0, "advisory lock must be available"
    # Only the indexed half had a non-NULL stamp before the reset.
    assert result["rows_reset"] == len(indexed_ids), (
        f"rows_reset mismatch: got {result['rows_reset']}, expected {len(indexed_ids)}"
    )

    async with admin_pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, indexed_at FROM entries WHERE user_id = $1 ORDER BY id",
            tenant_id,
        )
    by_id = {r["id"]: r["indexed_at"] for r in rows}
    for entry_id in indexed_ids + unindexed_ids:
        assert by_id[entry_id] is None, (
            f"entry {entry_id} still has indexed_at after _reset_all_indexed_at"
        )


async def test_reset_all_indexed_at_releases_advisory_lock(
    admin_pool: asyncpg.Pool,
    mixed_indexed_entries: tuple[UUID, list[int], list[int]],
) -> None:
    """The advisory lock released after the wipe so the next caller can acquire it."""
    _ = mixed_indexed_entries  # only needed for the seeded fixture state.

    ctx = _make_app_ctx(_CountingEmbeddingService(), admin_pool)

    # First call: should succeed and release the lock.
    first = await _reset_all_indexed_at(admin_pool, ctx)
    assert first["status_locked"] == 0

    # Second call (back-to-back): should ALSO succeed -- if the lock had
    # leaked, this would return ``status_locked == 1``.
    second = await _reset_all_indexed_at(admin_pool, ctx)
    assert second["status_locked"] == 0
    # Second call has nothing to reset (first call cleared everything).
    assert second["rows_reset"] == 0


async def test_reset_all_indexed_at_skips_when_lock_held(
    admin_pool: asyncpg.Pool,
    mixed_indexed_entries: tuple[UUID, list[int], list[int]],
) -> None:
    """If a concurrent process already holds the advisory lock, the call no-ops.

    Acquire the lock from a second connection, then call
    ``_reset_all_indexed_at``. The function must report
    ``status_locked = 1`` and leave ``indexed_at`` untouched.
    """
    _tenant_id, indexed_ids, _unindexed_ids = mixed_indexed_entries
    lock_key = 2048976971  # mirrors _REINDEX_ADVISORY_LOCK_KEY -- private to admin.py

    ctx = _make_app_ctx(_CountingEmbeddingService(), admin_pool)

    blocker = await admin_pool.acquire()
    try:
        got = await blocker.fetchval("SELECT pg_try_advisory_lock($1)", lock_key)
        assert got is True, "blocker connection failed to acquire advisory lock"

        result = await _reset_all_indexed_at(admin_pool, ctx)
        assert result["status_locked"] == 1
        assert result["rows_reset"] == 0

        # The indexed half must STILL be stamped -- the no-op must not have
        # leaked through.
        async with admin_pool.acquire() as conn:
            still_stamped = await conn.fetchval(
                "SELECT count(*) FROM entries WHERE id = ANY($1) AND indexed_at IS NOT NULL",
                indexed_ids,
            )
        assert still_stamped == len(indexed_ids)
    finally:
        # Release the blocker's lock so the test fixture can clean up.
        await blocker.execute("SELECT pg_advisory_unlock($1)", lock_key)
        await admin_pool.release(blocker)
