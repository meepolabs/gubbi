"""Reindex ``last_id`` correctness under partial / total batch failure.

Before the fix, ``_run_reindex`` advanced ``last_id = batch[-1]["id"]``
unconditionally at the end of each iteration. After a compensating reset
on a failed entry, the row had ``indexed_at = NULL`` AND ``id <= last_id``;
the next iteration's ``get_unindexed`` query (cursor: ``id > $1``) would
skip it forever. Stranded rows.

Fix: track ``succeeded_ids`` separately; advance ``last_id`` only past
the max of the actually-succeeded ids. If the entire batch failed,
break out of the loop instead of looping forever on the same cursor
(rows still get picked up by a future ``_run_reindex`` invocation since
their ``indexed_at`` is now NULL).
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock
from uuid import UUID

import asyncpg
import pytest
import pytest_asyncio

from gubbi.app_context import AppContext
from gubbi.crypto.cipher import ContentCipher
from gubbi.tools.admin import _run_reindex
from tests.fixtures.tenants import seed_for

pytestmark = pytest.mark.asyncio(loop_scope="session")


class _CountingEmbeddingService:
    """Embedding service stand-in that fails for a configurable id set."""

    def __init__(self, fail_on_ids: set[int] | None = None) -> None:
        self.fail_on_ids: set[int] = fail_on_ids or set()
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
        if entry_id in self.fail_on_ids:
            raise RuntimeError(f"injected save failure for entry {entry_id}")
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
async def seeded_entries_for_cursor(
    admin_pool: asyncpg.Pool,
    tenant_a: UUID,
) -> tuple[UUID, list[int]]:
    """Seed N entries; clear ``indexed_at`` so they all land in the unindexed queue."""
    seed = await seed_for(admin_pool, tenant_a, topic_path="reindex-cursor/notes", n_entries=10)
    async with admin_pool.acquire() as conn:
        await conn.execute("DELETE FROM entry_embeddings WHERE user_id = $1", tenant_a)
        await conn.execute("UPDATE entries SET indexed_at = NULL WHERE user_id = $1", tenant_a)
    return tenant_a, list(seed.entry_ids)


async def test_run_reindex_processes_failed_entries_on_retry(
    admin_pool: asyncpg.Pool,
    cipher: ContentCipher,
    seeded_entries_for_cursor: tuple[UUID, list[int]],
) -> None:
    """A failed mid-batch entry must be retryable on a follow-up ``_run_reindex``.

    Pre-fix: after the compensating reset, ``last_id`` advanced past the
    failed id. The next ``_run_reindex`` (cursor starts at 0) would
    actually pick it up because the cursor is per-call. But within a
    single multi-batch run, an early-batch failure stranded the row
    behind a cursor that never came back. With BATCH_SIZE = 64 (default)
    this is hard to surface in unit tests, so we exercise the
    cross-call retry path here as the durable contract.
    """
    tenant_id, entry_ids = seeded_entries_for_cursor
    failing_id = entry_ids[len(entry_ids) // 2]

    # First pass: failing_id raises, others succeed.
    svc1 = _CountingEmbeddingService(fail_on_ids={failing_id})
    ctx1 = _make_app_ctx(svc1, admin_pool)
    result1 = await _run_reindex(ctx1, admin_pool, cipher)

    assert result1["embeddings_failed"] == 1
    assert sorted(svc1.save_calls) == sorted(entry_ids), (
        "First pass attempted every id (failure happens inside save_by_vector); "
        f"saw {sorted(svc1.save_calls)}"
    )

    # The failing id's indexed_at was cleared by the compensating reset.
    async with admin_pool.acquire() as conn:
        failed_row_state = await conn.fetchval(
            "SELECT indexed_at FROM entries WHERE id = $1", failing_id
        )
    assert (
        failed_row_state is None
    ), f"Compensating reset must NULL indexed_at on the failed id; got {failed_row_state}"

    # Second pass with no injected failure: must pick up the previously-
    # failed id and succeed.
    svc2 = _CountingEmbeddingService()
    ctx2 = _make_app_ctx(svc2, admin_pool)
    result2 = await _run_reindex(ctx2, admin_pool, cipher)

    assert result2["embeddings_failed"] == 0
    assert svc2.save_calls == [failing_id], (
        "Second pass must encode exactly the previously-failed id; " f"got {svc2.save_calls}"
    )


async def test_run_reindex_terminates_on_whole_batch_failure(
    admin_pool: asyncpg.Pool,
    cipher: ContentCipher,
    seeded_entries_for_cursor: tuple[UUID, list[int]],
) -> None:
    """Whole-batch failure must NOT loop forever -- the loop must terminate.

    Inject a save failure for every seeded id. The compensating reset
    succeeds (it just NULLs indexed_at). With the old behaviour
    (``last_id = batch[-1]["id"]``) the loop advanced past the reset
    rows; with the new behaviour ``last_id`` is held and the loop must
    break to avoid grinding on the same cursor.

    Either way the loop must terminate within a reasonable time.
    """
    tenant_id, entry_ids = seeded_entries_for_cursor

    svc = _CountingEmbeddingService(fail_on_ids=set(entry_ids))
    ctx = _make_app_ctx(svc, admin_pool)

    # Bound the call with a wall-clock timeout so an infinite-loop
    # regression surfaces as a test failure rather than a hung test
    # session.
    result = await asyncio.wait_for(_run_reindex(ctx, admin_pool, cipher), timeout=15.0)

    assert result["embeddings_generated"] == 0
    assert result["embeddings_failed"] == len(entry_ids)
    assert result["semantic_status"] == "partial"

    # Every row must end with indexed_at IS NULL (compensating reset
    # ran; loop broke without re-claiming).
    async with admin_pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, indexed_at FROM entries WHERE user_id = $1 ORDER BY id",
            tenant_id,
        )
    for r in rows:
        assert r["indexed_at"] is None, (
            f"entry {r['id']} stayed stamped after whole-batch failure; "
            "compensating reset must NULL indexed_at across the board"
        )
