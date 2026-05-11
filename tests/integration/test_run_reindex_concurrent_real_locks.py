"""X-4: ``_run_reindex`` correctly skips rows locked by another transaction.

The pre-fix shape used ``FOR UPDATE OF e SKIP LOCKED`` on the joined
SELECT, which (per architect D2) carried a plan-shape risk where the
join could pull rows the inner ``WHERE indexed_at IS NULL`` filter had
already excluded. The post-fix shape pushes the lock into a subquery
against ``entries`` alone, then joins ``topics`` only for the rows the
inner SELECT actually returned.

This test creates **real** PostgreSQL row-lock contention by holding an
explicit ``SELECT ... FOR UPDATE`` transaction on a known set of entry
ids, then runs the reindex worker. The worker must skip the locked rows
in its first pass; once the lock-holder commits, a follow-up pass picks
up the previously-skipped rows. Both passes must process disjoint id
sets and no rows must be lost.
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
    """Records every encode + save call for disjointness assertions."""

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
async def seeded_unindexed(
    admin_pool: asyncpg.Pool,
    tenant_a: UUID,
) -> tuple[UUID, list[int]]:
    """Seed N unindexed entries -- ``indexed_at = NULL`` and no embedding row."""
    seed = await seed_for(admin_pool, tenant_a, topic_path="reindex-x4/notes", n_entries=10)
    async with admin_pool.acquire() as conn:
        await conn.execute("DELETE FROM entry_embeddings WHERE user_id = $1", tenant_a)
        await conn.execute("UPDATE entries SET indexed_at = NULL WHERE user_id = $1", tenant_a)
    return tenant_a, list(seed.entry_ids)


async def test_run_reindex_skips_locked_rows_then_processes_them_after_release(
    admin_pool: asyncpg.Pool,
    cipher: ContentCipher,
    seeded_unindexed: tuple[UUID, list[int]],
) -> None:
    """Worker skips rows locked by a held transaction, then processes them on the next pass.

    Step 1: open a transaction that locks the first half of the entry
    ids via ``SELECT ... FOR UPDATE``. Hold the transaction open.
    Step 2: run the reindex worker. It must skip the locked rows and
    encode only the unlocked half.
    Step 3: commit the holding transaction (releasing the locks).
    Step 4: run the reindex worker again. The previously-locked rows
    are now eligible and must be encoded by this second pass.

    Across both passes: disjoint id sets, full coverage of the seed,
    no duplicate ``entry_embeddings`` rows.
    """
    tenant_id, entry_ids = seeded_unindexed
    locked_ids = sorted(entry_ids)[: len(entry_ids) // 2]
    free_ids = sorted(entry_ids)[len(entry_ids) // 2 :]

    holder_conn = await admin_pool.acquire()
    try:
        # Start a transaction on the holder connection and lock the first
        # half of the entry rows. The lock is held until commit/rollback.
        tx = holder_conn.transaction()
        await tx.start()
        try:
            held_rows = await holder_conn.fetch(
                "SELECT id FROM entries WHERE id = ANY($1) FOR UPDATE",
                locked_ids,
            )
            assert {r["id"] for r in held_rows} == set(
                locked_ids
            ), "holder transaction must lock the requested ids before the worker runs"

            # First pass: worker must skip the locked rows.
            svc1 = _CountingEmbeddingService()
            ctx1 = _make_app_ctx(svc1, admin_pool)
            result1 = await _run_reindex(ctx1, admin_pool, cipher)

            assert result1["status"] == "rebuilt"
            assert sorted(svc1.save_calls) == sorted(free_ids), (
                f"First pass must process only unlocked ids; "
                f"saved={sorted(svc1.save_calls)}, expected={sorted(free_ids)}, "
                f"locked={locked_ids}"
            )
            assert result1["embeddings_failed"] == 0

            # Verify locked rows are still NULL in entries.indexed_at -- the
            # worker must not have stamped them.
            async with admin_pool.acquire() as conn:
                still_unindexed = await conn.fetch(
                    "SELECT id FROM entries "
                    "WHERE id = ANY($1) AND indexed_at IS NULL "
                    "ORDER BY id",
                    locked_ids,
                )
            assert {r["id"] for r in still_unindexed} == set(
                locked_ids
            ), "Locked rows must NOT have been claimed by the first worker pass"
        finally:
            # Release the lock so the second pass can claim the rows.
            await tx.commit()
    finally:
        await admin_pool.release(holder_conn)

    # Second pass: previously-locked rows are now free.
    svc2 = _CountingEmbeddingService()
    ctx2 = _make_app_ctx(svc2, admin_pool)
    result2 = await _run_reindex(ctx2, admin_pool, cipher)

    assert result2["status"] == "rebuilt"
    assert sorted(svc2.save_calls) == sorted(locked_ids), (
        f"Second pass must process the previously-locked ids; "
        f"saved={sorted(svc2.save_calls)}, expected={sorted(locked_ids)}"
    )
    assert result2["embeddings_failed"] == 0

    # Disjointness across both passes (the core property).
    pass1_ids = set(svc1.save_calls)
    pass2_ids = set(svc2.save_calls)
    assert pass1_ids.isdisjoint(
        pass2_ids
    ), f"Passes must process disjoint id sets; overlap={pass1_ids & pass2_ids}"

    # Full coverage: every seeded id ended up in exactly one pass.
    assert pass1_ids | pass2_ids == set(
        entry_ids
    ), f"Coverage gap: missing ids = {set(entry_ids) - (pass1_ids | pass2_ids)}"

    # Database invariants: every seeded entry has an embedding row, no dupes.
    async with admin_pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT entry_id FROM entry_embeddings WHERE user_id = $1",
            tenant_id,
        )
    indexed_ids = [r["entry_id"] for r in rows]
    assert sorted(indexed_ids) == sorted(entry_ids), (
        f"entry_embeddings mismatch: missing={set(entry_ids) - set(indexed_ids)}, "
        f"extra={set(indexed_ids) - set(entry_ids)}"
    )
    assert len(indexed_ids) == len(
        set(indexed_ids)
    ), f"Duplicate entry_embeddings rows: {sorted(indexed_ids)}"
