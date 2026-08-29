"""Concurrency tests for the reindex claim-then-process refactor.

Two reindex tasks run concurrently; ``FOR UPDATE OF e SKIP LOCKED`` plus
the in-transaction ``mark_indexed_batch`` claim must produce a disjoint
partitioning of the unindexed entries -- no duplicate ``encode`` calls,
no duplicate ``entry_embeddings`` rows, all entries reach a terminal
state (indexed or compensated).
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
from gubbi.storage.repositories import entries as entry_repo
from gubbi.tools.admin import _run_reindex
from tests.fixtures.tenants import seed_for

pytestmark = pytest.mark.asyncio(loop_scope="session")


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _CountingEmbeddingService:
    """Minimal stand-in for ``EmbeddingService`` that records every encode call.

    ``_run_reindex`` only touches ``encode`` (sync, CPU-bound stand-in)
    and ``save_by_vector`` (async, INSERT into ``entry_embeddings``).
    Tracks invocation counts per entry id so duplicates surface as
    failed assertions.
    """

    def __init__(self, fail_on_ids: set[int] | None = None) -> None:
        self.fail_on_ids: set[int] = fail_on_ids or set()
        self.encode_calls: list[str] = []
        self.save_calls: list[int] = []
        self._lock = asyncio.Lock()

    def encode(self, text: str) -> list[float]:
        # ``encode`` runs in ``asyncio.to_thread`` so a plain sync method
        # is what ``_run_reindex`` expects. Record the call without any
        # heavy work; the test only cares about counts and disjointness.
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
    """Spec-typed MagicMock so ``_run_reindex`` can read ``embedding_service``.

    ``admin_pool`` is wired identity-wise so ``_run_reindex``'s
    ``assert admin_pool is app_ctx.admin_pool`` boundary check passes.
    """
    app_ctx = MagicMock(spec=AppContext)
    app_ctx.embedding_service = embedding_service
    app_ctx.admin_pool = admin_pool
    return app_ctx


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def seeded_entries(
    admin_pool: asyncpg.Pool,
    tenant_a: UUID,
) -> tuple[UUID, list[int]]:
    """Seed N entries for tenant A and clear ``indexed_at`` so they look unindexed.

    ``seed_for`` plants ``entry_embeddings`` rows with ``indexed_at = now()``
    on the embedding row, but the test path looks at ``entries.indexed_at``
    -- which ``seed_for`` does not stamp explicitly, so the rows are
    already eligible for ``get_unindexed``. Drop the embedding rows
    so the encode pipeline has work to do, and wipe ``entries.indexed_at``
    defensively in case future ``seed_for`` changes start stamping it.
    """
    seed = await seed_for(admin_pool, tenant_a, topic_path="reindex-c/notes", n_entries=20)
    async with admin_pool.acquire() as conn:
        await conn.execute("DELETE FROM entry_embeddings WHERE user_id = $1", tenant_a)
        await conn.execute("UPDATE entries SET indexed_at = NULL WHERE user_id = $1", tenant_a)
    return tenant_a, list(seed.entry_ids)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_two_run_reindex_tasks_no_duplicate_embeddings(
    admin_pool: asyncpg.Pool,
    cipher: ContentCipher,
    seeded_entries: tuple[UUID, list[int]],
) -> None:
    """Two concurrent ``_run_reindex`` callers must produce disjoint claims.

    Skip the advisory lock here -- this is the contention path the new
    SKIP LOCKED clause must defend on its own. Each entry id is encoded
    by AT MOST one task, and ``entry_embeddings`` ends up with exactly
    one row per seeded id.
    """
    tenant_id, entry_ids = seeded_entries

    svc1 = _CountingEmbeddingService()
    svc2 = _CountingEmbeddingService()
    ctx1 = _make_app_ctx(svc1, admin_pool)
    ctx2 = _make_app_ctx(svc2, admin_pool)

    results = await asyncio.gather(
        _run_reindex(ctx1, admin_pool, cipher),
        _run_reindex(ctx2, admin_pool, cipher),
    )

    # Both runs returned a coherent result envelope.
    for r in results:
        assert r["status"] == "rebuilt"

    # No entry was encoded twice across the two tasks.
    saved = svc1.save_calls + svc2.save_calls
    assert len(saved) == len(set(saved)), f"Duplicate save_by_vector calls: {sorted(saved)}"

    # Every seeded entry got exactly one embedding row.
    async with admin_pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT entry_id FROM entry_embeddings WHERE user_id = $1",
            tenant_id,
        )
    indexed_ids = {r["entry_id"] for r in rows}
    assert indexed_ids == set(entry_ids), (
        f"entry_embeddings mismatch: missing={set(entry_ids) - indexed_ids}, "
        f"extra={indexed_ids - set(entry_ids)}"
    )


async def test_run_reindex_single_runner_unchanged(
    admin_pool: asyncpg.Pool,
    cipher: ContentCipher,
    seeded_entries: tuple[UUID, list[int]],
) -> None:
    """Single-runner reindex still produces one embedding per entry.

    Regression guard: the claim-then-process refactor must not change
    the happy-path behaviour for the existing single-runner caller.
    """
    tenant_id, entry_ids = seeded_entries
    svc = _CountingEmbeddingService()
    ctx = _make_app_ctx(svc, admin_pool)

    result = await _run_reindex(ctx, admin_pool, cipher)

    assert result["status"] == "rebuilt"
    assert result["semantic_status"] == "ok"
    assert result["embeddings_generated"] == len(entry_ids)
    assert result["embeddings_failed"] == 0
    assert sorted(svc.save_calls) == sorted(entry_ids)

    async with admin_pool.acquire() as conn:
        count = await conn.fetchval(
            "SELECT count(*) FROM entry_embeddings WHERE user_id = $1",
            tenant_id,
        )
    assert count == len(entry_ids)


async def test_compensating_reset_on_save_failure(
    admin_pool: asyncpg.Pool,
    cipher: ContentCipher,
    seeded_entries: tuple[UUID, list[int]],
) -> None:
    """Failed entries get ``indexed_at`` cleared so the next pass retries.

    Inject a save failure for one specific entry id; assert that entry's
    ``indexed_at`` is NULL after ``_run_reindex`` returns, while the
    other entries are stamped.
    """
    tenant_id, entry_ids = seeded_entries
    failing_id = entry_ids[len(entry_ids) // 2]

    svc = _CountingEmbeddingService(fail_on_ids={failing_id})
    ctx = _make_app_ctx(svc, admin_pool)

    result = await _run_reindex(ctx, admin_pool, cipher)

    assert result["embeddings_failed"] == 1
    assert result["semantic_status"] == "partial"

    async with admin_pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, indexed_at FROM entries WHERE user_id = $1 ORDER BY id",
            tenant_id,
        )
    by_id = {r["id"]: r["indexed_at"] for r in rows}
    assert by_id[failing_id] is None, (
        "Failed entry must have indexed_at cleared so the next reindex retries"
    )
    others_stamped = [eid for eid, ts in by_id.items() if eid != failing_id and ts is not None]
    assert len(others_stamped) == len(entry_ids) - 1, (
        f"Non-failing entries must remain stamped; stamped={others_stamped}, "
        f"all_ids={entry_ids}, failing={failing_id}"
    )


async def test_reset_indexed_at_for_ids_clears_only_target_rows(
    admin_pool: asyncpg.Pool,
    seeded_entries: tuple[UUID, list[int]],
) -> None:
    """Direct-call regression: ``reset_indexed_at_for_ids`` only touches the supplied ids."""
    tenant_id, entry_ids = seeded_entries
    async with admin_pool.acquire() as conn:
        # Pre-stamp every entry so the reset signal is observable.
        await conn.execute(
            "UPDATE entries SET indexed_at = now() WHERE user_id = $1",
            tenant_id,
        )

        target_ids = entry_ids[:3]
        await entry_repo.reset_indexed_at_for_ids(conn, target_ids)

        rows = await conn.fetch(
            "SELECT id, indexed_at FROM entries WHERE user_id = $1 ORDER BY id",
            tenant_id,
        )
    by_id = {r["id"]: r["indexed_at"] for r in rows}
    for eid in target_ids:
        assert by_id[eid] is None, f"entry {eid} should have been reset"
    for eid in entry_ids[3:]:
        assert by_id[eid] is not None, f"entry {eid} should remain stamped"


async def test_reset_indexed_at_for_ids_skips_tombstoned_rows(
    admin_pool: asyncpg.Pool,
    seeded_entries: tuple[UUID, list[int]],
) -> None:
    """Direct-call regression: tombstoned ids must remain stamped during reset."""
    tenant_id, entry_ids = seeded_entries
    live_id, tombstoned_id = entry_ids[:2]

    async with admin_pool.acquire() as conn:
        await conn.execute(
            "UPDATE entries SET indexed_at = now() WHERE user_id = $1",
            tenant_id,
        )
        await conn.execute(
            "UPDATE entries SET deleted_at = now() WHERE id = $1",
            tombstoned_id,
        )

        await entry_repo.reset_indexed_at_for_ids(conn, [live_id, tombstoned_id])

        rows = await conn.fetch(
            "SELECT id, indexed_at, deleted_at FROM entries WHERE user_id = $1 ORDER BY id",
            tenant_id,
        )

    by_id = {r["id"]: (r["indexed_at"], r["deleted_at"]) for r in rows}
    assert by_id[live_id] == (None, None), f"live entry {live_id} should have been reset"
    assert by_id[tombstoned_id][0] is not None, (
        f"tombstoned entry {tombstoned_id} must keep its indexed_at stamp"
    )
    assert by_id[tombstoned_id][1] is not None, (
        f"tombstoned entry {tombstoned_id} must remain tombstoned"
    )
