"""Per-row ``app.current_user_id`` propagation in ``_run_reindex``.

``_run_reindex`` runs on the BYPASSRLS admin pool so ``mark_indexed_batch``
and ``reset_indexed_at_for_ids`` can issue cross-user UPDATEs.  On that
pool ``app.current_user_id`` is unset on a fresh checkout.  The embedding
upsert in ``save_by_vector`` reads that GUC to populate
``entry_embeddings.user_id``; without an explicit bind, the column
lands NULL and HNSW + RLS later filter the row out of every tenant's
semantic search.

The fix: before each ``save_by_vector`` call, bind
``app.current_user_id`` to the row's owning user via
``SELECT set_config('app.current_user_id', $1, true)`` inside a
transaction so ``SET LOCAL`` releases on commit.

This unit test asserts that the bind runs once per row, with the row's
``user_id`` argument, and that it happens BEFORE the ``save_by_vector``
call on the same connection.

The pool-identity guard (``ValueError`` on a mismatched ``admin_pool``)
is covered by ``test_admin_bypassrls_guard.py`` and not re-asserted here.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from gubbi.app_context import AppContext
from gubbi.tools.admin import _run_reindex


@pytest.mark.unit
async def test_run_reindex_binds_current_user_id_per_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Per-row ``set_config('app.current_user_id', row.user_id, true)`` runs
    before ``save_by_vector`` so ``entry_embeddings.user_id`` is populated.

    Pre-fix: ``save_by_vector`` ran on a BYPASSRLS connection with an
    unset GUC, the embedding UPSERT bound ``user_id`` from
    ``current_setting('app.current_user_id', true)`` which returned ''
    and was coerced to NULL by the ``NULLIF`` wrapper.  Stranded rows.
    """
    # Two rows from two different users, plus an unindexed-batch fixture.
    user_a, user_b = uuid4(), uuid4()
    rows = [
        {"id": 1, "user_id": user_a, "content": "hello", "tags": []},
        {"id": 2, "user_id": user_b, "content": "world", "tags": []},
    ]

    # Connection captures every ``execute`` call so we can pin order.
    executed: list[tuple[str, tuple[Any, ...]]] = []

    class _Conn:
        async def execute(self, sql: str, *args: Any) -> None:
            executed.append((sql, args))

        def transaction(self) -> Any:
            ctx = MagicMock()
            ctx.__aenter__ = AsyncMock(return_value=ctx)
            ctx.__aexit__ = AsyncMock(return_value=False)
            return ctx

    # Pool yields the same connection every time and records the order of
    # ``save_by_vector`` calls relative to the ``set_config`` execs.
    save_calls: list[tuple[int, UUID | None]] = []

    async def _save_by_vector(conn: Any, entry_id: int, embedding: list[float]) -> None:
        # Snapshot the most recently executed ``set_config`` so we can
        # assert it landed BEFORE this save.
        last_set_config = next(
            (
                args
                for sql, args in reversed(executed)
                if "set_config" in sql and "app.current_user_id" in sql
            ),
            None,
        )
        bound_user = UUID(last_set_config[0]) if last_set_config else None
        save_calls.append((entry_id, bound_user))

    # ``safe_acquire`` yields a fresh ``_Conn`` per call; patch it so the
    # whole loop sees the recording connection.  The transaction() is a
    # no-op context manager (we are asserting on call order, not on
    # commit/rollback semantics).
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _fake_acquire(_pool: Any) -> Any:
        yield _Conn()

    monkeypatch.setattr("gubbi.tools.admin.safe_acquire", _fake_acquire)

    # Mock the claim step so it returns ``rows`` exactly once, then [].
    batches = iter([rows, []])

    async def _fake_get_unindexed(
        _conn: Any, _cipher: Any, _last_id: int, _batch_size: int
    ) -> list[dict[str, Any]]:
        return next(batches)

    async def _fake_mark_indexed_batch(_conn: Any, _ids: list[int]) -> None:
        return None

    monkeypatch.setattr("gubbi.storage.repositories.entries.get_unindexed", _fake_get_unindexed)
    monkeypatch.setattr(
        "gubbi.storage.repositories.entries.mark_indexed_batch",
        _fake_mark_indexed_batch,
    )

    admin_pool = MagicMock()
    app_ctx = MagicMock(spec=AppContext)
    app_ctx.admin_pool = admin_pool
    embedding_service = MagicMock()
    embedding_service.encode = MagicMock(return_value=[0.0] * 384)
    embedding_service.save_by_vector = _save_by_vector
    app_ctx.embedding_service = embedding_service
    cipher = MagicMock()

    result = await _run_reindex(app_ctx, admin_pool, cipher)

    # Two rows processed, no failures.
    assert result["embeddings_generated"] == 2
    assert result["embeddings_failed"] == 0

    # Bind ran with each row's user_id before its save.
    assert save_calls == [(1, user_a), (2, user_b)], (
        "Each save_by_vector must observe a prior set_config bound to "
        f"the row's owning user; got {save_calls!r}"
    )

    # And the executed set_config sequence used the GUC name + true (local).
    set_config_sqls = [sql for sql, _ in executed if "set_config" in sql]
    assert len(set_config_sqls) == 2
    assert all("app.current_user_id" in sql for sql in set_config_sqls)
    assert all("true" in sql for sql in set_config_sqls), (
        "set_config must be invoked with the is_local=true flag so the "
        "binding is transaction-scoped (SET LOCAL semantics); "
        f"got {set_config_sqls!r}"
    )
