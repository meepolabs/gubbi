"""Unit tests for ``reset_indexed_at_for_ids``.

The compensating-reset helper used by ``_run_reindex`` to roll back the
claim stamp when encode/save fails for a subset of a claimed batch.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from gubbi.storage.repositories import entries as entry_repo


@pytest.mark.unit
async def test_reset_indexed_at_for_ids_runs_update() -> None:
    """``reset_indexed_at_for_ids`` issues the expected UPDATE shape with the id list."""
    captured: dict[str, Any] = {}

    async def fake_execute(query: str, *args: Any, **kwargs: Any) -> str:
        captured["query"] = query
        captured["args"] = args
        return "UPDATE 0"

    conn = MagicMock()
    conn.execute = AsyncMock(side_effect=fake_execute)

    ids = [1, 2, 3]
    await entry_repo.reset_indexed_at_for_ids(conn, ids)

    assert captured["query"].strip().startswith("UPDATE entries SET indexed_at = NULL")
    assert "WHERE id = ANY($1)" in captured["query"]
    assert captured["args"] == (ids,)


@pytest.mark.unit
async def test_reset_indexed_at_for_ids_empty_list_is_noop() -> None:
    """Empty id list short-circuits before issuing a query."""
    conn = MagicMock()
    conn.execute = AsyncMock()

    await entry_repo.reset_indexed_at_for_ids(conn, [])

    conn.execute.assert_not_awaited()


@pytest.mark.unit
async def test_reset_indexed_at_for_ids_filters_tombstoned_rows() -> None:
    """The UPDATE WHERE clause must include ``AND deleted_at IS NULL``.

    Symmetric with ``reset_indexed_at`` (entries.py:402) which already
    filters tombstoned rows. Without this guard a tombstoned id in the
    failed-batch list would have its ``indexed_at`` cleared; the next
    ``get_unindexed`` would skip it (it filters ``deleted_at IS NULL``),
    so the row would be permanently stranded. The compensating reset
    must be a no-op for tombstoned rows.
    """
    captured: dict[str, Any] = {}

    async def fake_execute(query: str, *args: Any, **kwargs: Any) -> str:
        captured["query"] = query
        captured["args"] = args
        return "UPDATE 0"

    conn = MagicMock()
    conn.execute = AsyncMock(side_effect=fake_execute)

    await entry_repo.reset_indexed_at_for_ids(conn, [1, 2, 3])

    assert (
        "AND deleted_at IS NULL" in captured["query"]
    ), f"reset_indexed_at_for_ids must skip tombstoned rows; query was: {captured['query']!r}"
