"""Unit tests for ``get_unindexed`` query shape.

Asserts the SELECT carries ``FOR UPDATE SKIP LOCKED`` inside an inner
subquery against ``entries`` only, so concurrent reindex workers can
claim disjoint batches without locking the joined ``topics`` rows.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from gubbi.storage.repositories import entries as entry_repo


@pytest.mark.unit
async def test_get_unindexed_query_uses_subquery_for_update_skip_locked() -> None:
    """The ``get_unindexed`` SELECT must lock rows via an inner subquery.

    Pre-fix shape: ``FOR UPDATE OF e SKIP LOCKED`` on the joined SELECT
    (locks were scoped to ``entries`` but plan-shape was tied to the
    join). Post-fix shape: an inner subquery scans ``entries`` alone,
    takes ``FOR UPDATE SKIP LOCKED`` against rows the worker actually
    claims, and the outer SELECT joins ``topics`` only for the locked
    ids. ``last_id`` advances past actually-claimed rows; locked-and-
    skipped rows stay visible to the next pass.
    """
    captured: dict[str, Any] = {}

    async def fake_fetch(query: str, *args: Any, **kwargs: Any) -> list[Any]:
        captured["query"] = query
        captured["args"] = args
        return []

    conn = MagicMock()
    conn.fetch = AsyncMock(side_effect=fake_fetch)
    cipher = MagicMock()

    await entry_repo.get_unindexed(conn, cipher, last_id=0, batch_size=10)

    assert "query" in captured, "conn.fetch was not invoked"
    query = captured["query"]

    # Inner subquery must lock rows.
    assert "FOR UPDATE SKIP LOCKED" in query, (
        "get_unindexed must claim rows with FOR UPDATE SKIP LOCKED so "
        "concurrent reindex workers get disjoint batches; current query: " + query
    )
    # Subquery shape: ``e.id IN (SELECT id FROM entries ...``. The locking
    # SELECT MUST NOT join ``topics`` -- otherwise ``FOR UPDATE`` would
    # need ``OF e`` again to avoid trying to lock topic rows.
    assert "e.id IN (" in query, (
        "get_unindexed must restrict the join to ids selected by the inner "
        "locking subquery; current query: " + query
    )
    assert "SELECT id FROM entries" in query, (
        "get_unindexed inner subquery must SELECT id FROM entries (no JOIN); "  # noqa: S608 -- assertion message, not a SQL fragment
        "current query: " + query
    )
    # Sanity: cursor + limit args still passed positionally (last_id, batch_size).
    assert captured["args"] == (0, 10)
