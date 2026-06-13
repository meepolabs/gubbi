"""Search hydration must not silently drop hits when a batch fetch fails.

When the entry or conversation batch-decrypt query raises ``asyncpg.PostgresError``,
the affected hits are kept (not dropped) with ``decryption_failed=True`` and the
client-facing sentinel, and a structured warning is logged with the affected
count. ``total`` therefore reflects the merged hit count, not a silently pruned
subset.
"""

from __future__ import annotations

from typing import Any

import asyncpg
import pytest

from gubbi.models.search import SearchResult
from gubbi.services import search as search_svc


class _FakeRepo:
    """Stand-in for entry_repo / conv_repo whose batch fetch raises."""

    def __init__(self, *, raise_on: str) -> None:
        self._raise_on = raise_on

    async def get_texts(
        self, _conn: Any, _cipher: Any, ids: list[int]
    ) -> dict[int, tuple[str, str | None]]:
        if self._raise_on == "entries":
            raise asyncpg.PostgresError("simulated entry batch failure")
        return {i: ("ok content", None) for i in ids}

    async def get_titles_summaries(
        self, _conn: Any, _cipher: Any, ids: list[int]
    ) -> dict[int, tuple[str, str]]:
        if self._raise_on == "convs":
            raise asyncpg.PostgresError("simulated conversation batch failure")
        return {i: ("a title", "a summary") for i in ids}


@pytest.fixture
def merged_results() -> list[SearchResult]:
    return [
        SearchResult(
            source_key="entry:1",
            doc_type="entry",
            topic="work",
            rank=1.0,
            date="2026-05-01",
            entry_id=1,
        ),
        SearchResult(
            source_key="entry:2",
            doc_type="entry",
            topic="work",
            rank=2.0,
            date="2026-05-02",
            entry_id=2,
        ),
        SearchResult(
            source_key="conversation:9",
            doc_type="conversation",
            topic="chat",
            rank=3.0,
            date="2026-05-03",
            conversation_id=9,
        ),
    ]


async def test_entry_batch_failure_keeps_hits_with_marker(
    monkeypatch: pytest.MonkeyPatch, merged_results: list[SearchResult]
) -> None:
    """When the entry batch fetch raises, entry hits survive with the marker."""
    # Arrange
    fake = _FakeRepo(raise_on="entries")
    monkeypatch.setattr(search_svc.entry_repo, "get_texts", fake.get_texts)
    monkeypatch.setattr(search_svc.conv_repo, "get_titles_summaries", fake.get_titles_summaries)

    # Act
    hydrated = await search_svc._hydrate_results(None, object(), merged_results)  # type: ignore[arg-type]

    # Assert -- all three hits retained; both entries marked failed.
    assert len(hydrated) == 3
    entries = [h for h in hydrated if h.doc_type == "entry"]
    assert len(entries) == 2
    assert all(h.decryption_failed for h in entries)
    assert all(h.content == search_svc.DECRYPTION_FAILED_SENTINEL for h in entries)
    # Conversation hydrated normally.
    convs = [h for h in hydrated if h.doc_type == "conversation"]
    assert len(convs) == 1
    assert convs[0].title == "a title"


async def test_conversation_batch_failure_keeps_hits_with_marker(
    monkeypatch: pytest.MonkeyPatch, merged_results: list[SearchResult]
) -> None:
    """When the conversation batch fetch raises, conversation hits survive."""
    # Arrange
    fake = _FakeRepo(raise_on="convs")
    monkeypatch.setattr(search_svc.entry_repo, "get_texts", fake.get_texts)
    monkeypatch.setattr(search_svc.conv_repo, "get_titles_summaries", fake.get_titles_summaries)

    # Act
    hydrated = await search_svc._hydrate_results(None, object(), merged_results)  # type: ignore[arg-type]

    # Assert
    assert len(hydrated) == 3
    convs = [h for h in hydrated if h.doc_type == "conversation"]
    assert len(convs) == 1
    assert convs[0].decryption_failed is True
    assert convs[0].title == search_svc.DECRYPTION_FAILED_SENTINEL


async def test_no_failure_total_unchanged(
    monkeypatch: pytest.MonkeyPatch, merged_results: list[SearchResult]
) -> None:
    """Happy path: nothing dropped, nothing marked failed."""
    # Arrange
    fake = _FakeRepo(raise_on="none")
    monkeypatch.setattr(search_svc.entry_repo, "get_texts", fake.get_texts)
    monkeypatch.setattr(search_svc.conv_repo, "get_titles_summaries", fake.get_titles_summaries)

    # Act
    hydrated = await search_svc._hydrate_results(None, object(), merged_results)  # type: ignore[arg-type]

    # Assert
    assert len(hydrated) == 3
    assert not any(h.decryption_failed for h in hydrated)


class _FakeEmbeddingService:
    """Embedding service whose encode either returns a vector or raises."""

    def __init__(self, *, raise_on_encode: bool) -> None:
        self._raise = raise_on_encode

    def encode(self, _query: str) -> list[float]:
        if self._raise:
            raise RuntimeError("simulated encode failure")
        return [0.1, 0.2, 0.3]


class _FakeAppCtx:
    def __init__(self, *, raise_on_encode: bool) -> None:
        self.embedding_service = _FakeEmbeddingService(raise_on_encode=raise_on_encode)


async def test_encode_query_returns_vector() -> None:
    """encode_query returns the embedding off the event loop on success."""
    # Arrange
    app_ctx = _FakeAppCtx(raise_on_encode=False)

    # Act
    embedding = await search_svc.encode_query(app_ctx, "hello")  # type: ignore[arg-type]

    # Assert
    assert embedding == [0.1, 0.2, 0.3]


async def test_encode_query_degrades_to_none_on_failure() -> None:
    """encode_query returns None (FTS-only degrade) when encoding raises."""
    # Arrange
    app_ctx = _FakeAppCtx(raise_on_encode=True)

    # Act
    embedding = await search_svc.encode_query(app_ctx, "hello")  # type: ignore[arg-type]

    # Assert
    assert embedding is None
