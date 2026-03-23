"""Tests for storage/index.py — FTS5 search and indexing."""

from journalctl.storage.index import SearchIndex
from journalctl.storage.markdown import MarkdownStorage


class TestIndexing:
    """Document indexing and rebuilding."""

    def test_upsert_and_search(
        self,
        storage: MarkdownStorage,
        index: SearchIndex,
    ) -> None:
        storage.append_entry("work/acme", "Got the promotion.", date="2025-06-01")
        path = storage.topic_path("work/acme")
        index.upsert_file(path)

        results = index.search("promotion")
        assert len(results) == 1
        assert results[0].topic == "work/acme"

    def test_search_no_results(self, index: SearchIndex) -> None:
        results = index.search("nonexistent query")
        assert results == []

    def test_search_with_topic_filter(
        self,
        storage: MarkdownStorage,
        index: SearchIndex,
    ) -> None:
        storage.append_entry("work/acme", "Acme work.", date="2025-01-01")
        storage.append_entry("hobbies/running", "Running is fun.", date="2025-01-01")
        index.upsert_file(storage.topic_path("work/acme"))
        index.upsert_file(storage.topic_path("hobbies/running"))

        results = index.search("fun", topic_prefix="hobbies")
        assert len(results) == 1
        assert results[0].topic == "hobbies/running"

    def test_search_with_date_filter(
        self,
        storage: MarkdownStorage,
        index: SearchIndex,
    ) -> None:
        storage.append_entry("test/dates", "Old entry.", date="2020-01-01")
        index.upsert_file(storage.topic_path("test/dates"))

        results = index.search("entry", date_from="2021-01-01")
        assert results == []

        results = index.search("entry", date_to="2019-12-31")
        assert results == []

        results = index.search("entry", date_from="2020-01-01", date_to="2020-12-31")
        assert len(results) == 1

    def test_upsert_is_idempotent(
        self,
        storage: MarkdownStorage,
        index: SearchIndex,
    ) -> None:
        storage.append_entry("test/idem", "Content.", date="2025-01-01")
        path = storage.topic_path("test/idem")

        index.upsert_file(path)
        index.upsert_file(path)
        index.upsert_file(path)

        results = index.search("Content")
        assert len(results) == 1


class TestRebuild:
    """Full and incremental index rebuild."""

    def test_full_rebuild(
        self,
        storage: MarkdownStorage,
        index: SearchIndex,
    ) -> None:
        storage.append_entry("work/acme", "Entry 1.", date="2025-01-01")
        storage.append_entry("hobbies/running", "Entry 2.", date="2025-02-01")

        result = index.rebuild()
        assert result["documents_indexed"] == 2

        results = index.search("Entry")
        assert len(results) == 2

    def test_rebuild_clears_old_data(
        self,
        storage: MarkdownStorage,
        index: SearchIndex,
    ) -> None:
        storage.append_entry("test/delete-me", "Ghost entry.", date="2025-01-01")
        index.upsert_file(storage.topic_path("test/delete-me"))

        storage.topic_path("test/delete-me").unlink()

        index.rebuild()
        results = index.search("Ghost")
        assert results == []

    def test_incremental_rebuild(
        self,
        storage: MarkdownStorage,
        index: SearchIndex,
    ) -> None:
        storage.append_entry("test/incr", "First.", date="2025-01-01")
        count = index.incremental_rebuild()
        assert count >= 1

        count = index.incremental_rebuild()
        assert count == 0


class TestStats:
    """Index statistics."""

    def test_stats_empty(self, index: SearchIndex) -> None:
        stats = index.get_stats()
        assert stats["total_documents"] == 0

    def test_stats_with_data(
        self,
        storage: MarkdownStorage,
        index: SearchIndex,
    ) -> None:
        from journalctl.models.entry import Message

        storage.append_entry("work/acme", "Entry.", date="2025-01-01")
        index.upsert_file(storage.topic_path("work/acme"))

        msgs = [Message(role="user", content="Test")]
        path, _ = storage.save_conversation("work/acme", "Chat", msgs)
        index.upsert_file(path)

        stats = index.get_stats()
        assert stats["topics"] == 1
        assert stats["conversations"] == 1
        assert stats["total_documents"] == 2


class TestTimelineQuery:
    """Date-range queries for timeline."""

    def test_entries_by_date_range(
        self,
        storage: MarkdownStorage,
        index: SearchIndex,
    ) -> None:
        storage.append_entry("work/acme", "Jan entry.", date="2025-01-15")
        storage.append_entry("hobbies/running", "Feb entry.", date="2025-02-10")
        storage.append_entry("test/march", "Mar entry.", date="2025-03-05")
        index.rebuild()

        results = index.get_entries_by_date_range("2025-01-01", "2025-01-31")
        assert len(results) == 1
        assert results[0]["topic"] == "work/acme"

        results = index.get_entries_by_date_range("2025-01-01", "2025-12-31")
        assert len(results) == 3
