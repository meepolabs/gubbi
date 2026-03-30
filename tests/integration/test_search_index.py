"""Tests for storage/search_index.py — FTS5 search and indexing."""

from journalctl.models.conversation import Message
from journalctl.storage.database import DatabaseStorage
from journalctl.storage.search_index import SearchIndex


class TestIndexing:
    """Entry and conversation indexing via upsert methods."""

    def test_upsert_entry_and_search(
        self,
        storage: DatabaseStorage,
        index: SearchIndex,
    ) -> None:
        entry_id, _ = storage.append_entry("work/acme", "Got the promotion.", date="2025-06-01")
        index.upsert_entry(
            entry_id=entry_id,
            topic="work/acme",
            title="Acme Corp Notes",
            date="2025-06-01",
            content="Got the promotion.",
            context=None,
            tags=[],
        )

        results = index.search("promotion")
        assert len(results) == 1
        assert results[0].topic == "work/acme"
        assert results[0].entry_id == entry_id

    def test_search_no_results(self, index: SearchIndex) -> None:
        results = index.search("nonexistent query")
        assert results == []

    def test_search_with_topic_filter(
        self,
        storage: DatabaseStorage,
        index: SearchIndex,
    ) -> None:
        id1, _ = storage.append_entry("work/acme", "Acme work.", date="2025-01-01")
        id2, _ = storage.append_entry("hobbies/running", "Running is fun.", date="2025-01-01")

        index.upsert_entry(id1, "work/acme", "Acme", "2025-01-01", "Acme work.", None, [])
        index.upsert_entry(
            id2, "hobbies/running", "Running", "2025-01-01", "Running is fun.", None, []
        )

        results = index.search("fun", topic_prefix="hobbies")
        assert len(results) == 1
        assert results[0].topic == "hobbies/running"

    def test_search_with_date_filter(
        self,
        storage: DatabaseStorage,
        index: SearchIndex,
    ) -> None:
        entry_id, _ = storage.append_entry("test/dates", "Old entry.", date="2020-01-01")
        index.upsert_entry(entry_id, "test/dates", "Dates", "2020-01-01", "Old entry.", None, [])

        results = index.search("entry", date_from="2021-01-01")
        assert results == []

        results = index.search("entry", date_to="2019-12-31")
        assert results == []

        results = index.search("entry", date_from="2020-01-01", date_to="2020-12-31")
        assert len(results) == 1

    def test_upsert_entry_is_idempotent(
        self,
        storage: DatabaseStorage,
        index: SearchIndex,
    ) -> None:
        entry_id, _ = storage.append_entry("test/idem", "Content.", date="2025-01-01")

        for _ in range(3):
            index.upsert_entry(entry_id, "test/idem", "Test", "2025-01-01", "Content.", None, [])

        results = index.search("Content")
        assert len(results) == 1

    def test_upsert_conversation(
        self,
        storage: DatabaseStorage,
        index: SearchIndex,
    ) -> None:
        msgs = [
            Message(role="user", content="Best ramen recommendation?"),
            Message(role="assistant", content="Try Jinya in Bellevue."),
        ]
        conv_id, summary = storage.save_conversation(
            "hobbies/food", "Ramen Chat", msgs, summary="Ramen recommendations in Bellevue"
        )
        index.upsert_conversation(
            conversation_id=conv_id,
            topic="hobbies/food",
            title="Ramen Chat",
            summary=summary,
            tags=[],
            updated="2025-01-01",
            message_content="Best ramen recommendation? Try Jinya in Bellevue.",
        )

        results = index.search("ramen")
        assert len(results) == 1
        assert results[0].doc_type == "conversation"
        assert results[0].conversation_id == conv_id

    def test_context_included_in_index(
        self,
        storage: DatabaseStorage,
        index: SearchIndex,
    ) -> None:
        entry_id, _ = storage.append_entry(
            "test/context",
            "Chose SQLite.",
            context="Reason: markdown has no stable IDs",
            date="2025-01-01",
        )
        index.upsert_entry(
            entry_id,
            "test/context",
            "Test",
            "2025-01-01",
            "Chose SQLite.",
            "Reason: markdown has no stable IDs",
            [],
        )

        results = index.search("stable IDs")
        assert len(results) == 1


class TestRebuild:
    """Rebuild FTS5 index from database."""

    def test_rebuild_from_db(
        self,
        storage: DatabaseStorage,
        index: SearchIndex,
    ) -> None:
        storage.append_entry("work/acme", "Entry 1.", date="2025-01-01")
        storage.append_entry("hobbies/running", "Entry 2.", date="2025-02-01")

        result = index.rebuild_from_db(storage)
        assert result["documents_indexed"] == 2

        results = index.search("Entry")
        assert len(results) == 2

    def test_rebuild_clears_stale_data(
        self,
        storage: DatabaseStorage,
        index: SearchIndex,
    ) -> None:
        entry_id, _ = storage.append_entry("test/stale", "Ghost entry.", date="2025-01-01")
        index.upsert_entry(entry_id, "test/stale", "Test", "2025-01-01", "Ghost entry.", None, [])

        # Manually delete entry from DB (simulate deletion)
        storage.conn.execute("DELETE FROM entries WHERE id = ?", (entry_id,))
        storage.conn.commit()

        # Rebuild should not include the deleted entry
        index.rebuild_from_db(storage)
        results = index.search("Ghost")
        assert results == []


class TestStats:
    """Search index stats."""

    def test_stats_empty(self, index: SearchIndex) -> None:
        stats = index.conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        assert stats == 0

    def test_stats_after_upsert(
        self,
        storage: DatabaseStorage,
        index: SearchIndex,
    ) -> None:
        entry_id, _ = storage.append_entry("work/acme", "Entry.")
        index.upsert_entry(entry_id, "work/acme", "Acme", "2025-01-01", "Entry.", None, [])

        total = index.conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        assert total == 1
