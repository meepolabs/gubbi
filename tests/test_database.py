"""Tests for storage/database.py — canonical SQLite storage layer."""

import pytest

from journalctl.models.entry import Message
from journalctl.storage.database import DatabaseStorage


class TestTopicCRUD:
    """Create and read topics."""

    def test_create_topic(self, storage: DatabaseStorage) -> None:
        topic_id = storage.create_topic(
            topic="work/acme",
            title="Acme Corp Notes",
            description="Work notes for Acme Corp",
            tags=["work", "acme"],
        )
        assert isinstance(topic_id, int)

        meta = storage.get_topic("work/acme")
        assert meta is not None
        assert meta.topic == "work/acme"
        assert meta.title == "Acme Corp Notes"
        assert meta.entry_count == 0
        assert "work" in meta.tags

    def test_create_topic_duplicate_raises(self, storage: DatabaseStorage) -> None:
        storage.create_topic("hobbies/cooking", "Cooking")
        with pytest.raises(ValueError, match="already exists"):
            storage.create_topic("hobbies/cooking", "Cooking Again")

    def test_read_nonexistent_topic_returns_none(self, storage: DatabaseStorage) -> None:
        assert storage.get_topic("nonexistent/topic") is None

    def test_read_entries_nonexistent_raises(self, storage: DatabaseStorage) -> None:
        with pytest.raises(FileNotFoundError):
            storage.read_entries("nonexistent/topic")

    def test_list_topics_empty(self, storage: DatabaseStorage) -> None:
        assert storage.list_topics() == []

    def test_list_topics_with_prefix(self, storage: DatabaseStorage) -> None:
        storage.create_topic("work/acme", "Acme")
        storage.create_topic("hobbies/running", "Running")
        storage.create_topic("work/startup", "Startup")

        result = storage.list_topics(topic_prefix="work")
        paths = [t.topic for t in result]
        assert "work/acme" in paths
        assert "work/startup" in paths
        assert "hobbies/running" not in paths


class TestEntries:
    """Append, read, and update entries."""

    def test_append_creates_topic_if_missing(self, storage: DatabaseStorage) -> None:
        entry_id, count = storage.append_entry(
            topic="life/food",
            content="Tried a new restaurant downtown.",
        )
        assert isinstance(entry_id, int)
        assert count == 1

        meta = storage.get_topic("life/food")
        assert meta is not None

    def test_append_multiple_entries(self, storage: DatabaseStorage) -> None:
        storage.create_topic("hobbies/running", "Running")
        storage.append_entry("hobbies/running", "First 5K completed.", tags=["milestone"])
        storage.append_entry("hobbies/running", "Hit 10K target.", tags=["milestone"])
        entry_id, count = storage.append_entry("hobbies/running", "Marathon registered.")
        assert count == 3

    def test_append_with_context(self, storage: DatabaseStorage) -> None:
        entry_id, _ = storage.append_entry(
            topic="projects/alpha",
            content="Decided to use SQLite as canonical storage.",
            context="Markdown has no stable IDs. SQLite allows relationships and namespaces.",
            tags=["decision"],
        )
        _, entries = storage.read_entries("projects/alpha")
        assert (
            entries[0].context
            == "Markdown has no stable IDs. SQLite allows relationships and namespaces."
        )
        assert entries[0].id == entry_id

    def test_read_entries_returns_stable_ids(self, storage: DatabaseStorage) -> None:
        storage.create_topic("test/stable", "Test")
        storage.append_entry("test/stable", "Entry 1", date="2025-01-01")
        storage.append_entry("test/stable", "Entry 2", date="2025-02-01")

        meta, entries = storage.read_entries("test/stable")
        assert len(entries) == 2
        assert all(e.id is not None for e in entries)
        # IDs are distinct
        assert entries[0].id != entries[1].id

    def test_read_entries_returns_n_most_recent(self, storage: DatabaseStorage) -> None:
        storage.create_topic("test/recent", "Test")
        storage.append_entry("test/recent", "Entry 1", date="2025-01-01")
        storage.append_entry("test/recent", "Entry 2", date="2025-06-01")
        storage.append_entry("test/recent", "Entry 3", date="2025-12-01")

        meta, entries = storage.read_entries("test/recent", n=2)
        assert len(entries) == 2
        assert "Entry 2" in entries[0].content
        assert "Entry 3" in entries[1].content

    def test_update_entry_replace(self, storage: DatabaseStorage) -> None:
        entry_id, _ = storage.append_entry("test/update", "Original content.")

        storage.update_entry(entry_id, "Updated content.", mode="replace")

        _, entries = storage.read_entries("test/update")
        assert entries[0].content == "Updated content."

    def test_update_entry_append(self, storage: DatabaseStorage) -> None:
        entry_id, _ = storage.append_entry("test/update", "Original.")

        storage.update_entry(entry_id, "Added more.", mode="append")

        _, entries = storage.read_entries("test/update")
        assert "Original." in entries[0].content
        assert "Added more." in entries[0].content

    def test_update_entry_invalid_mode(self, storage: DatabaseStorage) -> None:
        entry_id, _ = storage.append_entry("test/update", "Content.")
        with pytest.raises(ValueError, match="Invalid mode"):
            storage.update_entry(entry_id, "New", mode="invalid")

    def test_update_nonexistent_entry_raises(self, storage: DatabaseStorage) -> None:
        with pytest.raises(IndexError):
            storage.update_entry(99999, "Content.")

    def test_update_context(self, storage: DatabaseStorage) -> None:
        entry_id, _ = storage.append_entry(
            "test/ctx", "Decision made.", context="Initial reasoning."
        )
        storage.update_entry(entry_id, "Decision made.", context="Updated reasoning.")

        _, entries = storage.read_entries("test/ctx")
        assert entries[0].context == "Updated reasoning."

    def test_entry_has_conversation_id_field(self, storage: DatabaseStorage) -> None:
        entry_id, _ = storage.append_entry("test/conv-ref", "Something happened.")
        _, entries = storage.read_entries("test/conv-ref")
        assert entries[0].conversation_id is None  # None by default


class TestConversations:
    """Save and read conversations."""

    def test_save_and_read_conversation(self, storage: DatabaseStorage) -> None:
        msgs = [
            Message(role="user", content="How should we approach this?"),
            Message(role="assistant", content="Start with the simplest solution."),
        ]
        conv_id, summary = storage.save_conversation(
            topic="work/acme",
            title="Planning Session",
            messages=msgs,
            source="claude",
            tags=["planning"],
        )
        assert isinstance(conv_id, int)
        assert "How should we approach" in summary or "Planning Session" in summary

        meta, messages = storage.read_conversation("work/acme", "Planning Session")
        assert meta.title == "Planning Session"
        assert meta.message_count == 2
        assert meta.source == "claude"
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[1].role == "assistant"

    def test_save_idempotent(self, storage: DatabaseStorage) -> None:
        msgs_v1 = [Message(role="user", content="Q1"), Message(role="assistant", content="A1")]
        msgs_v2 = [
            Message(role="user", content="Q1"),
            Message(role="assistant", content="A1"),
            Message(role="user", content="Q2"),
            Message(role="assistant", content="A2"),
        ]
        id1, _ = storage.save_conversation("test/idem", "Chat", msgs_v1)
        id2, _ = storage.save_conversation("test/idem", "Chat", msgs_v2)

        # Same conversation — same ID
        assert id1 == id2

        meta, messages = storage.read_conversation("test/idem", "Chat")
        assert meta.message_count == 4
        assert len(messages) == 4

    def test_save_preserves_created_date(self, storage: DatabaseStorage) -> None:
        msgs = [Message(role="user", content="Hello")]
        id1, _ = storage.save_conversation("test/dates", "Chat", msgs)

        meta1, _ = storage.read_conversation("test/dates", "Chat")
        original_created = meta1.created

        # Re-save
        storage.save_conversation(
            "test/dates", "Chat", msgs + [Message(role="assistant", content="Hi")]
        )
        meta2, _ = storage.read_conversation("test/dates", "Chat")
        assert meta2.created == original_created

    def test_list_conversations(self, storage: DatabaseStorage) -> None:
        msgs = [Message(role="user", content="Content")]
        storage.save_conversation("work/acme", "Chat 1", msgs)
        storage.save_conversation("work/acme", "Chat 2", msgs)
        storage.save_conversation("hobbies/running", "Run Chat", msgs)

        all_convs = storage.list_conversations()
        assert len(all_convs) == 3

        work_convs = storage.list_conversations(topic_prefix="work")
        assert len(work_convs) == 2

    def test_read_nonexistent_conversation_raises(self, storage: DatabaseStorage) -> None:
        with pytest.raises(FileNotFoundError):
            storage.read_conversation("test/nope", "Missing Chat")

    def test_conversation_json_archive_written(self, storage: DatabaseStorage) -> None:
        msgs = [Message(role="user", content="Test message")]
        conv_id, _ = storage.save_conversation("test/archive", "Archive Test", msgs)

        json_dir = storage.conversations_json_dir / "test" / "archive"
        json_files = list(json_dir.glob("*.json"))
        assert len(json_files) == 1

        import json  # noqa: PLC0415

        data = json.loads(json_files[0].read_text())
        assert "meta" in data
        assert "messages" in data
        assert data["messages"][0]["content"] == "Test message"


class TestKnowledge:
    """Knowledge file reading (still file-based)."""

    def test_read_knowledge_returns_empty_if_missing(self, storage: DatabaseStorage) -> None:
        result = storage.read_knowledge("nonexistent")
        assert result == ""

    def test_read_knowledge_returns_content(self, storage: DatabaseStorage) -> None:
        profile_path = storage.journal_root / "knowledge" / "user-profile.md"
        profile_path.write_text("# User\n\nSoftware engineer.", encoding="utf-8")

        result = storage.read_knowledge("user-profile")
        assert "Software engineer" in result


class TestStats:
    """Stats and date range queries."""

    def test_stats_empty(self, storage: DatabaseStorage) -> None:
        stats = storage.get_stats()
        assert stats["total_documents"] == 0
        assert stats["topics"] == 0
        assert stats["conversations"] == 0

    def test_stats_with_data(self, storage: DatabaseStorage) -> None:
        storage.append_entry("work/acme", "Entry 1.")
        msgs = [Message(role="user", content="Hello")]
        storage.save_conversation("work/acme", "Chat", msgs)

        stats = storage.get_stats()
        assert stats["topics"] == 1
        assert stats["conversations"] == 1
        # 2 entries (1 manual + 1 auto-created by save_conversation) + 1 conversation
        assert stats["total_documents"] == 3

    def test_entries_by_date_range(self, storage: DatabaseStorage) -> None:
        storage.append_entry("work/acme", "Jan entry.", date="2025-01-15")
        storage.append_entry("hobbies/running", "Feb entry.", date="2025-02-10")
        storage.append_entry("test/march", "Mar entry.", date="2025-03-05")

        results = storage.get_entries_by_date_range("2025-01-01", "2025-01-31")
        assert len(results) == 1
        assert results[0]["topic"] == "work/acme"

        results = storage.get_entries_by_date_range("2025-01-01", "2025-12-31")
        assert len(results) == 3
