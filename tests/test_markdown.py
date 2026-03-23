"""Tests for the markdown storage layer."""

from pathlib import Path

import pytest

from journalctl.models.entry import Message
from journalctl.storage.markdown import MarkdownStorage


class TestTopicCRUD:
    """Create, read, append topics."""

    def test_create_topic(self, storage: MarkdownStorage) -> None:
        storage.create_topic(
            topic="work/acme",
            title="Acme Corp Notes",
            description="Work notes for Acme Corp",
            tags=["work", "acme"],
        )

        meta, body = storage.read_topic("work/acme")
        assert meta.topic == "work/acme"
        assert meta.title == "Acme Corp Notes"
        assert meta.entry_count == 0
        assert "work" in meta.tags

    def test_create_topic_duplicate_raises(self, storage: MarkdownStorage) -> None:
        storage.create_topic("hobbies/cooking", "Cooking")
        with pytest.raises(FileExistsError):
            storage.create_topic("hobbies/cooking", "Cooking Again")

    def test_read_nonexistent_topic_raises(self, storage: MarkdownStorage) -> None:
        with pytest.raises(FileNotFoundError):
            storage.read_topic("nonexistent/topic")

    def test_append_creates_topic_if_missing(self, storage: MarkdownStorage) -> None:
        path, count = storage.append_entry(
            topic="life/food",
            content="Tried a new restaurant downtown.",
        )
        assert path.exists()
        assert count == 1

        meta, _ = storage.read_topic("life/food")
        assert meta.entry_count == 1

    def test_append_multiple_entries(self, storage: MarkdownStorage) -> None:
        storage.create_topic("hobbies/running", "Running")
        storage.append_entry("hobbies/running", "First 5K completed.", tags=["milestone"])
        storage.append_entry("hobbies/running", "Learned proper form.")
        storage.append_entry("hobbies/running", "New personal best.", tags=["milestone"])

        meta, body = storage.read_topic("hobbies/running")
        assert meta.entry_count == 3
        assert "First 5K" in body
        assert "personal best" in body

    def test_append_with_custom_date(self, storage: MarkdownStorage) -> None:
        storage.append_entry(
            "work/acme",
            "Started new project.",
            date="2024-03-15",
        )
        meta, body = storage.read_topic("work/acme")
        assert "## 2024-03-15" in body
        assert meta.updated == "2024-03-15"


class TestEntryParsing:
    """Parse entries from topic files."""

    def test_parse_entries(self, storage: MarkdownStorage) -> None:
        storage.append_entry("test/parsing", "First entry.", date="2024-01-01")
        storage.append_entry(
            "test/parsing",
            "Second entry with tags.",
            tags=["decision", "important"],
            date="2024-06-15",
        )
        storage.append_entry("test/parsing", "Third entry.", date="2025-01-01")

        _, body = storage.read_topic("test/parsing")
        entries = storage.parse_entries(body)

        assert len(entries) == 3
        assert entries[0].date == "2024-01-01"
        assert entries[0].index == 1
        assert entries[1].date == "2024-06-15"
        assert "decision" in entries[1].tags
        assert entries[2].date == "2025-01-01"

    def test_parse_entries_empty_topic(self, storage: MarkdownStorage) -> None:
        storage.create_topic("test/empty", "Empty Topic")
        _, body = storage.read_topic("test/empty")
        entries = storage.parse_entries(body)
        assert len(entries) == 0


class TestEntryUpdate:
    """Update existing entries."""

    def test_update_replace(self, storage: MarkdownStorage) -> None:
        storage.create_topic("test/update", "Update Test")
        storage.append_entry("test/update", "Original content.", date="2024-01-01")
        storage.append_entry("test/update", "Keep this.", date="2024-02-01")

        storage.update_entry("test/update", entry_index=1, content="Replaced content.")

        _, body = storage.read_topic("test/update")
        entries = storage.parse_entries(body)
        assert "Replaced content." in entries[0].content
        assert "Keep this." in entries[1].content

    def test_update_append(self, storage: MarkdownStorage) -> None:
        storage.create_topic("test/update2", "Update Test 2")
        storage.append_entry("test/update2", "Original.", date="2024-01-01")

        storage.update_entry(
            "test/update2",
            entry_index=1,
            content="Addendum.",
            mode="append",
        )

        _, body = storage.read_topic("test/update2")
        entries = storage.parse_entries(body)
        assert "Original." in entries[0].content
        assert "Addendum." in entries[0].content

    def test_update_out_of_range(self, storage: MarkdownStorage) -> None:
        storage.append_entry("test/range", "Entry.", date="2024-01-01")
        with pytest.raises(IndexError):
            storage.update_entry("test/range", entry_index=5, content="X")

    def test_update_invalid_mode(self, storage: MarkdownStorage) -> None:
        storage.append_entry("test/mode", "Entry.", date="2024-01-01")
        with pytest.raises(ValueError, match="Invalid mode"):
            storage.update_entry("test/mode", entry_index=1, content="X", mode="bad")


class TestTopicListing:
    """List topics."""

    def test_list_topics(self, storage: MarkdownStorage) -> None:
        storage.create_topic("work/acme", "Acme")
        storage.create_topic("work/initech", "Initech")
        storage.create_topic("hobbies/running", "Running")

        topics = storage.list_topics()
        assert len(topics) == 3

    def test_list_topics_with_prefix(self, storage: MarkdownStorage) -> None:
        storage.create_topic("work/acme", "Acme")
        storage.create_topic("work/initech", "Initech")
        storage.create_topic("hobbies/running", "Running")

        work_topics = storage.list_topics(topic_prefix="work")
        assert len(work_topics) == 2
        assert all(t.topic.startswith("work/") for t in work_topics)

    def test_list_topics_empty(self, storage: MarkdownStorage) -> None:
        topics = storage.list_topics()
        assert len(topics) == 0


class TestConversations:
    """Save, list, and read conversations."""

    def test_save_and_read_conversation(self, storage: MarkdownStorage) -> None:
        msgs = [
            Message(role="user", content="How do I write tests?"),
            Message(role="assistant", content="Use pytest with fixtures."),
        ]
        path, summary = storage.save_conversation(
            topic="work/acme",
            title="Testing Strategy",
            messages=msgs,
            tags=["work", "testing"],
        )

        assert "Testing Strategy" in summary
        assert path.exists()

        meta, content = storage.read_conversation("work/acme", "Testing Strategy")
        assert meta.title == "Testing Strategy"
        assert meta.message_count == 2
        assert "How do I write tests?" in content

    def test_save_conversation_idempotent(self, storage: MarkdownStorage) -> None:
        msgs_v1 = [Message(role="user", content="V1")]
        msgs_v2 = [
            Message(role="user", content="V1"),
            Message(role="user", content="V2"),
        ]

        storage.save_conversation("hobbies/running", "Weekly Recap", msgs_v1)
        storage.save_conversation("hobbies/running", "Weekly Recap", msgs_v2)

        convos = storage.list_conversations(topic_prefix="hobbies")
        assert len(convos) == 1

        meta, _ = storage.read_conversation("hobbies/running", "Weekly Recap")
        assert meta.message_count == 2

    def test_save_conversation_preserves_created_date(self, storage: MarkdownStorage) -> None:
        msgs = [Message(role="user", content="Hello")]
        storage.save_conversation("test/dates", "Chat", msgs)
        meta1, _ = storage.read_conversation("test/dates", "Chat")

        storage.save_conversation("test/dates", "Chat", msgs)
        meta2, _ = storage.read_conversation("test/dates", "Chat")
        assert meta1.created == meta2.created

    def test_list_conversations(self, storage: MarkdownStorage) -> None:
        msgs = [Message(role="user", content="Hello")]
        storage.save_conversation("work/acme", "Chat 1", msgs)
        storage.save_conversation("work/acme", "Chat 2", msgs)
        storage.save_conversation("hobbies/running", "Chat 3", msgs)

        all_convos = storage.list_conversations()
        assert len(all_convos) == 3

        work_convos = storage.list_conversations(topic_prefix="work")
        assert len(work_convos) == 2

    def test_read_nonexistent_conversation(self, storage: MarkdownStorage) -> None:
        with pytest.raises(FileNotFoundError):
            storage.read_conversation("work/acme", "Nonexistent")


class TestConversationSummaryUpsert:
    """Auto-summary entries in topic files."""

    def test_upsert_creates_summary_entry(self, storage: MarkdownStorage) -> None:
        storage.create_topic("work/acme", "Acme")
        storage.upsert_conversation_summary(
            topic="work/acme",
            conv_title="Testing Strategy",
            summary="Discussed testing best practices.",
            conv_date="2025-08-15",
        )

        _, body = storage.read_topic("work/acme")
        assert "Discussed testing best practices." in body
        assert "[[conversations/work/acme/testing-strategy]]" in body

    def test_upsert_updates_existing_summary(self, storage: MarkdownStorage) -> None:
        storage.create_topic("work/acme", "Acme")
        storage.upsert_conversation_summary("work/acme", "Chat", "Summary v1.", "2025-01-01")
        storage.upsert_conversation_summary("work/acme", "Chat", "Summary v2.", "2025-01-02")

        _, body = storage.read_topic("work/acme")
        assert "Summary v2." in body
        assert body.count("[[conversations/work/acme/chat]]") == 1


class TestKnowledge:
    """Knowledge file reading."""

    def test_read_knowledge(self, storage: MarkdownStorage, tmp_journal: Path) -> None:
        knowledge_dir = tmp_journal / "knowledge"
        (knowledge_dir / "user-profile.md").write_text(
            "# User Profile\n\nA software engineer based in Seattle.",
            encoding="utf-8",
        )

        content = storage.read_knowledge("user-profile")
        assert "software engineer" in content

    def test_read_missing_knowledge(self, storage: MarkdownStorage) -> None:
        content = storage.read_knowledge("nonexistent")
        assert content == ""
