"""Tests for storage repositories and knowledge — PostgreSQL storage."""

import json
from pathlib import Path

import asyncpg
import pytest

from journalctl.models.conversation import Message
from journalctl.storage import knowledge
from journalctl.storage.exceptions import (
    ConversationNotFoundError,
    EntryNotFoundError,
    TopicNotFoundError,
)
from journalctl.storage.repositories import conversations as conv_repo
from journalctl.storage.repositories import entries as entry_repo
from journalctl.storage.repositories import topics as topic_repo


class TestTopicCRUD:
    """Create and read topics."""

    async def test_create_topic(self, clean_pool: asyncpg.Pool) -> None:
        async with clean_pool.acquire() as conn:
            topic_id = await topic_repo.create(
                conn,
                topic="work/acme",
                title="Acme Corp Notes",
                description="Work notes for Acme Corp",
            )
            assert isinstance(topic_id, int)

            meta = await topic_repo.get(conn, "work/acme")
        assert meta is not None
        assert meta.topic == "work/acme"
        assert meta.title == "Acme Corp Notes"
        assert meta.entry_count == 0

    async def test_create_topic_duplicate_raises(self, clean_pool: asyncpg.Pool) -> None:
        async with clean_pool.acquire() as conn:
            await topic_repo.create(conn, "hobbies/cooking", "Cooking")
            with pytest.raises(ValueError, match="already exists"):
                await topic_repo.create(conn, "hobbies/cooking", "Cooking Again")

    async def test_read_nonexistent_topic_returns_none(self, clean_pool: asyncpg.Pool) -> None:
        async with clean_pool.acquire() as conn:
            assert await topic_repo.get(conn, "nonexistent/topic") is None

    async def test_read_entries_nonexistent_raises(self, clean_pool: asyncpg.Pool) -> None:
        async with clean_pool.acquire() as conn:
            with pytest.raises(TopicNotFoundError):
                await entry_repo.read(conn, "nonexistent/topic")

    async def test_list_topics_empty(self, clean_pool: asyncpg.Pool) -> None:
        async with clean_pool.acquire() as conn:
            assert await topic_repo.list_all(conn) == []

    async def test_list_topics_with_prefix(self, clean_pool: asyncpg.Pool) -> None:
        async with clean_pool.acquire() as conn:
            await topic_repo.create(conn, "work/acme", "Acme")
            await topic_repo.create(conn, "hobbies/running", "Running")
            await topic_repo.create(conn, "work/startup", "Startup")

            result = await topic_repo.list_all(conn, topic_prefix="work")
        paths = [t.topic for t in result]
        assert "work/acme" in paths
        assert "work/startup" in paths
        assert "hobbies/running" not in paths


class TestEntries:
    """Append, read, and update entries."""

    async def test_append_raises_if_topic_missing(self, clean_pool: asyncpg.Pool) -> None:
        async with clean_pool.acquire() as conn:
            with pytest.raises(TopicNotFoundError):
                await entry_repo.append(conn, topic="life/food", content="Tried a new restaurant.")

    async def test_append_multiple_entries(self, clean_pool: asyncpg.Pool) -> None:
        async with clean_pool.acquire() as conn:
            await topic_repo.create(conn, "hobbies/running", "Running")
            await entry_repo.append(
                conn, "hobbies/running", "First 5K completed.", tags=["milestone"]
            )
            await entry_repo.append(conn, "hobbies/running", "Hit 10K target.", tags=["milestone"])
            await entry_repo.append(conn, "hobbies/running", "Marathon registered.")
        # Verify count via read
        async with clean_pool.acquire() as conn:
            _, entries, total = await entry_repo.read(conn, "hobbies/running")
        assert total == 3

    async def test_append_with_reasoning(self, clean_pool: asyncpg.Pool) -> None:
        async with clean_pool.acquire() as conn:
            await topic_repo.create(conn, "projects/alpha", "Projects Alpha")
            entry_id = await entry_repo.append(
                conn,
                topic="projects/alpha",
                content="Decided to use PostgreSQL as canonical storage.",
                reasoning="SQLite had no async driver and limited concurrency.",
                tags=["decision"],
            )
            _, entries, _ = await entry_repo.read(conn, "projects/alpha")
        assert entries[0].reasoning == "SQLite had no async driver and limited concurrency."
        assert entries[0].id == entry_id

    async def test_read_entries_returns_stable_ids(self, clean_pool: asyncpg.Pool) -> None:
        async with clean_pool.acquire() as conn:
            await topic_repo.create(conn, "test/stable", "Test")
            await entry_repo.append(conn, "test/stable", "Entry 1", date="2025-01-01")
            await entry_repo.append(conn, "test/stable", "Entry 2", date="2025-02-01")

            meta, entries, _ = await entry_repo.read(conn, "test/stable")
        assert len(entries) == 2
        assert all(e.id is not None for e in entries)
        assert entries[0].id != entries[1].id

    async def test_read_entries_returns_n_most_recent(self, clean_pool: asyncpg.Pool) -> None:
        async with clean_pool.acquire() as conn:
            await topic_repo.create(conn, "test/recent", "Test")
            await entry_repo.append(conn, "test/recent", "Entry 1", date="2025-01-01")
            await entry_repo.append(conn, "test/recent", "Entry 2", date="2025-06-01")
            await entry_repo.append(conn, "test/recent", "Entry 3", date="2025-12-01")

            meta, entries, _ = await entry_repo.read(conn, "test/recent", limit=2)
        assert len(entries) == 2
        assert "Entry 2" in entries[0].content
        assert "Entry 3" in entries[1].content

    async def test_update_entry_replace(self, clean_pool: asyncpg.Pool) -> None:
        async with clean_pool.acquire() as conn:
            await topic_repo.create(conn, "test/update", "Test Update")
            entry_id = await entry_repo.append(conn, "test/update", "Original content.")
            await entry_repo.update(conn, entry_id, content="Updated content.", mode="replace")
            _, entries, _ = await entry_repo.read(conn, "test/update")
        assert entries[0].content == "Updated content."

    async def test_update_entry_append(self, clean_pool: asyncpg.Pool) -> None:
        async with clean_pool.acquire() as conn:
            await topic_repo.create(conn, "test/update", "Test Update")
            entry_id = await entry_repo.append(conn, "test/update", "Original.")
            await entry_repo.update(conn, entry_id, content="Added more.", mode="append")
            _, entries, _ = await entry_repo.read(conn, "test/update")
        assert "Original." in entries[0].content
        assert "Added more." in entries[0].content

    async def test_update_entry_invalid_mode(self, clean_pool: asyncpg.Pool) -> None:
        async with clean_pool.acquire() as conn:
            await topic_repo.create(conn, "test/update", "Test Update")
            entry_id = await entry_repo.append(conn, "test/update", "Content.")
            with pytest.raises(ValueError, match="Invalid mode"):
                await entry_repo.update(conn, entry_id, content="New", mode="invalid")

    async def test_update_nonexistent_entry_raises(self, clean_pool: asyncpg.Pool) -> None:
        async with clean_pool.acquire() as conn:
            with pytest.raises(EntryNotFoundError):
                await entry_repo.update(conn, 99999, content="Content.")

    async def test_update_reasoning(self, clean_pool: asyncpg.Pool) -> None:
        async with clean_pool.acquire() as conn:
            await topic_repo.create(conn, "test/ctx", "Test Ctx")
            entry_id = await entry_repo.append(
                conn, "test/ctx", "Decision made.", reasoning="Initial reasoning."
            )
            await entry_repo.update(conn, entry_id, reasoning="Updated reasoning.")
            _, entries, _ = await entry_repo.read(conn, "test/ctx")
        assert entries[0].reasoning == "Updated reasoning."

    async def test_entry_has_conversation_id_field(self, clean_pool: asyncpg.Pool) -> None:
        async with clean_pool.acquire() as conn:
            await topic_repo.create(conn, "test/conv-ref", "Test Conv Ref")
            await entry_repo.append(conn, "test/conv-ref", "Something happened.")
            _, entries, _ = await entry_repo.read(conn, "test/conv-ref")
        assert entries[0].conversation_id is None


class TestConversations:
    """Save and read conversations."""

    async def test_save_and_read_conversation(
        self, clean_pool: asyncpg.Pool, tmp_journal: Path
    ) -> None:
        async with clean_pool.acquire() as conn:
            await topic_repo.create(conn, "work/acme", "Acme Corp Notes")
        msgs = [
            Message(role="user", content="How should we approach this?"),
            Message(role="assistant", content="Start with the simplest solution."),
        ]
        conv_id, summary, _ = await conv_repo.save_conversation(
            clean_pool,
            conversations_json_dir=tmp_journal / "conversations_json",
            topic="work/acme",
            title="Planning Session",
            messages=msgs,
            summary="Discussion about project approach",
            source="claude",
            tags=["planning"],
        )
        assert isinstance(conv_id, int)
        assert summary == "Discussion about project approach"

        async with clean_pool.acquire() as conn:
            meta, messages = await conv_repo.read_conversation(
                conn, "work/acme", "Planning Session"
            )
        assert meta.title == "Planning Session"
        assert meta.message_count == 2
        assert meta.source == "claude"
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[1].role == "assistant"

    async def test_save_idempotent(self, clean_pool: asyncpg.Pool, tmp_journal: Path) -> None:
        async with clean_pool.acquire() as conn:
            await topic_repo.create(conn, "test/idem", "Test Idem")
        msgs_v1 = [Message(role="user", content="Q1"), Message(role="assistant", content="A1")]
        msgs_v2 = msgs_v1 + [
            Message(role="user", content="Q2"),
            Message(role="assistant", content="A2"),
        ]
        json_dir = tmp_journal / "conversations_json"
        id1, _, _2 = await conv_repo.save_conversation(
            clean_pool, json_dir, "test/idem", "Chat", msgs_v1, summary="v1"
        )
        id2, _, _3 = await conv_repo.save_conversation(
            clean_pool, json_dir, "test/idem", "Chat", msgs_v2, summary="v2"
        )

        assert id1 == id2
        async with clean_pool.acquire() as conn:
            meta, messages = await conv_repo.read_conversation(conn, "test/idem", "Chat")
        assert meta.message_count == 4
        assert len(messages) == 4

    async def test_save_preserves_created_date(
        self, clean_pool: asyncpg.Pool, tmp_journal: Path
    ) -> None:
        json_dir = tmp_journal / "conversations_json"
        async with clean_pool.acquire() as conn:
            await topic_repo.create(conn, "test/dates", "Test Dates")
        msgs = [Message(role="user", content="Hello")]
        await conv_repo.save_conversation(
            clean_pool, json_dir, "test/dates", "Chat", msgs, summary="test"
        )
        async with clean_pool.acquire() as conn:
            meta1, _ = await conv_repo.read_conversation(conn, "test/dates", "Chat")
        original_created = meta1.created

        await conv_repo.save_conversation(
            clean_pool,
            json_dir,
            "test/dates",
            "Chat",
            msgs + [Message(role="assistant", content="Hi")],
            summary="test updated",
        )
        async with clean_pool.acquire() as conn:
            meta2, _ = await conv_repo.read_conversation(conn, "test/dates", "Chat")
        assert meta2.created == original_created

    async def test_list_conversations(self, clean_pool: asyncpg.Pool, tmp_journal: Path) -> None:
        json_dir = tmp_journal / "conversations_json"
        msgs = [Message(role="user", content="Content")]
        async with clean_pool.acquire() as conn:
            await topic_repo.create(conn, "work/acme", "Acme")
            await topic_repo.create(conn, "hobbies/running", "Running")
        await conv_repo.save_conversation(
            clean_pool, json_dir, "work/acme", "Chat 1", msgs, summary="chat 1"
        )
        await conv_repo.save_conversation(
            clean_pool, json_dir, "work/acme", "Chat 2", msgs, summary="chat 2"
        )
        await conv_repo.save_conversation(
            clean_pool, json_dir, "hobbies/running", "Run Chat", msgs, summary="run chat"
        )

        async with clean_pool.acquire() as conn:
            all_convs = await conv_repo.list_conversations(conn)
            assert len(all_convs) == 3

            work_convs = await conv_repo.list_conversations(conn, topic_prefix="work")
            assert len(work_convs) == 2

    async def test_read_nonexistent_conversation_raises(self, clean_pool: asyncpg.Pool) -> None:
        async with clean_pool.acquire() as conn:
            with pytest.raises(ConversationNotFoundError):
                await conv_repo.read_conversation(conn, "test/nope", "Missing Chat")

    async def test_conversation_json_archive_written(
        self, clean_pool: asyncpg.Pool, tmp_journal: Path
    ) -> None:
        json_dir = tmp_journal / "conversations_json"
        async with clean_pool.acquire() as conn:
            await topic_repo.create(conn, "test/archive", "Test Archive")
        msgs = [Message(role="user", content="Test message")]
        conv_id, _, _2 = await conv_repo.save_conversation(
            clean_pool, json_dir, "test/archive", "Archive Test", msgs, summary="archive test"
        )

        json_files = list(json_dir.glob("*.json"))
        assert len(json_files) == 1
        data = json.loads(json_files[0].read_text())
        assert "meta" in data
        assert "messages" in data
        assert data["messages"][0]["content"] == "Test message"


class TestKnowledge:
    """Knowledge file reading (file-based, via storage/knowledge.py)."""

    def test_read_knowledge_returns_empty_if_missing(self, tmp_journal: Path) -> None:
        result = knowledge.read(tmp_journal, "nonexistent")
        assert result == ""

    def test_read_knowledge_returns_content(self, tmp_journal: Path) -> None:
        profile_path = tmp_journal / "knowledge" / "user-profile.md"
        profile_path.write_text("# User\n\nSoftware engineer.", encoding="utf-8")

        result = knowledge.read(tmp_journal, "user-profile")
        assert "Software engineer" in result


class TestStats:
    """Stats and date range queries."""

    async def test_stats_empty(self, clean_pool: asyncpg.Pool) -> None:
        async with clean_pool.acquire() as conn:
            stats = await entry_repo.get_stats(conn)
        assert stats["total_documents"] == 0
        assert stats["topics"] == 0
        assert stats["conversations"] == 0

    async def test_stats_with_data(self, clean_pool: asyncpg.Pool, tmp_journal: Path) -> None:
        msgs = [Message(role="user", content="Hello")]
        async with clean_pool.acquire() as conn:
            await topic_repo.create(conn, "work/acme", "Acme")
            await entry_repo.append(conn, "work/acme", "Entry 1.")
        await conv_repo.save_conversation(
            clean_pool,
            tmp_journal / "conversations_json",
            "work/acme",
            "Chat",
            msgs,
            summary="test",
        )
        async with clean_pool.acquire() as conn:
            stats = await entry_repo.get_stats(conn)
        assert stats["topics"] == 1
        assert stats["conversations"] == 1
        # 2 entries (1 manual + 1 auto-created by save_conversation) + 1 conversation
        assert stats["total_documents"] == 3

    async def test_entries_by_date_range(self, clean_pool: asyncpg.Pool) -> None:
        async with clean_pool.acquire() as conn:
            await topic_repo.create(conn, "work/acme", "Acme")
            await topic_repo.create(conn, "hobbies/running", "Running")
            await topic_repo.create(conn, "test/march", "March")
            await entry_repo.append(conn, "work/acme", "Jan entry.", date="2025-01-15")
            await entry_repo.append(conn, "hobbies/running", "Feb entry.", date="2025-02-10")
            await entry_repo.append(conn, "test/march", "Mar entry.", date="2025-03-05")

            results = await entry_repo.get_by_date_range(conn, "2025-01-01", "2025-01-31")
            assert len(results) == 1
            assert results[0]["topic"] == "work/acme"

            results = await entry_repo.get_by_date_range(conn, "2025-01-01", "2025-12-31")
            assert len(results) == 3
