"""Tests for the extract_conversation Arq job."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from journalctl.extraction.jobs.extract_conversation import extract_conversation
from journalctl.extraction.service import CategorizationResult, ExtractedEntry
from journalctl.storage.exceptions import TopicNotFoundError


@pytest.fixture
def mock_ctx() -> dict:
    """Build a minimal Arq worker context with mocked dependencies."""
    return {
        "pool": MagicMock(),
        "cipher": MagicMock(),
        "extraction_service": AsyncMock(),
        "redis": AsyncMock(),
    }


@pytest.fixture
def mock_conn() -> AsyncMock:
    """Return a fake asyncpg connection."""
    conn = AsyncMock()
    # By default: conversation is NOT already processed.
    conn.fetchval.return_value = None
    return conn


class TestExtractConversationJob:
    """Tests for the extract_conversation job function."""

    # ------------------------------------------------------------------
    # Happy path: calls service in correct order
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_extract_conversation_calls_service_in_order(
        self,
        mock_ctx: dict,
        mock_conn: AsyncMock,
    ) -> None:
        """Verify the job calls ExtractionService methods in the correct
        order with the correct arguments."""
        conversation_id = 42
        user_id = "00000000-0000-0000-0000-000000000001"

        # Mock user_scoped_connection context manager.
        with (
            patch(
                "journalctl.extraction.jobs.extract_conversation.user_scoped_connection",
            ) as mock_usc,
            patch(
                "journalctl.storage.repositories.conversations.read_conversation_by_id",
            ) as mock_read_conv,
            patch(
                "journalctl.storage.repositories.topics.list_all",
            ) as mock_list_topics,
            patch(
                "journalctl.storage.repositories.topics.get_id",
            ) as mock_get_topic_id,
            patch(
                "journalctl.storage.repositories.topics.create",
            ) as mock_create_topic,
            patch(
                "journalctl.storage.repositories.entries.append",
            ) as mock_entry_append,
        ):
            mock_usc.return_value.__aenter__.return_value = mock_conn

            # --- Fake conversation data ---
            fake_meta = MagicMock()
            fake_messages = [
                MagicMock(role="user", content="Hello"),
                MagicMock(role="assistant", content="Hi there"),
            ]
            mock_read_conv.return_value = (fake_meta, fake_messages, 2)

            # --- Existing topics ---
            existing_topic_meta = MagicMock()
            existing_topic_meta.topic = "existing/topic"
            mock_list_topics.return_value = ([existing_topic_meta], 1)

            # --- Categorization ---
            mock_categorization = CategorizationResult(
                topic_path="health/fitness",
                topic_title="Fitness Routine",
                summary="User discussed workout",
                confidence=0.95,
            )
            mock_ctx[
                "extraction_service"
            ].categorize_conversation.return_value = mock_categorization

            # --- Topic upsert: first get_id raises (topic does not exist),
            #     then create succeeds ---
            mock_get_topic_id.side_effect = TopicNotFoundError("not found")

            # --- Extracted entries ---
            fake_entries = [
                ExtractedEntry(
                    content="Started jogging",
                    reasoning="New habit",
                    tags=["fitness"],
                    entry_date="2026-04-15",
                ),
                ExtractedEntry(
                    content="Planned gym",
                    reasoning="Weekly goal",
                    tags=["gym"],
                    entry_date="2026-04-16",
                ),
            ]
            mock_ctx["extraction_service"].extract_entries.return_value = fake_entries

            # --- Execute ---
            result = await extract_conversation(mock_ctx, conversation_id, user_id)

            # --- Assert result ---
            assert result["topic_path"] == "health/fitness"
            assert result["entries_created"] == 2
            assert result["input_tokens"] == 0
            assert result["output_tokens"] == 0
            assert result["skipped"] is False

            # --- Assert calls in order ---
            # 1. Idempotency check
            mock_conn.fetchval.assert_any_call(
                "SELECT processed_at FROM conversations WHERE id = $1",
                conversation_id,
            )

            # 2. Load conversation
            mock_read_conv.assert_awaited_once_with(mock_conn, mock_ctx["cipher"], conversation_id)

            # 3. List existing topics
            mock_list_topics.assert_awaited_once_with(mock_conn)

            # 4. Categorize conversation
            mock_ctx["extraction_service"].categorize_conversation.assert_awaited_once_with(
                [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there"},
                ],
                ["existing/topic"],
            )

            # 5. Topic not found -> create topic
            mock_get_topic_id.assert_awaited_once_with(mock_conn, "health/fitness")
            mock_create_topic.assert_awaited_once_with(
                mock_conn, "health/fitness", title="Fitness Routine"
            )

            # 6. Extract entries
            mock_ctx["extraction_service"].extract_entries.assert_awaited_once_with(
                [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there"},
                ],
                "health/fitness",
            )

            # 7. Append entries
            assert mock_entry_append.await_count == 2
            mock_entry_append.assert_has_awaits(
                [
                    call(
                        mock_conn,
                        mock_ctx["cipher"],
                        topic="health/fitness",
                        content="Started jogging",
                        reasoning="New habit",
                        tags=["fitness"],
                        date="2026-04-15",
                    ),
                    call(
                        mock_conn,
                        mock_ctx["cipher"],
                        topic="health/fitness",
                        content="Planned gym",
                        reasoning="Weekly goal",
                        tags=["gym"],
                        date="2026-04-16",
                    ),
                ]
            )

            # 8. Mark conversation processed
            mock_conn.execute.assert_any_call(
                "UPDATE conversations SET processed_at = now() WHERE id = $1",
                conversation_id,
            )

            # 9. Redis publish
            mock_ctx["redis"].publish.assert_awaited_once_with(
                f"extraction:{user_id}",
                '{"topic_path": "health/fitness", "entries_created": 2}',
            )

    # ------------------------------------------------------------------
    # Idempotent skip
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_extract_conversation_idempotent_skip(
        self,
        mock_ctx: dict,
        mock_conn: AsyncMock,
    ) -> None:
        """When processed_at is already set, the job returns early with
        skipped=True and never calls ExtractionService."""
        conversation_id = 99
        user_id = "00000000-0000-0000-0000-000000000002"

        # Simulate already-processed conversation.
        from datetime import UTC, datetime

        mock_conn.fetchval.return_value = datetime.now(UTC)

        with patch(
            "journalctl.extraction.jobs.extract_conversation.user_scoped_connection",
        ) as mock_usc:
            mock_usc.return_value.__aenter__.return_value = mock_conn

            result = await extract_conversation(mock_ctx, conversation_id, user_id)

            # Assert early return with skipped=True
            assert result["skipped"] is True
            assert result["entries_created"] == 0
            assert result["topic_path"] is None

            # ExtractionService should never be called.
            mock_ctx["extraction_service"].categorize_conversation.assert_not_called()
            mock_ctx["extraction_service"].extract_entries.assert_not_called()

            # Redis publish should never be called.
            mock_ctx["redis"].publish.assert_not_called()

            # Only the idempotency check fetchval should have been called.
            mock_conn.fetchval.assert_awaited_once_with(
                "SELECT processed_at FROM conversations WHERE id = $1",
                conversation_id,
            )

    # ------------------------------------------------------------------
    # Redis publish
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_extract_conversation_publishes_redis_event(
        self,
        mock_ctx: dict,
        mock_conn: AsyncMock,
    ) -> None:
        """After successful extraction, a Redis event is published on the
        correct channel with the expected JSON payload."""
        conversation_id = 7
        user_id = "00000000-0000-0000-0000-000000000003"

        with (
            patch(
                "journalctl.extraction.jobs.extract_conversation.user_scoped_connection",
            ) as mock_usc,
            patch(
                "journalctl.storage.repositories.conversations.read_conversation_by_id",
            ) as mock_read_conv,
            patch(
                "journalctl.storage.repositories.topics.list_all",
            ) as mock_list_topics,
            patch(
                "journalctl.storage.repositories.topics.get_id",
            ) as mock_get_topic_id,
            patch(
                "journalctl.storage.repositories.entries.append",
            ),
        ):
            mock_usc.return_value.__aenter__.return_value = mock_conn

            # Minimal stubs to get the job to complete.
            fake_meta = MagicMock()
            fake_messages = [MagicMock(role="user", content="Test message")]
            mock_read_conv.return_value = (fake_meta, fake_messages, 1)

            existing_meta = MagicMock()
            existing_meta.topic = "existing/topic"
            mock_list_topics.return_value = ([existing_meta], 1)

            mock_categorization = CategorizationResult(
                topic_path="work/dev",
                topic_title="Dev Work",
                summary="Discussed coding",
                confidence=0.88,
            )
            mock_ctx[
                "extraction_service"
            ].categorize_conversation.return_value = mock_categorization

            # Topic already exists -- get_id succeeds, create not called.
            mock_get_topic_id.return_value = 42

            fake_entry = ExtractedEntry(
                content="Fixed a bug",
                reasoning="Debugging session",
                tags=["coding"],
                entry_date="2026-05-01",
            )
            mock_ctx["extraction_service"].extract_entries.return_value = [fake_entry]

            await extract_conversation(mock_ctx, conversation_id, user_id)

            # Verify Redis publish was called with correct channel and JSON.
            mock_ctx["redis"].publish.assert_awaited_once_with(
                f"extraction:{user_id}",
                '{"topic_path": "work/dev", "entries_created": 1}',
            )
