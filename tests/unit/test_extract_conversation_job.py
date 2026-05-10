"""Tests for the extract_conversation Arq job (m-h5-h6 connection-split)."""

from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from gubbi.extraction.jobs.extract_conversation import extract_conversation
from gubbi.extraction.service import CategorizationResult, ExtractedEntry
from gubbi.storage.exceptions import TopicNotFoundError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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
    """Return a fake asyncpg connection (single-connection tests)."""
    conn = AsyncMock()
    conn.fetchval.return_value = None
    return conn


@pytest.fixture
def conn1() -> AsyncMock:
    """Phase-1 connection mock (read-only load)."""
    c = AsyncMock()
    c.fetchval.return_value = None  # not yet processed
    return c


@pytest.fixture
def conn2() -> AsyncMock:
    """Phase-3 connection mock (persistence).

    transaction() is called as an async context manager (not awaited), so it
    must return an object with __aenter__ and __aexit__.  We use a MagicMock
    for the transaction method so that conn2.transaction() returns the tx
    context-manager directly rather than wrapping it in a coroutine.
    """
    c = AsyncMock()
    c.fetchval.return_value = None  # still not processed (race guard)
    tx = MagicMock()
    tx.__aenter__ = AsyncMock(return_value=None)
    tx.__aexit__ = AsyncMock(return_value=False)
    # Use MagicMock (not AsyncMock) so the call returns tx directly.
    c.transaction = MagicMock(return_value=tx)
    return c


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_usc_side_effect(*conns: AsyncMock) -> list[AsyncMock]:
    """Build a list of context-manager mocks, one per conn, for side_effect."""
    cms = []
    for conn in conns:
        cm = AsyncMock()
        cm.__aenter__ = AsyncMock(return_value=conn)
        cm.__aexit__ = AsyncMock(return_value=False)
        cms.append(cm)
    return cms


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestExtractConversationJob:
    """Tests for the extract_conversation job function."""

    # ------------------------------------------------------------------
    # Happy path: two user_scoped_connection calls, correct order
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_extract_conversation_calls_service_in_order(
        self,
        mock_ctx: dict,
        conn1: AsyncMock,
        conn2: AsyncMock,
    ) -> None:
        """Verify the job calls ExtractionService methods in the correct
        order with the correct arguments, using two separate connections."""
        conversation_id = 42
        user_id = "00000000-0000-0000-0000-000000000001"

        with (
            patch(
                "gubbi.extraction.jobs.extract_conversation.user_scoped_connection",
            ) as mock_usc,
            patch(
                "gubbi.storage.repositories.conversations.read_conversation_by_id",
            ) as mock_read_conv,
            patch(
                "gubbi.storage.repositories.conversations.get_processed_at",
            ) as mock_get_processed_at,
            patch(
                "gubbi.storage.repositories.conversations.mark_processed",
            ) as mock_mark_processed,
            patch(
                "gubbi.storage.repositories.topics.list_all",
            ) as mock_list_topics,
            patch(
                "gubbi.storage.repositories.topics.get_id",
            ) as mock_get_topic_id,
            patch(
                "gubbi.storage.repositories.topics.create",
            ) as mock_create_topic,
            patch(
                "gubbi.storage.repositories.entries.append",
            ) as mock_entry_append,
        ):
            # get_processed_at returns None both times (conn1 check + conn2 race guard).
            mock_get_processed_at.return_value = None
            mock_usc.side_effect = _make_usc_side_effect(conn1, conn2)

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

            # --- Assert two connections were acquired ---
            assert mock_usc.call_count == 2

            # --- Assert calls in order ---
            # 1. Idempotency check (conn1 + conn2 race guard).
            assert mock_get_processed_at.await_count == 2
            mock_get_processed_at.assert_any_await(conn1, conversation_id)
            mock_get_processed_at.assert_any_await(conn2, conversation_id)

            # 2. Load conversation (conn1)
            mock_read_conv.assert_awaited_once_with(conn1, mock_ctx["cipher"], conversation_id)

            # 3. List existing topics (conn1)
            mock_list_topics.assert_awaited_once_with(conn1)

            # 4. Categorize conversation (no conn held)
            mock_ctx["extraction_service"].categorize_conversation.assert_awaited_once_with(
                [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there"},
                ],
                ["existing/topic"],
            )

            # 5. extract_entries (no conn held)
            mock_ctx["extraction_service"].extract_entries.assert_awaited_once_with(
                [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there"},
                ],
                "health/fitness",
            )

            # 6. Topic upsert (conn2)
            mock_get_topic_id.assert_awaited_once_with(conn2, "health/fitness")
            mock_create_topic.assert_awaited_once_with(
                conn2, "health/fitness", title="Fitness Routine"
            )

            # 7. Append entries (conn2)
            assert mock_entry_append.await_count == 2
            mock_entry_append.assert_has_awaits(
                [
                    call(
                        conn2,
                        mock_ctx["cipher"],
                        topic="health/fitness",
                        content="Started jogging",
                        reasoning="New habit",
                        tags=["fitness"],
                        date="2026-04-15",
                    ),
                    call(
                        conn2,
                        mock_ctx["cipher"],
                        topic="health/fitness",
                        content="Planned gym",
                        reasoning="Weekly goal",
                        tags=["gym"],
                        date="2026-04-16",
                    ),
                ]
            )

            # 8. Mark conversation processed (conn2, inside _persist_extraction)
            mock_mark_processed.assert_awaited_once_with(conn2, conversation_id)

            # 9. Redis publish (new channel format with job_id)
            expected_event = {
                "topic_path": "health/fitness",
                "entries_created": 2,
                "job_id": "unknown",
                "conversation_id": 42,
            }
            mock_ctx["redis"].publish.assert_awaited_once()
            call_args = mock_ctx["redis"].publish.await_args
            assert call_args is not None
            channel, payload = call_args.args
            assert channel == f"extraction:user:{user_id}:job:unknown"
            assert json.loads(payload) == expected_event

    # ------------------------------------------------------------------
    # Idempotent skip (conn1 check)
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_extract_conversation_idempotent_skip(
        self,
        mock_ctx: dict,
        conn1: AsyncMock,
    ) -> None:
        """When processed_at is already set on conn1 check, the job returns early with
        skipped=True and never acquires conn2 or calls ExtractionService."""
        conversation_id = 99
        user_id = "00000000-0000-0000-0000-000000000002"

        with (
            patch(
                "gubbi.extraction.jobs.extract_conversation.user_scoped_connection",
            ) as mock_usc,
            patch(
                "gubbi.storage.repositories.conversations.get_processed_at",
            ) as mock_get_processed_at,
        ):
            mock_usc.side_effect = _make_usc_side_effect(conn1)

            # Simulate already-processed conversation.
            mock_get_processed_at.return_value = datetime(2020, 1, 1)

            result = await extract_conversation(mock_ctx, conversation_id, user_id)

            # Assert early return with skipped=True
            assert result["skipped"] is True
            assert result["entries_created"] == 0
            assert result["topic_path"] is None

            # Only one connection acquired (conn1 early-exit).
            assert mock_usc.call_count == 1

            # ExtractionService should never be called.
            mock_ctx["extraction_service"].categorize_conversation.assert_not_called()
            mock_ctx["extraction_service"].extract_entries.assert_not_called()

            # Redis publish should never be called.
            mock_ctx["redis"].publish.assert_not_called()

            # Only the idempotency check via repo should have been called.
            mock_get_processed_at.assert_awaited_once_with(conn1, conversation_id)

    # ------------------------------------------------------------------
    # Redis publish
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_extract_conversation_publishes_redis_event(
        self,
        mock_ctx: dict,
        conn1: AsyncMock,
        conn2: AsyncMock,
    ) -> None:
        """After successful extraction, a Redis event is published on the
        correct channel with the expected JSON payload."""
        conversation_id = 7
        user_id = "00000000-0000-0000-0000-000000000003"

        with (
            patch(
                "gubbi.extraction.jobs.extract_conversation.user_scoped_connection",
            ) as mock_usc,
            patch(
                "gubbi.storage.repositories.conversations.read_conversation_by_id",
            ) as mock_read_conv,
            patch(
                "gubbi.storage.repositories.conversations.get_processed_at",
            ) as mock_get_processed_at,
            patch(
                "gubbi.storage.repositories.topics.list_all",
            ) as mock_list_topics,
            patch(
                "gubbi.storage.repositories.topics.get_id",
            ) as mock_get_topic_id,
            patch(
                "gubbi.storage.repositories.entries.append",
            ),
            patch(
                "gubbi.storage.repositories.conversations.mark_processed",
            ),
        ):
            mock_get_processed_at.return_value = None
            mock_usc.side_effect = _make_usc_side_effect(conn1, conn2)

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
            expected_event = {
                "topic_path": "work/dev",
                "entries_created": 1,
                "job_id": "unknown",
                "conversation_id": 7,
            }
            mock_ctx["redis"].publish.assert_awaited_once()
            call_args = mock_ctx["redis"].publish.await_args
            assert call_args is not None
            channel, payload = call_args.args
            assert channel == f"extraction:user:{user_id}:job:unknown"
            assert json.loads(payload) == expected_event

    # ------------------------------------------------------------------
    # New resilience tests (m-h5-h6)
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_two_user_scoped_connection_calls_on_happy_path(
        self,
        mock_ctx: dict,
        conn1: AsyncMock,
        conn2: AsyncMock,
    ) -> None:
        """Happy path acquires exactly two user_scoped_connection instances."""
        conversation_id = 55
        user_id = "00000000-0000-0000-0000-000000000010"

        with (
            patch(
                "gubbi.extraction.jobs.extract_conversation.user_scoped_connection",
            ) as mock_usc,
            patch("gubbi.storage.repositories.conversations.read_conversation_by_id") as mock_rcbi,
            patch("gubbi.storage.repositories.conversations.get_processed_at") as mock_gpa,
            patch("gubbi.storage.repositories.conversations.mark_processed"),
            patch("gubbi.storage.repositories.topics.list_all") as mock_la,
            patch("gubbi.storage.repositories.topics.get_id") as mock_gti,
            patch("gubbi.storage.repositories.entries.append"),
        ):
            mock_gpa.return_value = None
            mock_usc.side_effect = _make_usc_side_effect(conn1, conn2)

            fake_meta = MagicMock()
            fake_messages = [MagicMock(role="user", content="msg")]
            mock_rcbi.return_value = (fake_meta, fake_messages, 1)
            mock_la.return_value = ([], 0)
            mock_gti.return_value = 1

            mock_ctx[
                "extraction_service"
            ].categorize_conversation.return_value = CategorizationResult(
                topic_path="test/path",
                topic_title="Test",
                summary="s",
                confidence=0.9,
            )
            mock_ctx["extraction_service"].extract_entries.return_value = [
                ExtractedEntry(content="c", reasoning=None, tags=[], entry_date="2026-01-01")
            ]

            result = await extract_conversation(mock_ctx, conversation_id, user_id)

            assert result["skipped"] is False
            assert mock_usc.call_count == 2

    @pytest.mark.asyncio
    async def test_llm_failure_releases_conn1_before_raise(
        self,
        mock_ctx: dict,
        conn1: AsyncMock,
    ) -> None:
        """When categorize_conversation raises, conn1 has already been released
        (its context manager exited cleanly) and conn2 is never acquired."""
        conversation_id = 77
        user_id = "00000000-0000-0000-0000-000000000011"

        with (
            patch(
                "gubbi.extraction.jobs.extract_conversation.user_scoped_connection",
            ) as mock_usc,
            patch("gubbi.storage.repositories.conversations.read_conversation_by_id") as mock_rcbi,
            patch("gubbi.storage.repositories.conversations.get_processed_at") as mock_gpa,
            patch("gubbi.storage.repositories.topics.list_all") as mock_la,
            patch("gubbi.storage.repositories.conversations.mark_processed") as mock_mp,
        ):
            mock_gpa.return_value = None
            cm1 = AsyncMock()
            cm1.__aenter__ = AsyncMock(return_value=conn1)
            cm1.__aexit__ = AsyncMock(return_value=False)
            mock_usc.side_effect = [cm1]

            fake_meta = MagicMock()
            fake_messages = [MagicMock(role="user", content="msg")]
            mock_rcbi.return_value = (fake_meta, fake_messages, 1)
            mock_la.return_value = ([], 0)

            mock_ctx["extraction_service"].categorize_conversation.side_effect = RuntimeError(
                "LLM down"
            )

            with pytest.raises(RuntimeError, match="LLM down"):
                await extract_conversation(mock_ctx, conversation_id, user_id)

            # conn1 context manager __aexit__ called cleanly (released).
            cm1.__aexit__.assert_awaited_once()

            # Only one USC call -- conn2 never acquired.
            assert mock_usc.call_count == 1

            # mark_processed never called.
            mock_mp.assert_not_called()

    @pytest.mark.asyncio
    async def test_extract_entries_failure_releases_conn1_no_conn2(
        self,
        mock_ctx: dict,
        conn1: AsyncMock,
    ) -> None:
        """When extract_entries raises (Phase 2), conn1 has already been released
        and conn2 is never acquired."""
        conversation_id = 88
        user_id = "00000000-0000-0000-0000-000000000012"

        with (
            patch(
                "gubbi.extraction.jobs.extract_conversation.user_scoped_connection",
            ) as mock_usc,
            patch("gubbi.storage.repositories.conversations.read_conversation_by_id") as mock_rcbi,
            patch("gubbi.storage.repositories.conversations.get_processed_at") as mock_gpa,
            patch("gubbi.storage.repositories.topics.list_all") as mock_la,
            patch("gubbi.storage.repositories.conversations.mark_processed") as mock_mp,
        ):
            mock_gpa.return_value = None
            cm1 = AsyncMock()
            cm1.__aenter__ = AsyncMock(return_value=conn1)
            cm1.__aexit__ = AsyncMock(return_value=False)
            mock_usc.side_effect = [cm1]

            fake_meta = MagicMock()
            fake_messages = [MagicMock(role="user", content="msg")]
            mock_rcbi.return_value = (fake_meta, fake_messages, 1)
            mock_la.return_value = ([], 0)

            mock_ctx[
                "extraction_service"
            ].categorize_conversation.return_value = CategorizationResult(
                topic_path="test/path",
                topic_title="Test",
                summary="s",
                confidence=0.9,
            )
            mock_ctx["extraction_service"].extract_entries.side_effect = RuntimeError(
                "LLM timed out"
            )

            with pytest.raises(RuntimeError, match="LLM timed out"):
                await extract_conversation(mock_ctx, conversation_id, user_id)

            # conn1 exited before extract_entries was called.
            cm1.__aexit__.assert_awaited_once()

            # Only one USC call -- conn2 never acquired.
            assert mock_usc.call_count == 1

            mock_mp.assert_not_called()

    @pytest.mark.asyncio
    async def test_persistence_failure_rolls_back_entries(
        self,
        mock_ctx: dict,
        conn1: AsyncMock,
        conn2: AsyncMock,
    ) -> None:
        """When entry_repo.append raises on second entry, mark_processed is never
        called (the SAVEPOINT transaction __aexit__ is called with the exception).
        """
        conversation_id = 99
        user_id = "00000000-0000-0000-0000-000000000013"

        with (
            patch(
                "gubbi.extraction.jobs.extract_conversation.user_scoped_connection",
            ) as mock_usc,
            patch("gubbi.storage.repositories.conversations.read_conversation_by_id") as mock_rcbi,
            patch("gubbi.storage.repositories.conversations.get_processed_at") as mock_gpa,
            patch("gubbi.storage.repositories.topics.list_all") as mock_la,
            patch("gubbi.storage.repositories.topics.get_id") as mock_gti,
            patch("gubbi.storage.repositories.entries.append") as mock_append,
            patch("gubbi.storage.repositories.conversations.mark_processed") as mock_mp,
        ):
            mock_gpa.return_value = None
            mock_usc.side_effect = _make_usc_side_effect(conn1, conn2)

            fake_meta = MagicMock()
            fake_messages = [MagicMock(role="user", content="msg")]
            mock_rcbi.return_value = (fake_meta, fake_messages, 1)
            mock_la.return_value = ([], 0)
            mock_gti.return_value = 1

            mock_ctx[
                "extraction_service"
            ].categorize_conversation.return_value = CategorizationResult(
                topic_path="test/path",
                topic_title="Test",
                summary="s",
                confidence=0.9,
            )
            mock_ctx["extraction_service"].extract_entries.return_value = [
                ExtractedEntry(content="e1", reasoning=None, tags=[], entry_date="2026-01-01"),
                ExtractedEntry(content="e2", reasoning=None, tags=[], entry_date="2026-01-02"),
            ]

            # Fail on second append.
            mock_append.side_effect = [None, RuntimeError("DB write error")]

            with pytest.raises(RuntimeError, match="DB write error"):
                await extract_conversation(mock_ctx, conversation_id, user_id)

            # mark_processed MUST NOT have been called (transaction rolled back).
            mock_mp.assert_not_called()

            # Transaction context manager __aexit__ was called with the exception
            # (conn2.transaction().__aexit__ was invoked).
            conn2.transaction.return_value.__aexit__.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_skipped_no_topic_uses_dedicated_short_conn(
        self,
        mock_ctx: dict,
        conn1: AsyncMock,
    ) -> None:
        """When topic_path is None (hardening rejected LLM output), the job
        opens a short dedicated connection to mark_processed, returning skipped=True.
        That short connection counts as USC call #2; conn2 (Phase 3) is NOT acquired.
        """
        conversation_id = 33
        user_id = "00000000-0000-0000-0000-000000000014"

        skip_conn = AsyncMock()
        skip_conn.fetchval.return_value = None

        with (
            patch(
                "gubbi.extraction.jobs.extract_conversation.user_scoped_connection",
            ) as mock_usc,
            patch("gubbi.storage.repositories.conversations.read_conversation_by_id") as mock_rcbi,
            patch("gubbi.storage.repositories.conversations.get_processed_at") as mock_gpa,
            patch("gubbi.storage.repositories.conversations.mark_processed") as mock_mp,
            patch("gubbi.storage.repositories.topics.list_all") as mock_la,
        ):
            mock_gpa.return_value = None
            mock_usc.side_effect = _make_usc_side_effect(conn1, skip_conn)

            fake_meta = MagicMock()
            fake_messages = [MagicMock(role="user", content="msg")]
            mock_rcbi.return_value = (fake_meta, fake_messages, 1)
            mock_la.return_value = ([], 0)

            # Return a CategorizationResult with a topic_path that will be
            # rejected by harden_llm_topic_path (empty string -> None).
            mock_ctx[
                "extraction_service"
            ].categorize_conversation.return_value = CategorizationResult(
                topic_path="",  # harden_llm_topic_path("") returns None
                topic_title="",
                summary="",
                confidence=0.1,
            )

            result = await extract_conversation(mock_ctx, conversation_id, user_id)

            assert result["skipped"] is True
            assert result["topic_path"] is None

            # Exactly two USC calls: conn1 (Phase 1) + skip_conn (no-topic path).
            assert mock_usc.call_count == 2

            # mark_processed called once (via the skip_conn).
            mock_mp.assert_awaited_once_with(skip_conn, conversation_id)

            # extract_entries never called.
            mock_ctx["extraction_service"].extract_entries.assert_not_called()
