"""Tests for the extract_conversation Arq job (m-h5-h6 connection-split)."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from gubbi.extraction.jobs.extract_conversation import _classify_error, extract_conversation
from gubbi.extraction.service import CategorizationResult, ExtractedEntry, ExtractionEntriesResult
from gubbi.storage.exceptions import TopicNotFoundError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_ctx() -> Any:
    """Build a minimal Arq worker context with mocked dependencies."""
    extraction_service = AsyncMock()
    extraction_service._llm = None
    return {
        "pool": MagicMock(),
        "cipher": MagicMock(),
        "extraction_service": extraction_service,
        "redis": AsyncMock(),
        "redis_pool": MagicMock(),
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


def _make_usc_side_effect(*conns: AsyncMock) -> list[MagicMock]:
    """Build a list of context-manager mocks, one per conn, for side_effect."""
    cms: list[MagicMock] = []
    for conn in conns:
        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=conn)
        cm.__aexit__ = AsyncMock(return_value=False)
        cms.append(cm)
    return cms


def _make_entries_result(
    entries: list[ExtractedEntry], input_tokens: int = 0, output_tokens: int = 0
) -> ExtractionEntriesResult:
    """Build an ExtractionEntriesResult for use in test mocks."""
    return ExtractionEntriesResult(
        entries=tuple(entries),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


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
        mock_ctx: Any,
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
            mock_ctx["extraction_service"].extract_entries.return_value = _make_entries_result(
                fake_entries
            )

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
        mock_ctx: Any,
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
        mock_ctx: Any,
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
            mock_ctx["extraction_service"].extract_entries.return_value = _make_entries_result(
                [fake_entry]
            )

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
        mock_ctx: Any,
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
            mock_ctx["extraction_service"].extract_entries.return_value = _make_entries_result(
                [ExtractedEntry(content="c", reasoning="", tags=[], entry_date="2026-01-01")]
            )

            result = await extract_conversation(mock_ctx, conversation_id, user_id)

            assert result["skipped"] is False
            assert mock_usc.call_count == 2

    @pytest.mark.asyncio
    async def test_llm_failure_releases_conn1_before_raise(
        self,
        mock_ctx: Any,
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
        mock_ctx: Any,
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
        mock_ctx: Any,
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
            mock_ctx["extraction_service"].extract_entries.return_value = _make_entries_result(
                [
                    ExtractedEntry(content="e1", reasoning="", tags=[], entry_date="2026-01-01"),
                    ExtractedEntry(content="e2", reasoning="", tags=[], entry_date="2026-01-02"),
                ]
            )

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
        mock_ctx: Any,
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

        mock_jobs = MagicMock()
        mock_jobs.mark_running = AsyncMock()
        mock_jobs.mark_completed = AsyncMock(return_value=True)
        mock_jobs.get_period_start = AsyncMock(return_value=None)

        with (
            patch(
                "gubbi.extraction.jobs.extract_conversation.user_scoped_connection",
            ) as mock_usc,
            patch("gubbi.storage.repositories.conversations.read_conversation_by_id") as mock_rcbi,
            patch("gubbi.storage.repositories.conversations.get_processed_at") as mock_gpa,
            patch("gubbi.storage.repositories.conversations.mark_processed") as mock_mp,
            patch("gubbi.storage.repositories.topics.list_all") as mock_la,
            patch("gubbi.extraction.jobs.extract_conversation.extraction_jobs", mock_jobs),
            patch(
                "gubbi.extraction.jobs.extract_conversation.record_audit", new=AsyncMock()
            ) as mock_audit,
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

            job_id = "cccccccc-0000-0000-0000-000000000014"
            result = await extract_conversation(mock_ctx, conversation_id, user_id, job_id)

            assert result["skipped"] is True
            assert result["topic_path"] is None

            # Exactly two USC calls: conn1 (Phase 1) + skip_conn (no-topic path).
            assert mock_usc.call_count == 2

            # mark_processed called once (via the skip_conn).
            mock_mp.assert_awaited_once_with(skip_conn, conversation_id)
            mock_jobs.mark_completed.assert_awaited_once()
            completed_call = mock_jobs.mark_completed.await_args
            assert completed_call is not None
            assert completed_call.kwargs["topics_created"] == 0
            assert completed_call.kwargs["entries_created"] == 0
            assert completed_call.kwargs["cents_spent"] == 0
            mock_audit.assert_awaited_once()

            # extract_entries never called.
            mock_ctx["extraction_service"].extract_entries.assert_not_called()


# ---------------------------------------------------------------------------
# Unit tests for _classify_error
# ---------------------------------------------------------------------------


class TestClassifyError:
    """Unit tests for the _classify_error private helper.

    After B2 the worker reads the provider-agnostic LLM* hierarchy from
    gubbi.extraction.llm.provider; vendor SDK types are translated at the
    AnthropicProvider boundary, never imported here.
    """

    def test_rate_limit_error(self) -> None:
        from gubbi.extraction.llm.provider import LLMRateLimitError

        exc = LLMRateLimitError("rate limited")
        assert _classify_error(exc) == "llm_rate_limited"

    def test_api_error(self) -> None:
        from gubbi.extraction.llm.provider import LLMProviderError

        exc = LLMProviderError("provider error")
        assert _classify_error(exc) == "llm_provider_error"

    def test_permanent_error_classified_as_provider_error(self) -> None:
        from gubbi.extraction.llm.provider import LLMPermanentError

        exc = LLMPermanentError("auth failed")
        assert _classify_error(exc) == "llm_provider_error"

    def test_transient_error_classified_as_provider_error(self) -> None:
        from gubbi.extraction.llm.provider import LLMTransientError

        exc = LLMTransientError("connection reset")
        assert _classify_error(exc) == "llm_provider_error"

    def test_runtime_error_falls_back_to_internal(self) -> None:
        exc = RuntimeError("something unexpected")
        assert _classify_error(exc) == "internal_error"

    def test_value_error_falls_back_to_internal(self) -> None:
        exc = ValueError("bad data")
        assert _classify_error(exc) == "internal_error"

    def test_exception_falls_back_to_internal(self) -> None:
        exc = Exception("generic")
        assert _classify_error(exc) == "internal_error"


# ---------------------------------------------------------------------------
# Unit tests for lifecycle UPDATEs (mark_running / mark_failed wiring)
# ---------------------------------------------------------------------------


class TestLifecycleUpdates:
    """Verify that mark_running is called in Phase 1 and mark_failed is called
    on exception, both using the correct connections."""

    @pytest.mark.asyncio
    async def test_mark_running_called_after_idempotency_check(
        self,
        mock_ctx: Any,
        conn1: AsyncMock,
        conn2: AsyncMock,
    ) -> None:
        """mark_running is called on conn1 when job_id is provided and
        the conversation has not been processed yet."""
        conversation_id = 201
        user_id = "00000000-0000-0000-0000-000000000201"
        job_id = "aaaaaaaa-0000-0000-0000-000000000001"

        mock_jobs = MagicMock()
        mock_jobs.mark_running = AsyncMock()
        mock_jobs.mark_completed = AsyncMock()
        mock_jobs.mark_failed = AsyncMock()
        mock_jobs.get_period_start = AsyncMock(return_value=None)

        with (
            patch("gubbi.extraction.jobs.extract_conversation.user_scoped_connection") as mock_usc,
            patch("gubbi.storage.repositories.conversations.read_conversation_by_id") as mock_rcbi,
            patch("gubbi.storage.repositories.conversations.get_processed_at") as mock_gpa,
            patch("gubbi.storage.repositories.conversations.mark_processed"),
            patch("gubbi.storage.repositories.topics.list_all") as mock_la,
            patch("gubbi.storage.repositories.topics.get_id") as mock_gti,
            patch("gubbi.storage.repositories.entries.append"),
            patch("gubbi.extraction.jobs.extract_conversation.extraction_jobs", mock_jobs),
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
                topic_path="test/lifecycle",
                topic_title="Lifecycle",
                summary="s",
                confidence=0.9,
            )
            mock_ctx["extraction_service"].extract_entries.return_value = _make_entries_result(
                [ExtractedEntry(content="e", reasoning="", tags=[], entry_date="2026-01-01")]
            )

            await extract_conversation(mock_ctx, conversation_id, user_id, job_id)

            # mark_running must have been called with conn1 and the UUID.
            mock_jobs.mark_running.assert_awaited_once()
            call_args = mock_jobs.mark_running.await_args
            assert call_args is not None
            assert call_args.args[0] is conn1

    @pytest.mark.asyncio
    async def test_mark_failed_called_on_llm_error_with_job_id(
        self,
        mock_ctx: Any,
        conn1: AsyncMock,
    ) -> None:
        """When extract_entries raises and job_id is given, mark_failed is
        called on a FRESH connection (not conn1 or conn2)."""
        conversation_id = 202
        user_id = "00000000-0000-0000-0000-000000000202"
        job_id = "bbbbbbbb-0000-0000-0000-000000000002"

        failure_conn = AsyncMock()

        mock_jobs = MagicMock()
        mock_jobs.mark_running = AsyncMock()
        mock_jobs.mark_failed = AsyncMock(return_value=True)
        mock_jobs.get_period_start = AsyncMock(return_value=None)

        with (
            patch("gubbi.extraction.jobs.extract_conversation.user_scoped_connection") as mock_usc,
            patch("gubbi.storage.repositories.conversations.read_conversation_by_id") as mock_rcbi,
            patch("gubbi.storage.repositories.conversations.get_processed_at") as mock_gpa,
            patch("gubbi.storage.repositories.topics.list_all") as mock_la,
            patch("gubbi.extraction.jobs.extract_conversation.extraction_jobs", mock_jobs),
        ):
            mock_gpa.return_value = None
            # conn1 for Phase 1, failure_conn for _mark_job_failed.
            mock_usc.side_effect = _make_usc_side_effect(conn1, failure_conn)

            fake_meta = MagicMock()
            fake_messages = [MagicMock(role="user", content="msg")]
            mock_rcbi.return_value = (fake_meta, fake_messages, 1)
            mock_la.return_value = ([], 0)

            mock_ctx[
                "extraction_service"
            ].categorize_conversation.return_value = CategorizationResult(
                topic_path="test/lifecycle",
                topic_title="Lifecycle",
                summary="s",
                confidence=0.9,
            )
            mock_ctx["extraction_service"].extract_entries.side_effect = RuntimeError("LLM down")

            with pytest.raises(RuntimeError, match="LLM down"):
                await extract_conversation(mock_ctx, conversation_id, user_id, job_id)

            # mark_failed must have been called.
            mock_jobs.mark_failed.assert_awaited_once()
            call_args = mock_jobs.mark_failed.await_args
            assert call_args is not None
            # Called with failure_conn (the fresh connection, not conn1).
            assert call_args.args[0] is failure_conn
            # error_code kwarg should be populated (internal_error for RuntimeError).
            assert call_args.kwargs.get("error_code") == "internal_error"

    @pytest.mark.asyncio
    async def test_original_exception_propagates_when_mark_failed_connection_fails(
        self,
        mock_ctx: Any,
        conn1: AsyncMock,
    ) -> None:
        """Secondary failure in _mark_job_failed must not mask the original error."""
        conversation_id = 205
        user_id = "00000000-0000-0000-0000-000000000205"
        job_id = "bbbbbbbb-0000-0000-0000-000000000005"

        mock_jobs = MagicMock()
        mock_jobs.mark_running = AsyncMock()
        mock_jobs.mark_failed = AsyncMock()
        mock_jobs.get_period_start = AsyncMock(return_value=None)

        failure_cm = MagicMock()
        failure_cm.__aenter__.side_effect = RuntimeError("fresh connection failed")
        failure_cm.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("gubbi.extraction.jobs.extract_conversation.user_scoped_connection") as mock_usc,
            patch("gubbi.storage.repositories.conversations.read_conversation_by_id") as mock_rcbi,
            patch("gubbi.storage.repositories.conversations.get_processed_at") as mock_gpa,
            patch("gubbi.storage.repositories.topics.list_all") as mock_la,
            patch("gubbi.extraction.jobs.extract_conversation.extraction_jobs", mock_jobs),
            patch("gubbi.extraction.jobs.extract_conversation.logger.warning", new=AsyncMock()),
        ):
            mock_gpa.return_value = None
            mock_usc.side_effect = [*_make_usc_side_effect(conn1), failure_cm]

            fake_meta = MagicMock()
            fake_messages = [MagicMock(role="user", content="msg")]
            mock_rcbi.return_value = (fake_meta, fake_messages, 1)
            mock_la.return_value = ([], 0)
            mock_ctx[
                "extraction_service"
            ].categorize_conversation.return_value = CategorizationResult(
                topic_path="test/lifecycle",
                topic_title="Lifecycle",
                summary="s",
                confidence=0.9,
            )
            mock_ctx["extraction_service"].extract_entries.side_effect = RuntimeError("LLM down")

            with pytest.raises(RuntimeError, match="LLM down"):
                await extract_conversation(mock_ctx, conversation_id, user_id, job_id)

    @pytest.mark.asyncio
    async def test_mark_failed_not_called_without_job_id(
        self,
        mock_ctx: Any,
        conn1: AsyncMock,
    ) -> None:
        """When no job_id is provided (default 'unknown'), mark_failed is
        never called even if an exception occurs."""
        conversation_id = 203
        user_id = "00000000-0000-0000-0000-000000000203"

        mock_jobs = MagicMock()
        mock_jobs.mark_running = AsyncMock()
        mock_jobs.mark_failed = AsyncMock(return_value=True)
        mock_jobs.get_period_start = AsyncMock(return_value=None)

        with (
            patch("gubbi.extraction.jobs.extract_conversation.user_scoped_connection") as mock_usc,
            patch("gubbi.storage.repositories.conversations.read_conversation_by_id") as mock_rcbi,
            patch("gubbi.storage.repositories.conversations.get_processed_at") as mock_gpa,
            patch("gubbi.storage.repositories.topics.list_all") as mock_la,
            patch("gubbi.extraction.jobs.extract_conversation.extraction_jobs", mock_jobs),
        ):
            mock_gpa.return_value = None
            mock_usc.side_effect = _make_usc_side_effect(conn1)

            fake_meta = MagicMock()
            fake_messages = [MagicMock(role="user", content="msg")]
            mock_rcbi.return_value = (fake_meta, fake_messages, 1)
            mock_la.return_value = ([], 0)

            mock_ctx["extraction_service"].categorize_conversation.side_effect = RuntimeError(
                "fail"
            )

            with pytest.raises(RuntimeError, match="fail"):
                await extract_conversation(mock_ctx, conversation_id, user_id)

            mock_jobs.mark_failed.assert_not_called()

    @pytest.mark.asyncio
    async def test_result_includes_cents_spent(
        self,
        mock_ctx: Any,
        conn1: AsyncMock,
        conn2: AsyncMock,
    ) -> None:
        """Result dict includes cents_spent derived from token counts."""
        conversation_id = 204
        user_id = "00000000-0000-0000-0000-000000000204"

        # Give the mock LLM provider an estimate_cost_cents method.
        mock_provider = MagicMock()
        mock_provider.estimate_cost_cents.return_value = 5.0  # will be rounded to int
        mock_ctx["extraction_service"]._llm = mock_provider

        mock_jobs = MagicMock()
        mock_jobs.mark_running = AsyncMock()
        mock_jobs.mark_completed = AsyncMock(return_value=True)
        mock_jobs.mark_failed = AsyncMock()
        mock_jobs.get_period_start = AsyncMock(return_value=None)

        with (
            patch("gubbi.extraction.jobs.extract_conversation.user_scoped_connection") as mock_usc,
            patch("gubbi.storage.repositories.conversations.read_conversation_by_id") as mock_rcbi,
            patch("gubbi.storage.repositories.conversations.get_processed_at") as mock_gpa,
            patch("gubbi.storage.repositories.conversations.mark_processed"),
            patch("gubbi.storage.repositories.topics.list_all") as mock_la,
            patch("gubbi.storage.repositories.topics.get_id") as mock_gti,
            patch("gubbi.storage.repositories.entries.append"),
            patch("gubbi.extraction.jobs.extract_conversation.extraction_jobs", mock_jobs),
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
                topic_path="test/cost",
                topic_title="Cost Test",
                summary="s",
                confidence=0.9,
                input_tokens=100,
                output_tokens=50,
            )
            mock_ctx["extraction_service"].extract_entries.return_value = _make_entries_result(
                [ExtractedEntry(content="e", reasoning="", tags=[], entry_date="2026-01-01")],
                input_tokens=200,
                output_tokens=80,
            )

            result = await extract_conversation(mock_ctx, conversation_id, user_id)

            # estimate_cost_cents should be called with sum of tokens.
            mock_provider.estimate_cost_cents.assert_called_once_with(300, 130)
            assert result["cents_spent"] == 5
            assert result["input_tokens"] == 300
            assert result["output_tokens"] == 130


# ---------------------------------------------------------------------------
# FSM-transition gate tests (Part 3 / HIGH-1)
# ---------------------------------------------------------------------------


class TestExtractConversationFSMTransitions:
    """Lock the worker-side FSM-transition gate.

    The ingest path passes ``str(job_uuid)`` as the 4th positional arg
    to extract_conversation; the worker reads it and calls
    mark_running / mark_completed / mark_failed only when job_id != "unknown".
    These tests assert that gate by calling extract_conversation
    directly with a real UUID and verifying each FSM call lands with the
    UUID-typed argument (not the raw string), AND that the negative case
    (job_id == "unknown") skips the FSM transitions entirely.

    A regression that reverts the gate would surface here even though
    the call-site assertions in test_ingest_enqueues_extraction.py
    would still pass (mocked arq never runs the worker).
    """

    @pytest.mark.asyncio
    async def test_mark_running_called_with_uuid_when_job_id_not_unknown(
        self,
        mock_ctx: Any,
        conn1: AsyncMock,
        conn2: AsyncMock,
    ) -> None:
        """mark_running receives ``UUID(job_id)`` (not the raw string) on conn1."""
        from uuid import UUID as _UUID

        conversation_id = 301
        user_id = "00000000-0000-0000-0000-000000000301"
        job_id = "11111111-2222-3333-4444-555555555555"

        mock_jobs = MagicMock()
        mock_jobs.mark_running = AsyncMock()
        mock_jobs.mark_completed = AsyncMock(return_value=True)
        mock_jobs.mark_failed = AsyncMock()
        mock_jobs.get_period_start = AsyncMock(return_value=None)

        with (
            patch("gubbi.extraction.jobs.extract_conversation.user_scoped_connection") as mock_usc,
            patch("gubbi.storage.repositories.conversations.read_conversation_by_id") as mock_rcbi,
            patch("gubbi.storage.repositories.conversations.get_processed_at") as mock_gpa,
            patch("gubbi.storage.repositories.conversations.mark_processed"),
            patch("gubbi.storage.repositories.topics.list_all") as mock_la,
            patch("gubbi.storage.repositories.topics.get_id") as mock_gti,
            patch("gubbi.storage.repositories.entries.append"),
            patch("gubbi.extraction.jobs.extract_conversation.extraction_jobs", mock_jobs),
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
                topic_path="test/fsm",
                topic_title="FSM",
                summary="s",
                confidence=0.9,
            )
            mock_ctx["extraction_service"].extract_entries.return_value = _make_entries_result(
                [ExtractedEntry(content="e", reasoning="", tags=[], entry_date="2026-01-01")]
            )

            await extract_conversation(mock_ctx, conversation_id, user_id, job_id)

            # mark_running was called with conn1 and the parsed UUID (not raw string).
            mock_jobs.mark_running.assert_awaited_once()
            call_args = mock_jobs.mark_running.await_args
            assert call_args is not None
            assert call_args.args[0] is conn1
            assert call_args.args[1] == _UUID(job_id), (
                "mark_running must receive UUID(job_id), not the raw string -- "
                f"got {call_args.args[1]!r}"
            )
            assert isinstance(call_args.args[1], _UUID)

    @pytest.mark.asyncio
    async def test_mark_completed_called_with_uuid_on_success(
        self,
        mock_ctx: Any,
        conn1: AsyncMock,
        conn2: AsyncMock,
    ) -> None:
        """On successful extraction, mark_completed is called on conn2 with ``UUID(job_id)``."""
        from uuid import UUID as _UUID

        conversation_id = 302
        user_id = "00000000-0000-0000-0000-000000000302"
        job_id = "22222222-3333-4444-5555-666666666666"

        mock_jobs = MagicMock()
        mock_jobs.mark_running = AsyncMock()
        mock_jobs.mark_completed = AsyncMock(return_value=True)
        mock_jobs.mark_failed = AsyncMock()
        mock_jobs.get_period_start = AsyncMock(return_value=None)

        with (
            patch("gubbi.extraction.jobs.extract_conversation.user_scoped_connection") as mock_usc,
            patch("gubbi.storage.repositories.conversations.read_conversation_by_id") as mock_rcbi,
            patch("gubbi.storage.repositories.conversations.get_processed_at") as mock_gpa,
            patch("gubbi.storage.repositories.conversations.mark_processed"),
            patch("gubbi.storage.repositories.topics.list_all") as mock_la,
            patch("gubbi.storage.repositories.topics.get_id") as mock_gti,
            patch("gubbi.storage.repositories.entries.append"),
            patch("gubbi.extraction.jobs.extract_conversation.extraction_jobs", mock_jobs),
            patch("gubbi.extraction.jobs.extract_conversation.record_audit", new=AsyncMock()),
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
                topic_path="test/fsm-completed",
                topic_title="FSM completed",
                summary="s",
                confidence=0.9,
            )
            mock_ctx["extraction_service"].extract_entries.return_value = _make_entries_result(
                [ExtractedEntry(content="e", reasoning="", tags=[], entry_date="2026-01-01")]
            )

            await extract_conversation(mock_ctx, conversation_id, user_id, job_id)

            # mark_completed received conn2 + UUID(job_id), not the raw string.
            mock_jobs.mark_completed.assert_awaited_once()
            call_args = mock_jobs.mark_completed.await_args
            assert call_args is not None
            # Signature: mark_completed(conn, UUID, topics_created=, entries_created=, cents_spent=)
            assert call_args.args[0] is conn2
            assert call_args.args[1] == _UUID(job_id), (
                "mark_completed must receive UUID(job_id), not the raw string -- "
                f"got {call_args.args[1]!r}"
            )
            assert isinstance(call_args.args[1], _UUID)

    @pytest.mark.asyncio
    async def test_mark_failed_called_with_uuid_on_extraction_error(
        self,
        mock_ctx: Any,
        conn1: AsyncMock,
    ) -> None:
        """On an extraction error, mark_failed receives ``UUID(job_id)`` on a fresh connection."""
        from uuid import UUID as _UUID

        conversation_id = 303
        user_id = "00000000-0000-0000-0000-000000000303"
        job_id = "33333333-4444-5555-6666-777777777777"

        failure_conn = AsyncMock()

        mock_jobs = MagicMock()
        mock_jobs.mark_running = AsyncMock()
        mock_jobs.mark_failed = AsyncMock(return_value=True)
        mock_jobs.get_period_start = AsyncMock(return_value=None)

        with (
            patch("gubbi.extraction.jobs.extract_conversation.user_scoped_connection") as mock_usc,
            patch("gubbi.storage.repositories.conversations.read_conversation_by_id") as mock_rcbi,
            patch("gubbi.storage.repositories.conversations.get_processed_at") as mock_gpa,
            patch("gubbi.storage.repositories.topics.list_all") as mock_la,
            patch("gubbi.extraction.jobs.extract_conversation.extraction_jobs", mock_jobs),
            patch("gubbi.extraction.jobs.extract_conversation.record_audit", new=AsyncMock()),
        ):
            mock_gpa.return_value = None
            mock_usc.side_effect = _make_usc_side_effect(conn1, failure_conn)

            fake_meta = MagicMock()
            fake_messages = [MagicMock(role="user", content="msg")]
            mock_rcbi.return_value = (fake_meta, fake_messages, 1)
            mock_la.return_value = ([], 0)

            mock_ctx[
                "extraction_service"
            ].categorize_conversation.return_value = CategorizationResult(
                topic_path="test/fsm-failed",
                topic_title="FSM failed",
                summary="s",
                confidence=0.9,
            )
            mock_ctx["extraction_service"].extract_entries.side_effect = RuntimeError("LLM down")

            with pytest.raises(RuntimeError, match="LLM down"):
                await extract_conversation(mock_ctx, conversation_id, user_id, job_id)

            # mark_failed called on the fresh failure connection with the
            # parsed UUID (not raw string).
            mock_jobs.mark_failed.assert_awaited_once()
            call_args = mock_jobs.mark_failed.await_args
            assert call_args is not None
            assert call_args.args[0] is failure_conn
            assert call_args.args[1] == _UUID(job_id), (
                "mark_failed must receive UUID(job_id), not the raw string -- "
                f"got {call_args.args[1]!r}"
            )
            assert isinstance(call_args.args[1], _UUID)

    @pytest.mark.asyncio
    async def test_no_fsm_transition_when_job_id_is_unknown(
        self,
        mock_ctx: Any,
        conn1: AsyncMock,
        conn2: AsyncMock,
    ) -> None:
        """When job_id == "unknown" (default sentinel), NO FSM transitions fire.

        This is the negative case for the worker-side ``job_id != "unknown"``
        gate. A regression that drops the guard would call mark_running /
        mark_completed with the literal string "unknown" -- ``UUID("unknown")``
        would raise ValueError, breaking production but escaping every
        test that mocks arq at the call-site.
        """
        conversation_id = 304
        user_id = "00000000-0000-0000-0000-000000000304"

        mock_jobs = MagicMock()
        mock_jobs.mark_running = AsyncMock()
        mock_jobs.mark_completed = AsyncMock(return_value=True)
        mock_jobs.mark_failed = AsyncMock()
        mock_jobs.get_period_start = AsyncMock(return_value=None)

        with (
            patch("gubbi.extraction.jobs.extract_conversation.user_scoped_connection") as mock_usc,
            patch("gubbi.storage.repositories.conversations.read_conversation_by_id") as mock_rcbi,
            patch("gubbi.storage.repositories.conversations.get_processed_at") as mock_gpa,
            patch("gubbi.storage.repositories.conversations.mark_processed"),
            patch("gubbi.storage.repositories.topics.list_all") as mock_la,
            patch("gubbi.storage.repositories.topics.get_id") as mock_gti,
            patch("gubbi.storage.repositories.entries.append"),
            patch("gubbi.extraction.jobs.extract_conversation.extraction_jobs", mock_jobs),
            patch("gubbi.extraction.jobs.extract_conversation.record_audit", new=AsyncMock()),
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
                topic_path="test/fsm-unknown",
                topic_title="FSM unknown",
                summary="s",
                confidence=0.9,
            )
            mock_ctx["extraction_service"].extract_entries.return_value = _make_entries_result(
                [ExtractedEntry(content="e", reasoning="", tags=[], entry_date="2026-01-01")]
            )

            # Call WITHOUT job_id -- defaults to "unknown".
            await extract_conversation(mock_ctx, conversation_id, user_id)

            # NO FSM transitions fired.
            mock_jobs.mark_running.assert_not_called()
            mock_jobs.mark_completed.assert_not_called()
            mock_jobs.mark_failed.assert_not_called()
            # And get_period_start (also gated on job_id != "unknown") not called.
            mock_jobs.get_period_start.assert_not_called()
