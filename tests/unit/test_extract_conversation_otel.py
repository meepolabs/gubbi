"""OTel span tests for the extract_conversation Arq job.

Verifies that the job entry point emits a parent ``extraction.job`` span
that wraps the body, captures success/failure outcome, records exceptions,
and that an ``extraction.llm_call`` child span is fired around each LLM
provider call. The wrap is the only signal HyperDX has when an LLM
extraction fails at public-beta scale.

The tests use the in-memory tracer fixture (``tests/conftest.py``) so the
SDK records spans without dialling an OTLP collector.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gubbi.extraction.jobs.extract_conversation import extract_conversation
from gubbi.extraction.llm.provider import LLMRateLimitError
from gubbi.extraction.service import CategorizationResult, ExtractedEntry, ExtractionEntriesResult

if TYPE_CHECKING:
    from tests.conftest import InMemoryExporter

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Local fixtures (mirror the patterns from test_extract_conversation_job.py
# but slimmed down -- the OTel tests do not need the full repository surface).
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


def _make_conn() -> AsyncMock:
    """Phase-1 / Phase-3 connection mock with idempotency check returning None."""
    c = AsyncMock()
    c.fetchval.return_value = None
    return c


def _make_conn2() -> AsyncMock:
    """Phase-3 connection mock whose ``transaction()`` is an async context manager."""
    c = AsyncMock()
    c.fetchval.return_value = None
    tx = MagicMock()
    tx.__aenter__ = AsyncMock(return_value=None)
    tx.__aexit__ = AsyncMock(return_value=False)
    c.transaction = MagicMock(return_value=tx)
    return c


def _make_provider(model: str = "fake-model-v1") -> MagicMock:
    """Build a simple provider stub with ``_model`` so the llm_call span has attrs."""
    provider = MagicMock()
    provider._model = model
    return provider


def _make_ctx(provider: MagicMock | None) -> Any:
    """Build an Arq context with extraction_service._llm = provider (or None)."""
    extraction_service = AsyncMock()
    extraction_service._llm = provider
    return {
        "pool": MagicMock(),
        "cipher": MagicMock(),
        "extraction_service": extraction_service,
        "redis": AsyncMock(),
        "redis_pool": MagicMock(),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestExtractionJobSpan:
    """Verify the parent extraction.job span captures the job outcome."""

    @pytest.mark.asyncio
    async def test_extraction_job_span_records_failure_on_llm_rate_limit(
        self,
        in_memory_tracer: tuple[Any, InMemoryExporter],
    ) -> None:
        """When the categorize LLM call raises LLMRateLimitError the parent
        extraction.job span is emitted with success=False and an exception
        event recorded."""
        _tracer, exporter = in_memory_tracer

        provider = _make_provider()
        ctx = _make_ctx(provider)
        ctx["extraction_service"].categorize_conversation.side_effect = LLMRateLimitError(
            "rate limited"
        )

        conn1 = _make_conn()
        conversation_id = 4242
        user_id = "00000000-0000-0000-0000-000000000b06"

        with (
            patch(
                "gubbi.extraction.jobs.extract_conversation.user_scoped_connection",
            ) as mock_usc,
            patch(
                "gubbi.storage.repositories.conversations.read_conversation_by_id",
            ) as mock_rcbi,
            patch(
                "gubbi.storage.repositories.topics.list_all",
            ) as mock_la,
        ):
            # Phase 1 acquires conn1; the failure path does not open conn2 nor
            # the failure_conn (job_id == "unknown" so _mark_job_failed is skipped).
            mock_usc.side_effect = _make_usc_side_effect(conn1)
            fake_meta = MagicMock()
            fake_messages = [MagicMock(role="user", content="hello")]
            mock_rcbi.return_value = (fake_meta, fake_messages, 1)
            mock_la.return_value = ([], 0)

            with pytest.raises(LLMRateLimitError, match="rate limited"):
                await extract_conversation(ctx, conversation_id, user_id)

        job_spans = [s for s in exporter.spans if s.name == "extraction.job"]
        assert len(job_spans) == 1, (
            f"expected 1 extraction.job span, got {[s.name for s in exporter.spans]}"
        )
        job_span = job_spans[0]
        attrs = dict(job_span.attributes) if job_span.attributes else {}
        assert attrs.get("success") is False
        # failure_reason is the _classify_error mapping for LLMRateLimitError.
        assert attrs.get("failure_reason") == "llm_rate_limited"
        # An exception event must have been recorded on the span.
        event_names = [ev.name for ev in job_span.events]
        assert "exception" in event_names, (
            f"expected an exception event on extraction.job span, got {event_names}"
        )

        # The categorize llm_call span must also have an exception event.
        llm_spans = [s for s in exporter.spans if s.name == "extraction.llm_call"]
        assert len(llm_spans) == 1
        llm_event_names = [ev.name for ev in llm_spans[0].events]
        assert "exception" in llm_event_names

    @pytest.mark.asyncio
    async def test_extraction_job_span_records_success_with_llm_child_span(
        self,
        in_memory_tracer: tuple[Any, InMemoryExporter],
    ) -> None:
        """Happy path: extraction.job span emitted with success=True, and at
        least one extraction.llm_call child span present with provider/model
        attributes captured."""
        _tracer, exporter = in_memory_tracer

        provider = _make_provider(model="fake-haiku-1")
        ctx = _make_ctx(provider)

        ctx["extraction_service"].categorize_conversation.return_value = CategorizationResult(
            topic_path="test/topic",
            topic_title="Test Topic",
            summary="s",
            confidence=0.9,
            input_tokens=100,
            output_tokens=20,
        )
        ctx["extraction_service"].extract_entries.return_value = ExtractionEntriesResult(
            entries=(
                ExtractedEntry(content="c", reasoning="r", tags=["t"], entry_date="2026-05-22"),
            ),
            input_tokens=80,
            output_tokens=15,
        )

        conn1 = _make_conn()
        conn2 = _make_conn2()
        conversation_id = 7
        user_id = "00000000-0000-0000-0000-000000000b07"

        with (
            patch(
                "gubbi.extraction.jobs.extract_conversation.user_scoped_connection",
            ) as mock_usc,
            patch(
                "gubbi.storage.repositories.conversations.read_conversation_by_id",
            ) as mock_rcbi,
            patch(
                "gubbi.storage.repositories.conversations.get_processed_at",
            ) as mock_gpa,
            patch("gubbi.storage.repositories.conversations.mark_processed"),
            patch("gubbi.storage.repositories.topics.list_all") as mock_la,
            patch("gubbi.storage.repositories.topics.get_id") as mock_gti,
            patch("gubbi.storage.repositories.entries.append"),
        ):
            mock_gpa.return_value = None
            mock_usc.side_effect = _make_usc_side_effect(conn1, conn2)
            fake_meta = MagicMock()
            fake_messages = [MagicMock(role="user", content="hello")]
            mock_rcbi.return_value = (fake_meta, fake_messages, 1)
            mock_la.return_value = ([], 0)
            mock_gti.return_value = 1

            result = await extract_conversation(ctx, conversation_id, user_id)
            assert result["skipped"] is False

        # Parent extraction.job span -- success path attrs.
        job_spans = [s for s in exporter.spans if s.name == "extraction.job"]
        assert len(job_spans) == 1
        job_attrs = dict(job_spans[0].attributes) if job_spans[0].attributes else {}
        assert job_attrs.get("success") is True
        assert "latency_ms" in job_attrs
        # failure_reason must NOT be set on the success path.
        assert "failure_reason" not in job_attrs

        # extraction.llm_call child spans -- one per LLM call (categorize +
        # extract_entries). Verify provider_name + model_name attributes wired
        # through safe_set_attributes (banned-key filter in place).
        llm_spans = [s for s in exporter.spans if s.name == "extraction.llm_call"]
        assert len(llm_spans) >= 1, (
            f"expected at least 1 extraction.llm_call span, got {[s.name for s in exporter.spans]}"
        )
        llm_attrs = dict(llm_spans[0].attributes) if llm_spans[0].attributes else {}
        assert llm_attrs.get("provider_name") == "MagicMock"
        assert llm_attrs.get("model_name") == "fake-haiku-1"
        # Token counts ride ``*_size`` keys -- see attrs.py allowlist note.
        assert "prompt_size" in llm_attrs
        assert "completion_size" in llm_attrs
        assert "latency_ms" in llm_attrs

        # Pin the parent-child relationship: every extraction.llm_call
        # span must be a child of the extraction.job span. A regression
        # that opens an llm_call span outside the job-span context (e.g.
        # by hoisting the categorize call above the
        # ``with tracer.start_as_current_span(EXTRACTION_JOB)`` block)
        # would silently emit sibling-tree spans that HyperDX cannot
        # link back to the job. The structural pin catches that without
        # depending on attribute presence.
        job_span = job_spans[0]
        for child in llm_spans:
            assert child.parent is not None, (
                f"extraction.llm_call span {child.name!r} has no parent; "
                "must be opened inside the extraction.job span context"
            )
            assert child.parent.span_id == job_span.context.span_id, (
                f"extraction.llm_call span parent.span_id "
                f"{child.parent.span_id:x} != job_span.span_id "
                f"{job_span.context.span_id:x}; the wrap order regressed -- "
                "llm_call spans must be children of extraction.job, not siblings"
            )
