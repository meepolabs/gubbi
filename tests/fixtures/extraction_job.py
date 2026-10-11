"""In-memory harness for ``extract_conversation`` unit tests.

The job runs end to end against a scripted LLM provider wrapped in the real
``ExtractionService`` and the real pipeline; only the database edge is faked.
Connections come from :class:`ConnectionTracker`, which records how many are
open whenever the provider is called, so a test can prove no connection is
held during an LLM call. Repository functions and the persistence step are
patched with ``AsyncMock`` so their calls can be asserted.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator, Mapping
from contextlib import ExitStack, asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from gubbi.extraction.jobs import extract_conversation as job_module
from gubbi.extraction.llm.provider import LLMMessage, LLMResponse, LLMUsage
from gubbi.extraction.persist import PersistResult
from gubbi.extraction.retry_policy import FailureOutcome
from gubbi.extraction.service import ExtractionService
from gubbi.models.conversation import Message
from gubbi.storage.repositories.conversations import ExtractionState

USER_ID = "00000000-0000-0000-0000-0000000000a1"
JOB_ID = "11111111-2222-3333-4444-555555555555"
CONVERSATION_ID = 4242
PLATFORM = "chatgpt"
CALL_USAGE = LLMUsage(input=1000, output=200, cached_input=300, reasoning=50)
# ScriptedProvider prices one token at a hundredth of a cent.
CALL_CENTS = (CALL_USAGE.input + CALL_USAGE.output) // 100

Step = Mapping[str, Any] | Exception


def chunk_output(
    topic: str = "health/fitness", entries: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """A single-call chunk response."""
    return {
        "topic_path": topic,
        "topic_title": "Fitness",
        "summary": "Talked about running.",
        "entries": [entry("Started running")] if entries is None else entries,
    }


def entry(content: str, source: int | None = 0) -> dict[str, Any]:
    """One entry of a chunk response."""
    return {
        "content": content,
        "why": f"why {content}",
        "tags": ["t"],
        "event_date": None,
        "source_message_index": source,
    }


def transcript() -> list[Message]:
    """A short dated transcript with one system turn the role filter must drop."""
    return [
        Message(role="system", content="SYSTEM-PROMPT-MARKER", timestamp="2026-03-01T08:00:00"),
        Message(
            role="user", content="I started running this week.", timestamp="2026-03-02T09:00:00"
        ),
        Message(role="assistant", content="Nice, keep it up.", timestamp="2026-03-02T09:01:00"),
    ]


@dataclass
class ConnectionTracker:
    """Stand-in for ``user_scoped_connection`` that counts open connections."""

    open_now: int = 0
    conns: list[AsyncMock] = field(default_factory=list)

    @asynccontextmanager
    async def connect(self, _pool: Any, *, user_id: Any) -> AsyncIterator[AsyncMock]:
        conn = AsyncMock()
        tx = MagicMock()
        tx.__aenter__ = AsyncMock(return_value=None)
        tx.__aexit__ = AsyncMock(return_value=False)
        conn.transaction = MagicMock(return_value=tx)
        self.conns.append(conn)
        self.open_now += 1
        try:
            yield conn
        finally:
            self.open_now -= 1


class ScriptedProvider:
    """Replays one scripted step per call and records what each call saw."""

    def __init__(
        self,
        steps: list[Step],
        tracker: ConnectionTracker | None = None,
        *,
        delay: float = 0.0,
    ) -> None:
        self._steps = list(steps)
        self._tracker = tracker
        self._delay = delay
        self._model = "scripted-model"
        self.user_contents: list[str] = []
        self.schemas: list[Mapping[str, Any] | None] = []
        self.open_connections_at_call: list[int] = []

    async def complete(
        self,
        messages: list[LLMMessage],
        system_prompt: str,
        output_schema: Mapping[str, Any] | None = None,
        *,
        time_budget: float | None = None,
    ) -> LLMResponse:
        self.user_contents.append(messages[0]["content"])
        self.schemas.append(output_schema)
        if self._tracker is not None:
            self.open_connections_at_call.append(self._tracker.open_now)
        step = self._steps.pop(0)
        if self._delay:
            await asyncio.sleep(self._delay)
        if isinstance(step, Exception):
            raise step
        return LLMResponse(
            content=dict(step),
            input_tokens=CALL_USAGE.input,
            output_tokens=CALL_USAGE.output,
            model=self._model,
            cached_input_tokens=CALL_USAGE.cached_input,
            reasoning_tokens=CALL_USAGE.reasoning,
        )

    def estimate_cost_cents(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens + output_tokens) / 100


class FakeEncoder:
    """Deterministic text encoder standing in for the ONNX embedding service."""

    def encode(self, text: str) -> list[float]:
        return [float(len(text)), 1.0]


@dataclass
class JobMocks:
    """Every faked database edge of one job run."""

    tracker: ConnectionTracker
    get_processed_at: AsyncMock
    lock_processed_at: AsyncMock
    read_conversation: AsyncMock
    get_extraction_state: AsyncMock
    list_topics: AsyncMock
    mark_processed: AsyncMock
    set_extraction_state: AsyncMock
    mark_running: AsyncMock
    mark_completed: AsyncMock
    mark_failed: AsyncMock
    persist: AsyncMock
    record_audit: AsyncMock
    settle_attempt: AsyncMock


def make_ctx(provider: ScriptedProvider) -> dict[str, Any]:
    """Arq context around ``provider``."""
    return {
        "pool": MagicMock(),
        "cipher": MagicMock(),
        "extraction_service": ExtractionService(provider),
        "embedding_service": FakeEncoder(),
        "redis": AsyncMock(),
    }


def extraction_state(
    *, stored_hash: str | None = None, stored_version: int | None = None
) -> ExtractionState:
    """Extraction state of a conversation never extracted before."""
    return ExtractionState(
        platform=PLATFORM,
        source="extension_chatgpt",
        created_at=datetime(2026, 3, 1, tzinfo=UTC),
        extracted_content_hash=stored_hash,
        extraction_version=stored_version,
    )


@contextmanager
def patched_job(
    tracker: ConnectionTracker,
    *,
    messages: list[Message] | None = None,
    state: ExtractionState | None = None,
    persisted_entries: int = 1,
) -> Iterator[JobMocks]:
    """Patch the job's database edge; yields the mocks for assertions."""
    conv = "gubbi.storage.repositories.conversations"
    jobs = "gubbi.storage.repositories.extraction_jobs"
    persisted = PersistResult(
        topic_id=7,
        topic_created=True,
        entry_ids=tuple(range(100, 100 + persisted_entries)),
        linked_entry_ids=(),
        suppressed=0,
    )
    loaded = messages if messages is not None else transcript()
    with ExitStack() as stack:

        def enter(target: str, **kwargs: Any) -> AsyncMock:
            mock = AsyncMock(**kwargs)
            stack.enter_context(patch(target, new=mock))
            return mock

        stack.enter_context(patch.object(job_module, "user_scoped_connection", tracker.connect))
        mocks = JobMocks(
            tracker=tracker,
            get_processed_at=enter(f"{conv}.get_processed_at", return_value=None),
            lock_processed_at=enter(f"{conv}.lock_processed_at", return_value=None),
            read_conversation=enter(
                f"{conv}.read_conversation_by_id",
                return_value=(MagicMock(), loaded, len(loaded)),
            ),
            get_extraction_state=enter(
                f"{conv}.get_extraction_state",
                return_value=state if state is not None else extraction_state(),
            ),
            list_topics=enter("gubbi.storage.repositories.topics.list_all", return_value=([], 0)),
            mark_processed=enter(f"{conv}.mark_processed"),
            set_extraction_state=enter(f"{conv}.set_extraction_state", return_value=True),
            mark_running=enter(f"{jobs}.mark_running", return_value=True),
            mark_completed=enter(f"{jobs}.mark_completed", return_value=True),
            mark_failed=enter(f"{jobs}.mark_failed", return_value=FailureOutcome.RETRY),
            persist=AsyncMock(return_value=persisted),
            record_audit=AsyncMock(),
            settle_attempt=AsyncMock(return_value=None),
        )
        stack.enter_context(patch.object(job_module, "persist_extraction", mocks.persist))
        stack.enter_context(patch.object(job_module, "record_audit", mocks.record_audit))
        stack.enter_context(patch.object(job_module, "settle_attempt", mocks.settle_attempt))
        yield mocks
