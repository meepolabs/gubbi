"""Audit-write failure telemetry must not export driver-supplied text.

A failing ``audit_log`` INSERT raises an asyncpg exception whose ``str()``
carries the server's message plus its ``DETAIL`` block. That text is
attacker-influenced and identity-bearing: the statement it describes
carries the actor id, the request IP and the User-Agent, and a rejection
message can quote them back. Two surfaces can export it:

* the writer's own ``audit.write`` span -- owned by gubbi-common, which
  replaces OTel's ``exception`` event with a sanitized one;
* every gubbi caller that catches (or whose span auto-records) the
  propagating exception -- owned here.

Each test plants distinct markers in the exception's message and
``DETAIL`` and asserts no marker survives onto any span or log line,
while the safe shape (exception class name, SQLSTATE) does. The
positive controls in :class:`TestHarnessCanFail` prove the harness sees
those markers when telemetry is left at its library default, so a clean
assertion elsewhere is evidence rather than an empty scan.
"""

from __future__ import annotations

import io
import logging
from contextlib import ExitStack
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import asyncpg
import pytest
import structlog
from gubbi_common.audit.sql import record_audit_async, record_audit_deduped_async
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from gubbi.app_context import AppContext
from gubbi.audit import audited
from gubbi.telemetry.sanitized_errors import (
    _MAX_CHAIN_DEPTH,
    DB_ERROR_EVENT_NAME,
    _chain,
    _walk_chain,
    is_driver_caused,
    record_exception_sanitized,
    safe_error_fields,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from opentelemetry.sdk.trace import ReadableSpan

    from tests.conftest import InMemoryExporter

pytestmark = pytest.mark.unit

_USER_ID = UUID("11111111-2222-3333-4444-555555555555")

# Synthetic values planted in the exception the driver would raise. Each
# stands for one class of identity-bearing text a real PostgreSQL error
# for this statement can quote back: the session id, the originating IP,
# the User-Agent, a bearer credential and the server's DETAIL block.
# Deliberately low-entropy, hyphenated words rather than key-shaped
# strings: a realistic-looking credential here trips the repo's secret
# scanner, and the assertions only need each marker to be unique.
_MARKER_SID = "planted-session-marker"
_MARKER_IP = "203.0.113.77"
_MARKER_USER_AGENT = "PlantedAgent/9.9"
_MARKER_TOKEN = "planted-bearer-marker"
_MARKER_DETAIL = "planted-detail-marker"
_MARKERS: tuple[str, ...] = (
    _MARKER_SID,
    _MARKER_IP,
    _MARKER_USER_AGENT,
    _MARKER_TOKEN,
    _MARKER_DETAIL,
)

_SQLSTATE = "42501"


def _planted_driver_error() -> asyncpg.PostgresError:
    """Build the failing-INSERT exception with every marker embedded.

    Both halves matter: ``str(exc)`` concatenates the message AND the
    ``DETAIL`` block, so a surface that stringifies the exception leaks
    both, while one that reads only ``exc.detail`` leaks only the
    second.
    """
    exc = asyncpg.exceptions.InsufficientPrivilegeError(
        "permission denied for table audit_log while inserting "
        f"sid={_MARKER_SID} ip={_MARKER_IP} ua={_MARKER_USER_AGENT} "
        f"token={_MARKER_TOKEN}"
    )
    exc.detail = f"DETAIL: rejected row carried {_MARKER_DETAIL}"
    return exc


def _failing_conn() -> MagicMock:
    """Return a connection mock whose every write raises the planted error."""
    conn = MagicMock()
    conn.execute = AsyncMock(side_effect=_planted_driver_error())
    conn.fetchval = AsyncMock(side_effect=_planted_driver_error())
    return conn


def _span_text(span: ReadableSpan) -> str:
    """Flatten every operator-visible string on a finished span.

    Covers the three places a recorded exception lands: span attributes,
    span events (name plus event attributes -- where ``record_exception``
    writes ``exception.message`` and ``exception.stacktrace``), and the
    status description.
    """
    parts: list[str] = [span.name]
    parts.extend(f"{key}={value}" for key, value in dict(span.attributes or {}).items())
    for event in span.events:
        parts.append(event.name)
        parts.extend(f"{key}={value}" for key, value in dict(event.attributes or {}).items())
    if span.status is not None and span.status.description:
        parts.append(span.status.description)
    return "\n".join(parts)


def _assert_no_markers(text: str, surface: str) -> None:
    leaked = [marker for marker in _MARKERS if marker in text]
    assert not leaked, f"{surface} exported planted driver text {leaked}: {text}"


def _event_attributes(span: ReadableSpan, event_name: str) -> dict[str, Any]:
    for event in span.events:
        if event.name == event_name:
            return dict(event.attributes or {})
    msg = f"no {event_name!r} event on span {span.name!r}; got {[e.name for e in span.events]}"
    raise AssertionError(msg)


class _LogCapture(logging.Handler):
    """Collect every log surface a structlog emit can write to.

    Two are needed. Structured fields land on the ``LogRecord`` and are
    rendered by the ProcessorFormatter. The exception itself does NOT:
    the configured processor chain ends in ``ExceptionPrettyPrinter``,
    which pops ``exc_info`` and prints the formatted traceback to its own
    file object, so a caller passing ``exc_info=True`` leaks through a
    surface that inspecting ``LogRecord.exc_text`` cannot see.
    """

    def __init__(self, exception_sink: io.StringIO) -> None:
        super().__init__(level=logging.DEBUG)
        self._exception_sink = exception_sink
        self.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                processors=[
                    structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                    structlog.processors.JSONRenderer(),
                ],
            )
        )
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(self.format(record))

    @property
    def text(self) -> str:
        """Every captured structured line plus every pretty-printed traceback."""
        return "\n".join([*self.lines, self._exception_sink.getvalue()])


@pytest.fixture
def log_capture() -> Iterator[_LogCapture]:
    """Attach a root handler and redirect the pretty-printer's own output."""
    printers = [
        processor
        for processor in structlog.get_config()["processors"]
        if isinstance(processor, structlog.processors.ExceptionPrettyPrinter)
    ]
    assert printers, (
        "the configured structlog chain has no ExceptionPrettyPrinter; "
        "the traceback surface this fixture captures has moved -- re-derive it "
        "from gubbi_common.telemetry.logging.initialize_logger before trusting "
        "any assertion built on this fixture"
    )
    sink = io.StringIO()
    original_files = [printer._file for printer in printers]
    for printer in printers:
        printer._file = sink

    handler = _LogCapture(sink)
    root = logging.getLogger()
    original_level = root.level
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    try:
        yield handler
    finally:
        root.removeHandler(handler)
        root.setLevel(original_level)
        for printer, original_file in zip(printers, original_files, strict=True):
            printer._file = original_file


def _make_app_ctx() -> AppContext:
    ctx = MagicMock(spec=AppContext)
    ctx.pool = MagicMock()
    return ctx


def _extraction_success_ctx() -> tuple[dict[str, Any], list[Any]]:
    """Build an extraction job context whose only failure point is the audit write.

    Returns the Arq context plus the patches that carry the job through
    both LLM calls and persistence, so the audit write is actually
    reached. Injecting at ``record_audit`` is what makes the assertion
    about an audit failure rather than a vendor failure.
    """
    from gubbi.extraction.service import (
        CategorizationResult,
        ExtractedEntry,
        ExtractionEntriesResult,
    )

    conn = AsyncMock()
    conn.fetchval.return_value = None
    transaction = MagicMock()
    transaction.__aenter__ = AsyncMock(return_value=None)
    transaction.__aexit__ = AsyncMock(return_value=False)
    conn.transaction = MagicMock(return_value=transaction)

    connection_cm = MagicMock()
    connection_cm.__aenter__ = AsyncMock(return_value=conn)
    connection_cm.__aexit__ = AsyncMock(return_value=False)

    extraction_service = AsyncMock()
    extraction_service._llm = MagicMock(_model="fake-model")
    extraction_service.categorize_conversation.return_value = CategorizationResult(
        topic_path="probe/topic",
        topic_title="Probe",
        summary="s",
        confidence=0.9,
        input_tokens=10,
        output_tokens=5,
    )
    extraction_service.extract_entries.return_value = ExtractionEntriesResult(
        entries=(ExtractedEntry(content="c", reasoning="r", tags=["t"], entry_date="2026-05-22"),),
        input_tokens=8,
        output_tokens=4,
    )

    ctx: dict[str, Any] = {
        "pool": MagicMock(),
        "cipher": MagicMock(),
        "extraction_service": extraction_service,
        "redis": AsyncMock(),
        "redis_pool": MagicMock(),
    }

    patches = [
        patch(
            "gubbi.extraction.jobs.extract_conversation.user_scoped_connection",
            return_value=connection_cm,
        ),
        patch(
            "gubbi.storage.repositories.conversations.read_conversation_by_id",
            return_value=(MagicMock(), [MagicMock(role="user", content="hi")], 1),
        ),
        patch("gubbi.storage.repositories.conversations.get_processed_at", return_value=None),
        patch("gubbi.storage.repositories.conversations.mark_processed"),
        patch("gubbi.storage.repositories.topics.list_all", return_value=([], 0)),
        patch("gubbi.storage.repositories.topics.get_id", return_value=1),
        patch("gubbi.storage.repositories.entries.append"),
    ]
    return ctx, patches


# ---------------------------------------------------------------------------
# Writer spans (gubbi-common, reached through the gubbi pin)
# ---------------------------------------------------------------------------


class TestWriterSpan:
    """The ``audit.write`` span describes a failure without driver text."""

    async def test_canonical_writer_failure_span_carries_only_type_and_sqlstate(
        self,
        in_memory_tracer: tuple[Any, InMemoryExporter],
    ) -> None:
        # Arrange
        _tracer, exporter = in_memory_tracer
        conn = _failing_conn()

        # Act
        with pytest.raises(asyncpg.exceptions.InsufficientPrivilegeError):
            await record_audit_async(
                conn,
                actor_type="user",
                actor_id=str(_USER_ID),
                action="entry.created",
                target_type="entry",
                target_id="99",
                target_kind="entry",
            )

        # Assert
        write_spans = [span for span in exporter.spans if span.name == "audit.write"]
        assert len(write_spans) == 1, f"expected one audit.write span, got {len(write_spans)}"
        span = write_spans[0]
        _assert_no_markers(_span_text(span), "audit.write span")
        assert "exception" not in [event.name for event in span.events], (
            "the SDK exception event carries exception.message and exception.stacktrace; "
            "the writer must add a sanitized event instead"
        )
        failure_events = [event for event in span.events if event.name != "exception"]
        assert len(failure_events) == 1, (
            f"expected exactly one sanitized failure event, got "
            f"{[event.name for event in span.events]}"
        )
        attributes = dict(failure_events[0].attributes or {})
        assert attributes.get("exception.type") == "InsufficientPrivilegeError"
        assert attributes.get("db.sqlstate") == _SQLSTATE
        assert span.status is not None
        assert span.status.status_code is StatusCode.ERROR

    async def test_dedup_writer_failure_span_carries_only_type_and_sqlstate(
        self,
        in_memory_tracer: tuple[Any, InMemoryExporter],
    ) -> None:
        # Arrange
        _tracer, exporter = in_memory_tracer
        conn = _failing_conn()

        # Act
        with pytest.raises(asyncpg.exceptions.InsufficientPrivilegeError):
            await record_audit_deduped_async(
                conn,
                actor_type="user",
                actor_id=str(_USER_ID),
                action="entry.created",
                target_kind="entry",
                target_id="99",
                metadata={"content_hash": "abc"},
            )

        # Assert
        write_spans = [span for span in exporter.spans if span.name == "audit.write"]
        assert len(write_spans) == 1, f"expected one audit.write span, got {len(write_spans)}"
        span = write_spans[0]
        _assert_no_markers(_span_text(span), "deduped audit.write span")
        assert "exception" not in [event.name for event in span.events]
        failure_events = [event for event in span.events if event.name != "exception"]
        assert len(failure_events) == 1
        attributes = dict(failure_events[0].attributes or {})
        assert attributes.get("exception.type") == "InsufficientPrivilegeError"
        assert attributes.get("db.sqlstate") == _SQLSTATE

    async def test_successful_write_span_reports_success_and_no_failure_event(
        self,
        in_memory_tracer: tuple[Any, InMemoryExporter],
    ) -> None:
        """The sanitization must not fire on the success path."""
        # Arrange
        _tracer, exporter = in_memory_tracer
        conn = MagicMock()
        conn.execute = AsyncMock(return_value=None)

        # Act
        await record_audit_async(
            conn,
            actor_type="user",
            actor_id=str(_USER_ID),
            action="entry.created",
        )

        # Assert
        write_spans = [span for span in exporter.spans if span.name == "audit.write"]
        assert len(write_spans) == 1
        span = write_spans[0]
        assert dict(span.attributes or {}).get("success") is True
        assert span.events == (), f"success path must add no failure event, got {span.events}"
        assert span.status is not None
        assert span.status.status_code is not StatusCode.ERROR


# ---------------------------------------------------------------------------
# Caller surfaces (owned here)
# ---------------------------------------------------------------------------


class TestAuditedDecoratorCallerLog:
    """``@audited`` swallows the failure; its own log must stay sanitized."""

    async def test_decorator_failure_log_omits_driver_text_and_keeps_safe_shape(
        self,
        in_memory_tracer: tuple[Any, InMemoryExporter],
        log_capture: _LogCapture,
    ) -> None:
        # Arrange
        _tracer, exporter = in_memory_tracer
        conn = _failing_conn()
        connection_cm = MagicMock()
        connection_cm.__aenter__ = AsyncMock(return_value=conn)
        connection_cm.__aexit__ = AsyncMock(return_value=False)

        async def _handler(**_kwargs: Any) -> dict[str, Any]:
            return {"status": "ok", "entry_id": 42}

        decorated = audited(
            "entry.created",
            target_type="entry",
            target_kind="entry",
            app_ctx=_make_app_ctx(),
        )(_handler)

        # Act
        with (
            patch("gubbi.audit.decorator.current_user_id") as mock_current_user_id,
            patch(
                "gubbi.audit.decorator.user_scoped_connection",
                return_value=connection_cm,
            ),
            patch("gubbi.audit.decorator.record_audit_persistence_failure"),
        ):
            mock_current_user_id.get.return_value = _USER_ID
            result = await decorated(topic="t", content="c")

        # Assert
        assert result == {"status": "ok", "entry_id": 42}, (
            "a failed audit write must stay best-effort at the decorator layer"
        )
        _assert_no_markers(log_capture.text, "audited() failure log")
        assert "InsufficientPrivilegeError" in log_capture.text, (
            "the sanitized log must still name the exception class for triage"
        )
        assert _SQLSTATE in log_capture.text, (
            "the sanitized log must still carry the SQLSTATE for triage"
        )
        for span in exporter.spans:
            _assert_no_markers(_span_text(span), f"{span.name} span")


class TestScaffoldOperatorCallerLog:
    """Startup provisioning swallows an audit failure; its log must be clean."""

    async def test_scaffold_audit_failure_log_omits_driver_text(
        self,
        log_capture: _LogCapture,
    ) -> None:
        # Arrange
        from gubbi.users.bootstrap import scaffold_operator

        conn = MagicMock()
        conn.execute = AsyncMock(side_effect=_planted_driver_error())
        conn.fetchval = AsyncMock(
            side_effect=[
                UUID("22222222-3333-4444-5555-666666666666"),
                UUID("22222222-3333-4444-5555-666666666666"),
            ]
        )
        acquire_cm = MagicMock()
        acquire_cm.__aenter__ = AsyncMock(return_value=conn)
        acquire_cm.__aexit__ = AsyncMock(return_value=False)
        pool = MagicMock()
        pool.acquire = MagicMock(return_value=acquire_cm)

        # Act
        await scaffold_operator(pool, "operator@example.com", "UTC")

        # Assert
        _assert_no_markers(log_capture.text, "scaffold_operator failure log")
        assert "InsufficientPrivilegeError" in log_capture.text
        assert _SQLSTATE in log_capture.text


class TestUnhandledDriverErrorHandler:
    """A propagated driver error reaching the app handler is logged sanitized."""

    async def test_wrapped_driver_error_takes_the_safe_path(
        self,
        log_capture: _LogCapture,
    ) -> None:
        """A translated wrapper must be classified by its cause, not its type.

        ``DatabaseUnavailable(str(exc))`` copies the driver message into
        its own ``str()`` while failing an ``isinstance(exc,
        asyncpg.PostgresError)`` check, so an outermost-type test routes it
        down the ``exc_info`` branch and exports the DETAIL. This is the
        case that distinguishes ``is_driver_caused`` from ``isinstance``.
        """
        # Arrange
        from gubbi.main import general_exception_handler
        from gubbi.storage.exceptions import DatabaseUnavailable

        driver = _planted_driver_error()
        try:
            raise DatabaseUnavailable(str(driver)) from driver
        except DatabaseUnavailable as exc:
            wrapped = exc

        request = MagicMock()
        request.url.path = "/api/v1/entries/move"
        request.method = "POST"

        # Act
        response = await general_exception_handler(request, wrapped)

        # Assert
        assert response.status_code == 500
        assert _MARKER_SID in str(wrapped), (
            "the wrapper must genuinely carry the driver text, or this proves nothing"
        )
        _assert_no_markers(log_capture.text, "general_exception_handler log (wrapped)")
        assert "DatabaseUnavailable" in log_capture.text
        assert _SQLSTATE in log_capture.text, (
            "the SQLSTATE is found through the cause chain, not on the wrapper"
        )

    async def test_general_exception_handler_omits_driver_text(
        self,
        log_capture: _LogCapture,
    ) -> None:
        # Arrange
        from gubbi.main import general_exception_handler

        request = MagicMock()
        request.url.path = "/api/v1/entries/move"
        request.method = "POST"

        # Act
        response = await general_exception_handler(request, _planted_driver_error())

        # Assert
        assert response.status_code == 500
        _assert_no_markers(log_capture.text, "general_exception_handler log")
        assert b"planted" not in response.body, "the response body must never echo driver text"
        assert "InsufficientPrivilegeError" in log_capture.text
        assert _SQLSTATE in log_capture.text

    async def test_non_driver_exception_keeps_its_traceback(
        self,
        log_capture: _LogCapture,
    ) -> None:
        """Sanitization is scoped to driver text, not to all error logging.

        An exception this codebase raises itself owns its message, and
        dropping its traceback would blind the operator for no gain.
        """
        # Arrange
        from gubbi.main import general_exception_handler

        request = MagicMock()
        request.url.path = "/api/v1/entries/move"
        request.method = "POST"

        # Act
        response = await general_exception_handler(request, RuntimeError("our own message"))

        # Assert
        assert response.status_code == 500
        assert "our own message" in log_capture.text, (
            "a non-driver exception must keep its traceback in the log"
        )


class TestHttpBoundaryServerSpan:
    """The HTTP server span must not auto-record the driver exception.

    This crosses the boundary the unit-level handler test cannot: the
    FastAPI instrumentation wraps the whole middleware stack and records
    any exception that escapes it, so an app-level ``Exception`` handler
    alone is not enough -- Starlette's ``ServerErrorMiddleware``
    re-raises after the handler builds the response.
    """

    @pytest.mark.parametrize(
        "make_exc",
        [
            pytest.param(_planted_driver_error, id="PostgresError"),
            pytest.param(
                lambda: asyncpg.InterfaceError(f"connection is closed {_MARKER_SID}"),
                id="InterfaceError",
            ),
            pytest.param(
                lambda: asyncpg.InternalClientError(f"client state {_MARKER_TOKEN}"),
                id="InternalClientError",
            ),
        ],
    )
    async def test_driver_error_escaping_a_route_leaves_no_marker_on_the_server_span(
        self,
        make_exc: Any,
        in_memory_tracer: tuple[Any, InMemoryExporter],
        restore_asyncpg_instrumentation: Any,
        log_capture: _LogCapture,
    ) -> None:
        """All three classified driver families must be registered handlers.

        Registering only ``PostgresError`` leaves the other two taking the
        ``Exception`` catch-all route, which ``ServerErrorMiddleware``
        re-raises past -- so the instrumentation records them on the server
        span with message and stacktrace even though the handler logged
        them safely. The markers are planted in each class's own message
        here, since those two carry no DETAIL of their own.
        """
        # Arrange
        from fastapi import FastAPI
        from httpx import ASGITransport, AsyncClient

        from gubbi.main import register_exception_handlers
        from gubbi.telemetry import _wire_instrumentors

        _tracer, exporter = in_memory_tracer
        app = FastAPI()
        register_exception_handlers(app)

        @app.get("/audit-boom")
        async def _boom() -> dict[str, str]:
            raise make_exc()

        _wire_instrumentors(app)

        # Act
        transport = ASGITransport(app=app, raise_app_exceptions=False)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.get("/audit-boom")

        # Assert
        assert response.status_code == 500
        _assert_no_markers(response.text, "HTTP response body")
        _assert_no_markers(log_capture.text, "handler log")
        server_spans = [span for span in exporter.spans if span.name.endswith("/audit-boom")]
        assert server_spans, f"expected a server span, got {[s.name for s in exporter.spans]}"
        for span in server_spans:
            _assert_no_markers(_span_text(span), f"{span.name} server span")

    async def test_an_ordinary_exception_still_reaches_the_catch_all_unchanged(
        self,
        in_memory_tracer: tuple[Any, InMemoryExporter],
        restore_asyncpg_instrumentation: Any,
        log_capture: _LogCapture,
    ) -> None:
        """Control: the extra registrations must not alter ordinary handling."""
        # Arrange
        from fastapi import FastAPI
        from httpx import ASGITransport, AsyncClient

        from gubbi.main import register_exception_handlers
        from gubbi.telemetry import _wire_instrumentors

        _tracer, _exporter = in_memory_tracer
        app = FastAPI()
        register_exception_handlers(app)

        @app.get("/plain-boom")
        async def _boom() -> dict[str, str]:
            raise RuntimeError("an ordinary failure")

        _wire_instrumentors(app)

        # Act
        transport = ASGITransport(app=app, raise_app_exceptions=False)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.get("/plain-boom")

        # Assert
        assert response.status_code == 500
        assert response.json() == {"error": "Internal server error"}
        assert "an ordinary failure" in log_capture.text, (
            "an ordinary exception keeps its traceback in the log"
        )

    def test_every_classified_driver_family_is_registered(self) -> None:
        """One source of truth: the handler map must cover the classified set.

        A class added to ``DRIVER_ERROR_TYPES`` but not registered here
        would be sanitized in the log and leaked on the span -- the exact
        split this test exists to prevent.
        """
        # Arrange
        from fastapi import FastAPI

        from gubbi.main import general_exception_handler, register_exception_handlers
        from gubbi.telemetry.sanitized_errors import DRIVER_ERROR_TYPES

        app = FastAPI()

        # Act
        register_exception_handlers(app)

        # Assert
        for driver_type in DRIVER_ERROR_TYPES:
            assert app.exception_handlers.get(driver_type) is general_exception_handler, (
                f"{driver_type.__name__} is classified as driver-caused but has no "
                "concrete handler, so it escapes to the server span un-sanitized"
            )


class TestMcpToolCallSpan:
    """``mcp.tool_call`` wraps tool dispatch, which can reach an audit write.

    The reachable route is the repository layer, not the ``@audited``
    decorator: ``@audited`` catches its own ``record_audit`` failure and
    logs it, so nothing propagates to this span from there. What DOES
    propagate is a direct ``record_audit`` inside a repository function
    called by a handler -- ``entries.move_entries_to_topic`` and
    ``topics.rename`` / ``merge`` / ``delete_with_reassign`` each write an
    audit row in the caller's transaction and let the rejection escape.

    Worth recording: as of this change those four functions are called
    only from ``api/v1/web/*``, so the propagating audit failure reaches
    this span only once a tool handler calls one of them. The sanitization
    is therefore a standing guard on a live route rather than a fix for a
    reproduced leak, and the test below drives the repository function
    directly through a patched handler to exercise the contract that makes
    it safe.

    ``gubbi.tools.registry`` resolves its tracer at import, so each test
    points that module attribute at the fixture's tracer -- otherwise the
    spans land in whichever provider was installed when the module first
    loaded and the scan reads an empty list.
    """

    async def test_direct_audit_failure_from_a_tool_leaves_no_marker_on_the_call_span(
        self,
        in_memory_tracer: tuple[Any, InMemoryExporter],
    ) -> None:
        # Arrange -- a handler that reaches the repository's own
        # record_audit, which propagates rather than swallowing.
        from gubbi.storage.repositories import entries as entries_repo
        from gubbi.tools import registry

        tracer, exporter = in_memory_tracer

        conn = MagicMock()
        conn.is_in_transaction = MagicMock(return_value=True)
        conn.fetchrow = AsyncMock(return_value={"path": "dest/topic"})
        conn.fetch = AsyncMock(return_value=[{"id": 7}])
        conn.execute = AsyncMock(return_value="UPDATE 1")

        class _ToolManager:
            async def call_tool(
                self,
                name: str,
                arguments: dict[str, Any] | None = None,
                **_kwargs: Any,
            ) -> Any:
                return await entries_repo.move_entries_to_topic(
                    conn, [7], 3, actor_id=str(_USER_ID)
                )

        manager = _ToolManager()

        # Act
        with (
            patch.object(registry, "_tracer", tracer),
            patch.object(
                entries_repo,
                "record_audit",
                AsyncMock(side_effect=_planted_driver_error()),
            ),
        ):
            registry.patch_tool_manager(manager)  # type: ignore[arg-type]  # duck-typed stand-in for the SDK ToolManager
            with pytest.raises(asyncpg.exceptions.InsufficientPrivilegeError):
                await manager.call_tool("journal_move_entries", {})

        # Assert
        call_spans = [span for span in exporter.spans if span.name == "mcp.tool_call"]
        assert len(call_spans) == 1, f"expected one mcp.tool_call span, got {len(call_spans)}"
        span = call_spans[0]
        _assert_no_markers(_span_text(span), "mcp.tool_call span")
        assert "exception" not in [event.name for event in span.events], (
            "the SDK exception event would carry the driver message and stacktrace"
        )
        assert _event_attributes(span, DB_ERROR_EVENT_NAME).get("db.sqlstate") == _SQLSTATE
        assert dict(span.attributes or {}).get("result") == "InsufficientPrivilegeError", (
            "the span must still name the failure class"
        )
        assert span.status is not None
        assert not span.status.description

    async def test_ordinary_tool_error_keeps_the_sdk_default_diagnostics(
        self,
        in_memory_tracer: tuple[Any, InMemoryExporter],
    ) -> None:
        """MCP diagnostics must be untouched for a non-database failure."""
        # Arrange
        from gubbi.tools import registry

        tracer, exporter = in_memory_tracer

        class _ToolManager:
            async def call_tool(
                self,
                name: str,
                arguments: dict[str, Any] | None = None,
                **_kwargs: Any,
            ) -> Any:
                raise ValueError("tool argument was invalid")

        manager = _ToolManager()

        # Act
        with patch.object(registry, "_tracer", tracer):
            registry.patch_tool_manager(manager)  # type: ignore[arg-type]  # duck-typed stand-in for the SDK ToolManager
            with pytest.raises(ValueError, match="tool argument was invalid"):
                await manager.call_tool("journal_append_entry", {})

        # Assert
        call_spans = [span for span in exporter.spans if span.name == "mcp.tool_call"]
        assert len(call_spans) == 1
        span = call_spans[0]
        assert "exception" in [event.name for event in span.events], (
            "an ordinary tool error keeps the SDK exception event"
        )
        assert "tool argument was invalid" in _span_text(span)
        assert span.status is not None
        assert span.status.description == "ValueError: tool argument was invalid", (
            "an ordinary tool error keeps the SDK status description"
        )
        assert dict(span.attributes or {}).get("result") == "ValueError"


class TestExtractionJobSpan:
    """The outer ``extraction.job`` span wraps the worker's own audit writes.

    Injection is at the audit write itself, not at the LLM calls: the two
    inner ``extraction.llm_call`` spans wrap only a vendor call and can
    never see an audit-write failure, so they keep the SDK's default
    recording and are not this module's concern.
    """

    async def test_audit_write_failure_leaves_no_marker_on_the_job_span(
        self,
        in_memory_tracer: tuple[Any, InMemoryExporter],
    ) -> None:
        # Arrange -- a full success path whose only failure is record_audit.
        from gubbi.extraction.jobs import extract_conversation as job_module

        _tracer, exporter = in_memory_tracer
        ctx, patches = _extraction_success_ctx()

        # Act
        with ExitStack() as stack:
            for cm in patches:
                stack.enter_context(cm)
            stack.enter_context(
                patch.object(
                    job_module,
                    "record_audit",
                    AsyncMock(side_effect=_planted_driver_error()),
                )
            )
            stack.enter_context(pytest.raises(asyncpg.exceptions.InsufficientPrivilegeError))
            await job_module.extract_conversation(ctx, 4242, "00000000-0000-0000-0000-0000000000b6")

        # Assert
        job_spans = [span for span in exporter.spans if span.name == "extraction.job"]
        assert len(job_spans) == 1, (
            f"expected one extraction.job span, got {[s.name for s in exporter.spans]}"
        )
        span = job_spans[0]
        _assert_no_markers(_span_text(span), "extraction.job span")
        assert _event_attributes(span, DB_ERROR_EVENT_NAME).get("db.sqlstate") == _SQLSTATE
        assert dict(span.attributes or {}).get("success") is False
        for other in exporter.spans:
            _assert_no_markers(_span_text(other), f"{other.name} span")

    async def test_non_driver_failure_keeps_the_sdk_default_shape_on_the_job_span(
        self,
        in_memory_tracer: tuple[Any, InMemoryExporter],
    ) -> None:
        """A failure this codebase owns keeps its message and description."""
        # Arrange
        from gubbi.extraction.jobs import extract_conversation as job_module

        _tracer, exporter = in_memory_tracer
        ctx, patches = _extraction_success_ctx()

        # Act
        with ExitStack() as stack:
            for cm in patches:
                stack.enter_context(cm)
            stack.enter_context(
                patch.object(
                    job_module,
                    "record_audit",
                    AsyncMock(side_effect=RuntimeError("our own message")),
                )
            )
            stack.enter_context(pytest.raises(RuntimeError, match="our own message"))
            await job_module.extract_conversation(ctx, 4242, "00000000-0000-0000-0000-0000000000b6")

        # Assert
        job_spans = [span for span in exporter.spans if span.name == "extraction.job"]
        assert len(job_spans) == 1
        span = job_spans[0]
        assert "exception" in [event.name for event in span.events], (
            "a non-driver failure must keep the SDK exception event"
        )
        assert "our own message" in _span_text(span)
        assert span.status is not None
        assert span.status.description == "RuntimeError: our own message", (
            "a non-driver failure must keep the SDK status description"
        )

    async def test_llm_failure_keeps_the_sdk_default_shape_on_the_llm_span(
        self,
        in_memory_tracer: tuple[Any, InMemoryExporter],
    ) -> None:
        """Regression: the inner vendor spans were reverted to SDK defaults."""
        # Arrange
        from gubbi.extraction.jobs import extract_conversation as job_module
        from gubbi.extraction.llm.provider import LLMRateLimitError

        _tracer, exporter = in_memory_tracer
        ctx, patches = _extraction_success_ctx()
        ctx["extraction_service"].categorize_conversation.side_effect = LLMRateLimitError(
            "vendor said slow down"
        )

        # Act
        with ExitStack() as stack:
            for cm in patches:
                stack.enter_context(cm)
            stack.enter_context(pytest.raises(LLMRateLimitError, match="vendor said slow down"))
            await job_module.extract_conversation(ctx, 4242, "00000000-0000-0000-0000-0000000000b6")

        # Assert
        llm_spans = [span for span in exporter.spans if span.name == "extraction.llm_call"]
        assert len(llm_spans) == 1
        assert "exception" in [event.name for event in llm_spans[0].events]
        assert "vendor said slow down" in _span_text(llm_spans[0])


class TestMarkJobFailedSecondaryAudit:
    """The swallowed secondary audit write in ``_mark_job_failed``."""

    async def test_secondary_audit_failure_log_omits_driver_text(
        self,
        log_capture: _LogCapture,
    ) -> None:
        # Arrange
        from gubbi.extraction.jobs import extract_conversation as job_module

        conn = MagicMock()
        connection_cm = MagicMock()
        connection_cm.__aenter__ = AsyncMock(return_value=conn)
        connection_cm.__aexit__ = AsyncMock(return_value=False)

        # Act
        with (
            patch.object(job_module, "user_scoped_connection", return_value=connection_cm),
            patch.object(
                job_module.extraction_jobs,
                "mark_failed",
                AsyncMock(return_value=True),
            ),
            patch.object(
                job_module,
                "record_audit",
                AsyncMock(side_effect=_planted_driver_error()),
            ),
        ):
            await job_module._mark_job_failed(  # the swallowing helper is the unit under test
                MagicMock(),
                UUID("22222222-3333-4444-5555-666666666666"),
                "33333333-4444-5555-6666-777777777777",
                4242,
                "internal_error",
            )

        # Assert
        assert "mark_job_failed_secondary_error" in log_capture.text, (
            "the secondary failure must still be logged"
        )
        _assert_no_markers(log_capture.text, "mark_job_failed_secondary_error log")
        assert "InsufficientPrivilegeError" in log_capture.text
        assert _SQLSTATE in log_capture.text


# ---------------------------------------------------------------------------
# Classification: the cause/context chain
# ---------------------------------------------------------------------------


class TestDriverCausedClassification:
    """A wrapper must be classified by what it wraps, not by its own type."""

    def test_wrapper_chained_from_a_driver_error_is_driver_caused(self) -> None:
        # Arrange
        from gubbi.storage.exceptions import DatabaseUnavailable

        driver = _planted_driver_error()
        try:
            raise DatabaseUnavailable(str(driver)) from driver
        except DatabaseUnavailable as exc:
            wrapper = exc

        # Act / Assert -- the wrapper's own str() carries the driver text,
        # which is exactly why classifying on its type alone would leak.
        assert _MARKER_SID in str(wrapper)
        assert is_driver_caused(wrapper)
        assert safe_error_fields(wrapper) == {
            "error_type": "DatabaseUnavailable",
            "db_sqlstate": _SQLSTATE,
        }

    def test_wrapper_carrying_a_driver_error_only_as_context_is_driver_caused(self) -> None:
        """Implicit chaining (a bare ``raise`` inside ``except``) counts too.

        ``raise ... from None`` is included as well. It sets
        ``__suppress_context__`` so tracebacks hide the original, but
        ``__context__`` still references it, and a wrapper's own message
        may have been built from the driver's regardless of how it was
        raised. Classification stays conservative: the cost of a false
        positive is one less message in the log, the cost of a false
        negative is exported identity data.
        """
        # Arrange
        try:
            try:
                raise _planted_driver_error()
            except asyncpg.PostgresError:
                raise RuntimeError("translated with from None") from None
        except RuntimeError as exc:
            suppressed = exc

        try:
            try:
                raise _planted_driver_error()
            except asyncpg.PostgresError:
                raise RuntimeError("translated implicitly")  # noqa: B904  # implicit __context__ is the case under test
        except RuntimeError as exc:
            with_context = exc

        # Act / Assert
        assert suppressed.__context__ is not None
        assert suppressed.__suppress_context__
        assert is_driver_caused(suppressed), (
            "__suppress_context__ hides the original from tracebacks but does not "
            "clear __context__, so the conservative call is still driver-caused"
        )
        assert is_driver_caused(with_context)
        assert safe_error_fields(with_context).get("db_sqlstate") == _SQLSTATE

    def test_a_wrapper_with_no_chain_at_all_is_not_driver_caused(self) -> None:
        """An exception raised outside any ``except`` has no chain to walk."""
        # Arrange
        exc = RuntimeError("raised standalone")

        # Act / Assert
        assert exc.__cause__ is None
        assert exc.__context__ is None
        assert not is_driver_caused(exc)

    def test_cause_free_wrapper_with_no_driver_anywhere_is_not_driver_caused(self) -> None:
        # Arrange
        exc = RuntimeError("entirely our own")

        # Act / Assert
        assert not is_driver_caused(exc)
        assert safe_error_fields(exc) == {"error_type": "RuntimeError"}

    def test_a_cyclic_chain_terminates_and_visits_each_link_once(self) -> None:
        """``raise x from y`` where y's cause is x must not revisit links.

        Termination alone is not evidence here: the depth bound would
        stop the walk regardless, so an assertion that merely returns
        passes with the cycle guard deleted. What the guard uniquely
        provides is that each link is visited exactly once, so the
        assertion is on the walked chain itself.
        """
        # Arrange
        first = RuntimeError("first")
        second = RuntimeError("second")
        first.__cause__ = second
        second.__cause__ = first

        # Act
        walked = _chain(first)

        # Assert
        assert [id(link) for link in walked] == [id(first), id(second)], (
            "a cyclic chain must yield each link once, not alternate up to the depth bound"
        )
        assert not is_driver_caused(first)
        assert safe_error_fields(first) == {"error_type": "RuntimeError"}

    def test_a_cycle_does_not_mask_a_driver_error_inside_it(self) -> None:
        """Deduplication must not stop the walk before the driver link."""
        # Arrange -- plain -> driver -> back to plain.
        plain = RuntimeError("wrapper")
        driver = _planted_driver_error()
        plain.__cause__ = driver
        driver.__cause__ = plain

        # Act / Assert
        assert is_driver_caused(plain)
        assert safe_error_fields(plain).get("db_sqlstate") == _SQLSTATE

    def test_a_malformed_sqlstate_is_dropped_not_forwarded(self) -> None:
        """The one driver-supplied string that reaches telemetry is shape-checked.

        ``db.sqlstate`` is forwarded verbatim, so its shape is the only
        thing standing between a five-character code and arbitrary
        attacker-influenced text on a span attribute. Each case below is
        rejected for a different reason, so deleting any one half of the
        validation leaves a case that fails.
        """
        # Arrange -- correct length but non-alphabet, wrong length, and
        # a marker-bearing value of exactly five characters.
        cases = {
            "425*1": "right length, character outside the SQLSTATE alphabet",
            "lower": "right length, lowercase is not the standard alphabet",
            "425011": "too long",
            "4250": "too short",
        }

        for candidate, reason in cases.items():
            exc = asyncpg.exceptions.InsufficientPrivilegeError("denied")
            exc.sqlstate = candidate

            # Act
            fields = safe_error_fields(exc)

            # Assert
            assert "db_sqlstate" not in fields, f"must drop {candidate!r} ({reason})"
            assert fields == {"error_type": "InsufficientPrivilegeError"}

        # A well-shaped value still passes through.
        good = asyncpg.exceptions.InsufficientPrivilegeError("denied")
        good.sqlstate = _SQLSTATE
        assert safe_error_fields(good).get("db_sqlstate") == _SQLSTATE

    def test_a_malformed_sqlstate_is_dropped_from_the_span_event_too(
        self,
        in_memory_tracer: tuple[Any, InMemoryExporter],
    ) -> None:
        """The span path shares the validation, not just the log path."""
        # Arrange
        _tracer, exporter = in_memory_tracer
        exc = asyncpg.exceptions.InsufficientPrivilegeError("denied")
        exc.sqlstate = f"x{_MARKER_SID}"

        # Act
        with trace.get_tracer("tests.sqlstate_shape").start_as_current_span("probe") as span:
            record_exception_sanitized(span, exc)

        # Assert
        spans = [item for item in exporter.spans if item.name == "probe"]
        assert len(spans) == 1
        attributes = _event_attributes(spans[0], DB_ERROR_EVENT_NAME)
        assert "db.sqlstate" not in attributes
        _assert_no_markers(_span_text(spans[0]), "probe span")

    def test_depth_exhaustion_fails_closed_and_withholds_markers(self) -> None:
        """Past the depth bound the chain is uncharacterised, so sanitize.

        The previous shape of this test asserted the OPPOSITE -- that a
        driver error beyond the bound is "not reached" -- which pinned an
        intentional leak as correct behavior. What matters is not whether
        the walk found the driver link but whether any marker can escape,
        so the assertion is now on marker absence.
        """
        # Arrange -- 40 plain wrappers, then the driver error.
        deepest: BaseException = _planted_driver_error()
        for index in range(40):
            wrapper = RuntimeError(f"layer {index}")
            wrapper.__cause__ = deepest
            deepest = wrapper

        # Act
        fields = safe_error_fields(deepest)

        # Assert
        assert is_driver_caused(deepest), (
            "a chain that exhausted the depth bound is uncharacterised and must "
            "be treated as driver-caused"
        )
        _assert_no_markers("\n".join(f"{k}={v}" for k, v in fields.items()), "safe fields")
        assert fields["error_type"] == "RuntimeError"

    def test_depth_exhaustion_withholds_markers_on_the_span_too(
        self,
        in_memory_tracer: tuple[Any, InMemoryExporter],
    ) -> None:
        """The span path must not record the deep chain's driver text."""
        # Arrange
        _tracer, exporter = in_memory_tracer
        deepest: BaseException = _planted_driver_error()
        for index in range(40):
            wrapper = RuntimeError(f"layer {index}")
            wrapper.__cause__ = deepest
            deepest = wrapper

        # Act
        with trace.get_tracer("tests.depth_bound").start_as_current_span("probe") as span:
            record_exception_sanitized(span, deepest)

        # Assert -- record_exception would format the whole __cause__
        # chain into exception.stacktrace, markers included.
        spans = [item for item in exporter.spans if item.name == "probe"]
        assert len(spans) == 1
        _assert_no_markers(_span_text(spans[0]), "probe span")
        assert "exception" not in [event.name for event in spans[0].events]

    def test_a_short_non_driver_chain_keeps_the_sdk_default(
        self,
        in_memory_tracer: tuple[Any, InMemoryExporter],
    ) -> None:
        """Control: fail-closed must not swallow ordinary shallow errors."""
        # Arrange -- two plain links, well inside the bound.
        inner = ValueError("inner cause")
        outer = RuntimeError("outer wrapper")
        outer.__cause__ = inner
        _tracer, exporter = in_memory_tracer

        # Act
        assert not is_driver_caused(outer)
        with trace.get_tracer("tests.depth_bound").start_as_current_span("probe") as span:
            record_exception_sanitized(span, outer)

        # Assert
        spans = [item for item in exporter.spans if item.name == "probe"]
        assert "exception" in [event.name for event in spans[0].events]
        assert spans[0].status is not None
        assert spans[0].status.description == "RuntimeError: outer wrapper"

    def test_a_chain_exactly_at_the_bound_is_still_characterised(self) -> None:
        """The boundary itself: a terminating chain of exactly the bound length."""
        # Arrange -- _MAX_CHAIN_DEPTH links, terminating (no deeper cause).
        chain_head: BaseException = ValueError("deepest plain")
        for index in range(_MAX_CHAIN_DEPTH - 1):
            wrapper = RuntimeError(f"layer {index}")
            wrapper.__cause__ = chain_head
            chain_head = wrapper

        # Act
        walked, exhausted = _walk_chain(chain_head)

        # Assert
        assert len(walked) == _MAX_CHAIN_DEPTH
        assert not exhausted, "a chain that ends exactly at the bound terminated normally"
        assert not is_driver_caused(chain_head), (
            "a fully-walked non-driver chain must keep the SDK default"
        )

    def test_a_falsey_cause_link_does_not_truncate_the_walk(self) -> None:
        """A falsey ``__cause__`` must be followed, not skipped.

        An exception subclass is free to define ``__bool__`` / ``__len__``
        and evaluate falsey. ``cause or context`` then skips a REAL cause:
        measured, a falsey link holding an asyncpg error as its own cause
        made the entire chain read as non-driver, and the buried SQLSTATE
        and DETAIL escaped classification entirely.
        """

        # Arrange
        class _FalseyError(RuntimeError):
            """Evaluates falsey while still being a real exception link."""

            def __bool__(self) -> bool:
                return False

        driver = _planted_driver_error()
        falsey = _FalseyError("falsey link")
        falsey.__cause__ = driver
        falsey.__context__ = None
        outer = RuntimeError("outer wrapper")
        outer.__cause__ = falsey
        # No context to fall back to: with truthiness the walk ends here.
        outer.__context__ = None

        # Act
        walked = _chain(outer)

        # Assert
        assert not bool(falsey), "the middle link must genuinely be falsey"
        assert [type(link).__name__ for link in walked] == [
            "RuntimeError",
            "_FalseyError",
            "InsufficientPrivilegeError",
        ], "the walk must follow a falsey cause through to the driver error"
        assert is_driver_caused(outer)
        assert safe_error_fields(outer).get("db_sqlstate") == _SQLSTATE

    def test_a_falsey_chain_with_no_driver_error_stays_non_driver(self) -> None:
        """Control: following falsey links must not classify everything as driver."""

        # Arrange
        class _FalseyError(RuntimeError):
            def __bool__(self) -> bool:
                return False

        innermost = ValueError("plain innermost")
        falsey = _FalseyError("falsey link")
        falsey.__cause__ = innermost
        falsey.__context__ = None
        outer = RuntimeError("outer wrapper")
        outer.__cause__ = falsey
        outer.__context__ = None

        # Act
        walked = _chain(outer)

        # Assert
        assert len(walked) == 3, "the falsey link is still followed"
        assert not is_driver_caused(outer), (
            "a fully-walked chain with no driver error keeps the SDK default"
        )
        assert safe_error_fields(outer) == {"error_type": "RuntimeError"}

    def test_a_falsey_context_only_link_is_still_followed(self) -> None:
        """The fallback side of the expression needs the same treatment."""

        # Arrange
        class _FalseyError(RuntimeError):
            def __bool__(self) -> bool:
                return False

        driver = _planted_driver_error()
        falsey = _FalseyError("falsey context link")
        falsey.__cause__ = None
        falsey.__context__ = driver
        outer = RuntimeError("outer")
        outer.__cause__ = None
        outer.__context__ = falsey

        # Act / Assert
        assert is_driver_caused(outer)
        assert safe_error_fields(outer).get("db_sqlstate") == _SQLSTATE

    def test_a_realistically_deep_driver_chain_is_reached(self) -> None:
        # Arrange
        near = RuntimeError("one layer")
        near.__cause__ = _planted_driver_error()

        # Act / Assert
        assert is_driver_caused(near)
        assert safe_error_fields(near).get("db_sqlstate") == _SQLSTATE

    @pytest.mark.parametrize(
        "factory",
        [
            pytest.param(
                lambda: asyncpg.InterfaceError("connection is closed"),
                id="InterfaceError",
            ),
            pytest.param(
                lambda: asyncpg.InternalClientError("could not resolve query result types"),
                id="InternalClientError",
            ),
        ],
    )
    def test_driver_side_asyncpg_classes_are_driver_caused(
        self,
        factory: Any,
        in_memory_tracer: tuple[Any, InMemoryExporter],
    ) -> None:
        """Driver-owned strings are withheld; the type is retained.

        Measured on a live PostgreSQL 17 audit-write path, these two
        classes carry fixed driver strings rather than statement values.
        They are classified as driver-caused anyway: the strings are the
        driver's to change, and the cost of withholding a fixed string is
        far below the cost of exporting a value if a future version starts
        interpolating one.
        """
        # Arrange
        _tracer, exporter = in_memory_tracer
        exc = factory()

        # Act
        assert is_driver_caused(exc)
        with trace.get_tracer("tests.driver_classes").start_as_current_span("probe") as span:
            record_exception_sanitized(span, exc)

        # Assert
        spans = [item for item in exporter.spans if item.name == "probe"]
        assert len(spans) == 1
        span = spans[0]
        attributes = _event_attributes(span, DB_ERROR_EVENT_NAME)
        assert attributes.get("exception.type") == type(exc).__name__, "the type is retained"
        assert "exception.message" not in attributes, "the driver message is withheld"
        assert "exception.stacktrace" not in attributes, "the stacktrace is withheld"
        assert str(exc) not in _span_text(span), "no part of the driver string may appear"
        assert "exception" not in [event.name for event in span.events]
        assert span.status is not None
        assert not span.status.description
        # No SQLSTATE on these classes -- the attribute is simply absent
        # rather than invented.
        assert "db.sqlstate" not in attributes

    def test_postgres_warning_is_not_classified_as_driver_caused(self) -> None:
        """It is never raised on the execute path, so it is deliberately excluded.

        Pinned so a later "make it symmetric" change has to confront the
        reason: asyncpg builds PostgresWarning through
        ``PostgresLogMessage.new`` for a connection log message and does
        not raise it from ``Connection.execute``.
        """
        # Arrange
        warning = asyncpg.exceptions.PostgresWarning.__new__(
            asyncpg.exceptions.PostgresWarning,
        )

        # Act / Assert
        assert not isinstance(warning, asyncpg.PostgresError)
        assert not is_driver_caused(warning)

    async def test_scaffold_runtime_error_message_omits_driver_text(self) -> None:
        """The provisioning wrapper must not interpolate the driver message."""
        # Arrange
        from gubbi.users.bootstrap import scaffold_operator

        conn = MagicMock()
        conn.execute = AsyncMock()
        conn.fetchval = AsyncMock(side_effect=_planted_driver_error())
        acquire_cm = MagicMock()
        acquire_cm.__aenter__ = AsyncMock(return_value=conn)
        acquire_cm.__aexit__ = AsyncMock(return_value=False)
        pool = MagicMock()
        pool.acquire = MagicMock(return_value=acquire_cm)

        # Act
        with pytest.raises(RuntimeError) as caught:
            await scaffold_operator(pool, "operator@example.com", "UTC")

        # Assert
        _assert_no_markers(str(caught.value), "scaffold RuntimeError message")
        assert "InsufficientPrivilegeError" in str(caught.value)
        assert is_driver_caused(caught.value), (
            "the original must stay chained so the sanitizer still classifies it"
        )


# ---------------------------------------------------------------------------
# Positive controls -- the harness must see a default auto-record
# ---------------------------------------------------------------------------


class TestHarnessCanFail:
    """Prove the scans above would catch an unsanitized caller."""

    async def test_capture_sees_default_exception_log(
        self,
        log_capture: _LogCapture,
    ) -> None:
        """``logger.exception()`` -- the default way to log a caught exception.

        Note the asymmetry this control pins: on the configured
        ``AsyncBoundLogger`` an ``exc_info=True`` emit exports NOTHING,
        because the emit is dispatched to a worker thread where
        ``sys.exc_info()`` is already empty. ``exception()`` captures the
        exception on the calling thread first, so it does export. A
        control built on ``exc_info=True`` would therefore pass on an
        unsanitized caller and prove nothing.
        """
        # Arrange
        logger = structlog.get_logger("tests.audit_failure_control")

        # Act
        try:
            raise _planted_driver_error()
        except asyncpg.PostgresError:
            await logger.exception("control_audit_write_failed")

        # Assert
        leaked = [marker for marker in _MARKERS if marker in log_capture.text]
        assert set(leaked) == set(_MARKERS), (
            f"the log capture must see every planted marker an exception() emit exports; "
            f"saw {leaked}"
        )

    async def test_capture_sees_stringified_exception_field(
        self,
        log_capture: _LogCapture,
    ) -> None:
        """``error=str(exc)`` -- the other shape in use across this codebase."""
        # Arrange
        logger = structlog.get_logger("tests.audit_failure_control")
        exc = _planted_driver_error()

        # Act
        await logger.warning("control_audit_write_failed", error=str(exc))

        # Assert
        leaked = [marker for marker in _MARKERS if marker in log_capture.text]
        assert set(leaked) == set(_MARKERS), (
            f"the log capture must see markers a stringified exception field exports; saw {leaked}"
        )

    def test_span_scan_sees_default_record_exception(
        self,
        in_memory_tracer: tuple[Any, InMemoryExporter],
    ) -> None:
        """``record_exception`` + a described status -- the OTel default shape."""
        # Arrange
        _tracer, exporter = in_memory_tracer
        exc = _planted_driver_error()

        # Act
        with trace.get_tracer("tests.audit_failure_control").start_as_current_span(
            "control.caller"
        ) as span:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, description=f"{type(exc).__name__}: {exc}"))

        # Assert
        control_spans = [s for s in exporter.spans if s.name == "control.caller"]
        assert len(control_spans) == 1
        text = _span_text(control_spans[0])
        leaked = [marker for marker in _MARKERS if marker in text]
        assert set(leaked) == set(_MARKERS), (
            f"the span scan must see every planted marker record_exception exports; saw {leaked}"
        )
