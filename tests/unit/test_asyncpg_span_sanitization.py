"""The asyncpg auto-instrumentation must not export driver text.

``AsyncPGInstrumentor`` opens a client span per ``Connection.execute``
and lets the exception escape, so the SDK's context exit records the
driver message and stacktrace and builds a status description from
``str(exc)``. Inside an ``audit.write`` trace that text is the rejected
audit row's actor id, IP and User-Agent, and it lands on a CHILD of the
span gubbi-common already sanitizes.

The tests below run against a real PostgreSQL (they skip when it is
unreachable), because the leak is produced by the driver's own error and
a hand-built exception would only prove the wrapper, not the path. Each
test restores the instrumentor's prior state so it cannot leak
instrumentation into the rest of the suite.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from typing import TYPE_CHECKING, Any
from unittest.mock import patch
from uuid import uuid4

import asyncpg
import pytest
from opentelemetry import trace
from opentelemetry.instrumentation.asyncpg import AsyncPGInstrumentor

from gubbi.telemetry.asyncpg_sanitized import (
    SanitizerUnsafeError,
    SanitizingTracer,
    _instrumented_targets,
    instrument_asyncpg_sanitized,
)

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import ReadableSpan

    from tests.conftest import InMemoryExporter

pytestmark = pytest.mark.unit

_DEFAULT_TEST_DB = "postgresql://journal:testpass@localhost:5433/journal_test"
_DSN = os.environ.get("TEST_DATABASE_URL", _DEFAULT_TEST_DB)

# Planted in a column value so the server quotes it back in the
# duplicate-key DETAIL, the way a real audit row's actor id, IP and
# User-Agent are quoted back. Low-entropy on purpose (see the secret
# scanner note in test_audit_failure_telemetry).
_MARKER = "planted-child-span-marker"


def _span_text(span: ReadableSpan) -> str:
    """Flatten every operator-visible string on a finished span."""
    parts: list[str] = [span.name]
    parts.extend(f"{key}={value}" for key, value in dict(span.attributes or {}).items())
    for event in span.events:
        parts.append(event.name)
        parts.extend(f"{key}={value}" for key, value in dict(event.attributes or {}).items())
    if span.status is not None and span.status.description:
        parts.append(span.status.description)
    return "\n".join(parts)


async def _provoke_duplicate_key(dsn: str) -> BaseException | None:
    """Insert a row twice inside an ``audit.write`` span; return the rejection.

    The marker is the PRIMARY KEY value, so PostgreSQL's ``DETAIL`` quotes
    it back ("Key (actor_id)=(...) already exists") -- precisely how a
    rejected audit row's actor id reaches the error text.

    The probe table is a connection-local ``TEMP`` table under a
    per-connection generated name. Two concurrent runs of this module (or
    a sibling worktree pointed at the same database) therefore cannot
    collide: TEMP puts the table in this session's own schema, dropped
    when the connection closes, and the unique suffix means even a shared
    non-temp namespace would not clash. A shared fixed name plus
    DROP/CREATE, which this replaced, is a race by construction.
    """
    conn = await asyncpg.connect(dsn, statement_cache_size=0)
    try:
        table = f"audit_child_probe_{uuid4().hex}"
        await conn.execute(
            f"CREATE TEMP TABLE {table} (actor_id text PRIMARY KEY, note text)"  # generated suffix is a uuid4 hex, never input
        )
        await conn.execute(
            f"INSERT INTO {table} (actor_id, note) VALUES ($1, 'first')",  # noqa: S608  # same
            _MARKER,
        )
        with trace.get_tracer("tests.audit_child_span").start_as_current_span("audit.write"):
            try:
                await conn.execute(
                    f"INSERT INTO {table} (actor_id, note) VALUES ($1, 'second')",  # noqa: S608  # same
                    _MARKER,
                )
            except asyncpg.PostgresError as exc:
                return exc
        return None
    finally:
        await conn.close()


async def _connect_or_skip(dsn: str) -> None:
    try:
        probe = await asyncpg.connect(dsn, timeout=5)
    except Exception as exc:  # any connect failure means no DB here
        pytest.skip(f"PostgreSQL not reachable at {dsn}: {exc}")
    else:
        await probe.close()


class TestAsyncpgChildSpan:
    """Default instrumentation leaks; the production wiring does not."""

    async def test_default_instrumentation_leaks_driver_detail_positive_control(
        self,
        in_memory_tracer: tuple[Any, InMemoryExporter],
        restore_asyncpg_instrumentation: AsyncPGInstrumentor,
    ) -> None:
        """Control: without the wrapper the child span carries the DETAIL.

        This is what makes the sanitized assertion below evidence rather
        than an empty scan -- it proves the exporter, the trace context
        and the marker placement can all surface the leak.
        """
        # Arrange
        await _connect_or_skip(_DSN)
        _tracer, exporter = in_memory_tracer
        instrumentor = restore_asyncpg_instrumentation
        instrumentor.instrument()

        # Act
        rejection = await _provoke_duplicate_key(_DSN)

        # Assert
        assert rejection is not None, "expected a duplicate-key rejection"
        assert _MARKER in str(rejection), "the server must quote the planted value back"
        child_spans = [span for span in exporter.spans if span.name != "audit.write"]
        assert child_spans, "expected instrumented client spans"
        leaking = [span for span in child_spans if _MARKER in _span_text(span)]
        assert leaking, (
            "default asyncpg instrumentation must leak the DETAIL for this control "
            f"to mean anything; scanned {[s.name for s in child_spans]}"
        )
        events = {event.name for span in leaking for event in span.events}
        assert "exception" in events

    async def test_sanitized_instrumentation_withholds_driver_detail(
        self,
        in_memory_tracer: tuple[Any, InMemoryExporter],
        restore_asyncpg_instrumentation: AsyncPGInstrumentor,
    ) -> None:
        """The production wiring: no message, stacktrace, or description."""
        # Arrange
        await _connect_or_skip(_DSN)
        _tracer, exporter = in_memory_tracer
        instrumentor = restore_asyncpg_instrumentation
        assert instrument_asyncpg_sanitized(instrumentor), "sanitized wiring must install"

        # Act
        rejection = await _provoke_duplicate_key(_DSN)

        # Assert
        assert rejection is not None
        assert _MARKER in str(rejection), "the rejection itself still carries the text"
        for span in exporter.spans:
            text = _span_text(span)
            assert _MARKER not in text, f"{span.name} span exported the driver DETAIL: {text}"

        failing = [
            span
            for span in exporter.spans
            for event in span.events
            if event.name == "exception" and span.name != "audit.write"
        ]
        assert failing, "the failure must still be recorded on the client span"
        span = failing[0]
        event = next(event for event in span.events if event.name == "exception")
        attributes = dict(event.attributes or {})
        assert attributes.get("exception.type") == "asyncpg.exceptions.UniqueViolationError"
        assert attributes.get("db.sqlstate") == "23505"
        assert "exception.message" not in attributes
        assert "exception.stacktrace" not in attributes
        assert span.status is not None
        assert span.status.status_code is trace.StatusCode.ERROR
        assert not span.status.description, "the status description is built from str(exc)"

    async def test_sanitized_instrumentation_preserves_parent_linkage_and_attributes(
        self,
        in_memory_tracer: tuple[Any, InMemoryExporter],
        restore_asyncpg_instrumentation: AsyncPGInstrumentor,
    ) -> None:
        """Sampling/resource/context come from the real tracer, not the wrapper."""
        # Arrange
        await _connect_or_skip(_DSN)
        _tracer, exporter = in_memory_tracer
        instrument_asyncpg_sanitized(restore_asyncpg_instrumentation)

        # Act
        await _provoke_duplicate_key(_DSN)

        # Assert
        parents = [span for span in exporter.spans if span.name == "audit.write"]
        assert len(parents) == 1
        parent = parents[0]
        failing = [
            span
            for span in exporter.spans
            for event in span.events
            if event.name == "exception" and span.name != "audit.write"
        ]
        assert failing
        child = failing[0]
        assert child.parent is not None, "the client span must stay a child"
        assert child.parent.span_id == parent.context.span_id
        assert child.context.trace_id == parent.context.trace_id
        assert child.kind is trace.SpanKind.CLIENT, "span kind must survive the wrapper"
        attributes = dict(child.attributes or {})
        assert attributes.get("db.system") == "postgresql", (
            "the instrumentor's own attributes must survive the wrapper"
        )
        assert child.resource is parent.resource, "resource must come from the same provider"


class TestSanitizingTracerUnits:
    """Behavior of the wrapper itself, independent of a live database."""

    def test_non_driver_exception_keeps_the_sdk_shape(
        self,
        in_memory_tracer: tuple[Any, InMemoryExporter],
    ) -> None:
        # Arrange
        tracer, exporter = in_memory_tracer
        wrapped = SanitizingTracer(tracer)

        # Act
        with (
            pytest.raises(ValueError, match="our own message"),
            wrapped.start_as_current_span("probe"),
        ):
            raise ValueError("our own message")

        # Assert
        spans = [span for span in exporter.spans if span.name == "probe"]
        assert len(spans) == 1
        span = spans[0]
        attributes = dict(next(e for e in span.events if e.name == "exception").attributes or {})
        assert attributes.get("exception.message") == "our own message", (
            "a non-driver exception keeps its message"
        )
        assert span.status is not None
        assert span.status.description == "ValueError: our own message"

    def test_success_path_records_nothing(
        self,
        in_memory_tracer: tuple[Any, InMemoryExporter],
    ) -> None:
        # Arrange
        tracer, exporter = in_memory_tracer
        wrapped = SanitizingTracer(tracer)

        # Act
        with wrapped.start_as_current_span("probe") as span:
            span.set_attribute("k", "v")

        # Assert
        spans = [span for span in exporter.spans if span.name == "probe"]
        assert len(spans) == 1
        assert spans[0].events == ()
        assert spans[0].status is not None
        assert spans[0].status.status_code is not trace.StatusCode.ERROR

    def test_unknown_attributes_delegate_to_the_wrapped_tracer(
        self,
        in_memory_tracer: tuple[Any, InMemoryExporter],
    ) -> None:
        """A future instrumentor entry point degrades, never breaks."""
        # Arrange
        tracer, _exporter = in_memory_tracer
        wrapped = SanitizingTracer(tracer)

        # Act
        span = wrapped.start_span("delegated")
        span.end()

        # Assert -- start_span is not wrapped; it must still work.
        assert wrapped.start_span is not None

    def test_already_instrumented_singleton_is_wrapped_not_reinstrumented(
        self,
        restore_asyncpg_instrumentation: AsyncPGInstrumentor,
    ) -> None:
        """A prior autoload leaves ``instrument()`` a no-op; wrap the live tracer."""
        # Arrange -- simulate the distro having instrumented already.
        instrumentor = restore_asyncpg_instrumentation
        instrumentor.instrument()
        assert instrumentor.is_instrumented_by_opentelemetry
        inner = instrumentor._tracer  # asserting on the state the wrapper replaces
        assert not isinstance(inner, SanitizingTracer)

        # Act
        installed = instrument_asyncpg_sanitized(instrumentor)

        # Assert
        assert installed
        assert isinstance(instrumentor._tracer, SanitizingTracer)  # same
        assert instrumentor._tracer._inner is inner  # same

    def test_idempotent_when_already_sanitized(
        self,
        restore_asyncpg_instrumentation: AsyncPGInstrumentor,
    ) -> None:
        """A second ``configure_otel`` must not stack wrappers."""
        # Arrange
        instrumentor = restore_asyncpg_instrumentation
        instrument_asyncpg_sanitized(instrumentor)
        first = instrumentor._tracer  # asserting the wrapper is not re-wrapped

        # Act
        assert instrument_asyncpg_sanitized(instrumentor)

        # Assert
        assert instrumentor._tracer is first  # same

    def test_failure_to_wrap_removes_instrumentation_rather_than_leaving_it(self) -> None:
        """A broken instrumentor must leave asyncpg un-instrumented, not default.

        Returning False while default wrappers stay installed would be the
        worst outcome: a reassuring answer over a process that exports the
        driver DETAIL of every failing audit write.
        """

        # Arrange
        class _Broken:
            is_instrumented_by_opentelemetry = False
            uninstrument_calls = 0

            def instrument(self) -> None:
                raise RuntimeError("instrumentor exploded")

            def uninstrument(self) -> None:
                type(self).uninstrument_calls += 1

        broken = _Broken()

        # Act
        installed = instrument_asyncpg_sanitized(broken)

        # Assert
        assert installed is False
        assert _Broken.uninstrument_calls == 1, (
            "the unsafe path must attempt removal, not just report failure"
        )
        assert not _instrumented_targets(), (
            "no un-sanitized wrapper may survive a failed installation"
        )

    def test_a_tracer_that_cannot_be_replaced_removes_instrumentation(
        self,
        restore_asyncpg_instrumentation: AsyncPGInstrumentor,
    ) -> None:
        """Incompatible tracer layout: refuse to run with default wrappers.

        Simulates a future instrumentor whose ``_tracer`` is read-only (a
        property, a slot, a descriptor). The wrappers are real -- installed
        by the genuine instrumentor first -- so the removal assertion is
        about actual driver state, not a mock.
        """
        # Arrange
        real = restore_asyncpg_instrumentation
        real.instrument()
        assert _instrumented_targets(), "the genuine wrappers must be installed first"

        class _ReadOnlyTracer:
            """Stands in for an instrumentor that refuses tracer assignment."""

            is_instrumented_by_opentelemetry = True

            def __init__(self, delegate: AsyncPGInstrumentor) -> None:
                self._delegate = delegate

            @property
            def _tracer(self) -> Any:
                return trace.get_tracer("immutable")

            def uninstrument(self) -> None:
                self._delegate.uninstrument()

        # Act
        installed = instrument_asyncpg_sanitized(_ReadOnlyTracer(real))

        # Assert
        assert installed is False
        assert not _instrumented_targets(), (
            "an un-replaceable tracer must leave asyncpg fully un-instrumented"
        )

    def test_no_tracer_attribute_removes_instrumentation(
        self,
        restore_asyncpg_instrumentation: AsyncPGInstrumentor,
    ) -> None:
        """A relocated tracer means the real one is un-sanitized: remove."""
        # Arrange
        real = restore_asyncpg_instrumentation
        real.instrument()
        assert _instrumented_targets()

        class _NoTracer:
            is_instrumented_by_opentelemetry = True

            def __init__(self, delegate: AsyncPGInstrumentor) -> None:
                self._delegate = delegate

            def uninstrument(self) -> None:
                self._delegate.uninstrument()

        # Act
        installed = instrument_asyncpg_sanitized(_NoTracer(real))

        # Assert
        assert installed is False
        assert not _instrumented_targets()

    def test_unremovable_unsafe_instrumentation_raises(self) -> None:
        """Cannot sanitize AND cannot remove: fail loudly, never silently.

        The wrappers here are genuine, and the stand-in's ``uninstrument``
        is a no-op, so the verification finds surviving wrappers -- the one
        state where a reassuring return value would be actively harmful.
        """
        # Arrange
        real = AsyncPGInstrumentor()
        was_instrumented = real.is_instrumented_by_opentelemetry
        if not was_instrumented:
            real.instrument()
        assert _instrumented_targets()

        class _Unremovable:
            is_instrumented_by_opentelemetry = True
            _tracer = None

            def uninstrument(self) -> None:
                """Deliberately does nothing, leaving the wrappers in place."""

        # Act / Assert
        try:
            with pytest.raises(SanitizerUnsafeError, match="could not be removed"):
                instrument_asyncpg_sanitized(_Unremovable())
        finally:
            if not was_instrumented:
                with contextlib.suppress(Exception):
                    real.uninstrument()

    def test_partial_install_leaves_no_wrapper_behind(
        self,
        restore_asyncpg_instrumentation: AsyncPGInstrumentor,
    ) -> None:
        """Every one of the nine wrapped targets must be gone after a failure."""
        # Arrange
        real = restore_asyncpg_instrumentation
        real.instrument()
        before = _instrumented_targets()
        assert len(before) == 9, f"expected all nine targets wrapped, got {sorted(before)}"

        class _NoTracer:
            is_instrumented_by_opentelemetry = True

            def __init__(self, delegate: AsyncPGInstrumentor) -> None:
                self._delegate = delegate

            def uninstrument(self) -> None:
                self._delegate.uninstrument()

        # Act
        instrument_asyncpg_sanitized(_NoTracer(real))

        # Assert -- named individually so a partial removal is legible.
        surviving = _instrumented_targets()
        assert surviving == set(), f"wrappers survived on {sorted(surviving)}"


class TestBaseExceptionPassthrough:
    """Cancellation and interpreter shutdown are not database failures.

    The wrapper catches ``Exception`` only, matching the OTel SDK's own
    context exit and the instrumentor's ``except Exception``. Stamping a
    db failure event onto a cancelled span would invent a failure the
    driver never reported.
    """

    @pytest.mark.parametrize(
        "exc_type",
        [
            pytest.param(asyncio.CancelledError, id="CancelledError"),
            pytest.param(KeyboardInterrupt, id="KeyboardInterrupt"),
            pytest.param(SystemExit, id="SystemExit"),
        ],
    )
    def test_base_exception_propagates_without_a_forced_failure(
        self,
        exc_type: type[BaseException],
        in_memory_tracer: tuple[Any, InMemoryExporter],
    ) -> None:
        # Arrange
        tracer, exporter = in_memory_tracer
        wrapped = SanitizingTracer(tracer)

        # Act
        with pytest.raises(exc_type), wrapped.start_as_current_span("probe"):
            raise exc_type

        # Assert
        spans = [span for span in exporter.spans if span.name == "probe"]
        assert len(spans) == 1, "the span must still end"
        span = spans[0]
        assert span.events == (), (
            f"a {exc_type.__name__} must not be recorded as a db failure event"
        )
        assert span.status is not None
        assert span.status.status_code is not trace.StatusCode.ERROR, (
            f"a {exc_type.__name__} must not force an ERROR status"
        )

    def test_a_cancelled_error_wrapping_a_driver_error_still_passes_through(
        self,
        in_memory_tracer: tuple[Any, InMemoryExporter],
    ) -> None:
        """Cancellation semantics win over classification, deliberately.

        A CancelledError whose context is a driver error is still a
        cancellation: the SDK would record nothing, and inventing an ERROR
        status here would misreport a task teardown as a DB fault. The
        driver text is not exported either way, which is what matters.
        """
        # Arrange
        tracer, exporter = in_memory_tracer
        wrapped = SanitizingTracer(tracer)

        def _cancel_after_driver_error() -> None:
            try:
                raise asyncpg.exceptions.InsufficientPrivilegeError(
                    "permission denied planted-cancel-marker"
                )
            except asyncpg.PostgresError as exc:
                raise asyncio.CancelledError from exc

        # Act
        with pytest.raises(asyncio.CancelledError), wrapped.start_as_current_span("probe"):
            _cancel_after_driver_error()

        # Assert
        spans = [span for span in exporter.spans if span.name == "probe"]
        assert len(spans) == 1
        span = spans[0]
        assert span.events == ()
        assert "planted-cancel-marker" not in _span_text(span)


class TestProductionWiring:
    """``_wire_instrumentors`` must route asyncpg through the sanitizer."""

    def test_wire_instrumentors_installs_the_sanitizing_tracer(
        self,
        restore_asyncpg_instrumentation: AsyncPGInstrumentor,
    ) -> None:
        """Pins the call site, not just the helper.

        The helper can be correct while production still calls bare
        ``instrument()``; this asserts the app path actually uses it.
        """
        # Arrange
        from fastapi import FastAPI

        from gubbi.telemetry import _wire_instrumentors

        instrumentor = restore_asyncpg_instrumentation

        # Act
        _wire_instrumentors(FastAPI())

        # Assert
        assert isinstance(
            instrumentor._tracer, SanitizingTracer
        ), (  # asserting the wiring installed the wrapper
            "gubbi.telemetry._wire_instrumentors must install the sanitizing tracer"
        )

    def test_wire_instrumentors_propagates_sanitizer_unsafe(
        self,
        restore_asyncpg_instrumentation: AsyncPGInstrumentor,
    ) -> None:
        """The unsafe state must abort wiring, not be degraded like the rest.

        Every other instrumentor in ``_wire_instrumentors`` is wrapped in
        ``except Exception`` and downgraded to a warning. This one must
        not be: the condition it signals is "asyncpg is instrumented, is
        NOT sanitized, and could not be un-instrumented", and continuing
        past it means exporting driver DETAIL on every failing audit write
        for the life of the process.

        Asserting the re-raise at the CALL SITE is what makes it real --
        the helper raising correctly proves nothing if the caller swallows
        it two lines later.
        """
        # Arrange
        from fastapi import FastAPI

        from gubbi import telemetry as telemetry_module

        # Act / Assert
        with (
            patch.object(
                telemetry_module,
                "_wire_instrumentors",
                telemetry_module._wire_instrumentors,
            ),
            patch(
                "gubbi.telemetry.asyncpg_sanitized.instrument_asyncpg_sanitized",
                side_effect=SanitizerUnsafeError("wrappers remain"),
            ),
            pytest.raises(SanitizerUnsafeError, match="wrappers remain"),
        ):
            telemetry_module._wire_instrumentors(FastAPI())

    def test_wire_instrumentors_still_degrades_an_ordinary_asyncpg_failure(
        self,
        restore_asyncpg_instrumentation: AsyncPGInstrumentor,
    ) -> None:
        """Control: only the unsafe state aborts; other failures still warn.

        Without this, the test above would also pass if ``_wire_instrumentors``
        had simply stopped catching anything.
        """
        # Arrange
        from fastapi import FastAPI

        from gubbi import telemetry as telemetry_module

        # Act -- an ordinary failure must NOT propagate.
        with patch(
            "gubbi.telemetry.asyncpg_sanitized.instrument_asyncpg_sanitized",
            side_effect=RuntimeError("some other instrumentor problem"),
        ):
            telemetry_module._wire_instrumentors(FastAPI())

        # Assert -- reaching here without raising IS the assertion.
