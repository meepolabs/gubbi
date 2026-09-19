"""Sanitized asyncpg auto-instrumentation.

``AsyncPGInstrumentor`` wraps ``Connection.execute`` and friends in a
client span and lets the exception escape the span's context manager. The
SDK's context exit then records the driver exception with its default
shape -- ``exception.message`` and ``exception.stacktrace`` on an
``exception`` event, plus a status description built from ``str(exc)``.

For a failing ``audit_log`` INSERT that text is the server's own message
and ``DETAIL`` block, which quote the rejected row: the actor id, the
originating IP and the User-Agent. Measured against PostgreSQL 17, a
duplicate-key rejection inside an ``audit.write`` trace put the full
``DETAIL`` on the child span's exception event and status description
while the parent span stayed clean -- so sanitizing the writer's own span
is not sufficient.

This module installs the instrumentor against a tracer that turns the
SDK's two auto-record flags off and records the failure itself with
bounded attributes. The standard ``exception`` event name is kept so
backend queries and alerting that key on it keep working; only the
unbounded attributes are withheld.

Everything else is delegated to the real tracer from the configured
provider, so sampling, resource, span kind, attributes, context
propagation and parent linkage are exactly what the instrumentor would
have produced.
"""

from __future__ import annotations

import contextlib
import importlib
import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Final

import wrapt
from opentelemetry.trace import Status, StatusCode

from gubbi.telemetry.sanitized_errors import is_driver_caused, safe_error_fields

if TYPE_CHECKING:
    from collections.abc import Iterator

    from opentelemetry.trace import Span, Tracer

__all__: list[str] = [
    "SanitizerUnsafeError",
    "SanitizingTracer",
    "instrument_asyncpg_sanitized",
]

_logger = logging.getLogger(__name__)


class SanitizerUnsafeError(RuntimeError):
    """Asyncpg instrumentation can neither be sanitized nor removed.

    Raised instead of returning, so a process that would export driver
    message/DETAIL on every failing audit write fails visibly at startup
    rather than running in that state.
    """


# Internal alias kept short at the call sites below.
_SanitizerUnsafe = SanitizerUnsafeError

# The exact call targets ``AsyncPGInstrumentor._instrument`` wraps, as
# (module, class, attribute). Mirrored from the instrumentor's own lists
# so a removal can be VERIFIED rather than trusted. A target this tuple
# misses would be a wrapper the verification cannot see -- re-read
# ``opentelemetry/instrumentation/asyncpg/__init__.py`` on a version bump.
_WRAPPED_TARGETS: Final[tuple[tuple[str, str, str], ...]] = (
    ("asyncpg.connection", "Connection", "execute"),
    ("asyncpg.connection", "Connection", "executemany"),
    ("asyncpg.connection", "Connection", "fetch"),
    ("asyncpg.connection", "Connection", "fetchval"),
    ("asyncpg.connection", "Connection", "fetchrow"),
    ("asyncpg.cursor", "Cursor", "fetch"),
    ("asyncpg.cursor", "Cursor", "forward"),
    ("asyncpg.cursor", "Cursor", "fetchrow"),
    ("asyncpg.cursor", "CursorIterator", "__anext__"),
)

# The OTel semantic-convention event name for a recorded exception. Kept
# rather than renamed: dashboards and alerts key on it, and the point
# here is to bound the event's attributes, not to hide the failure.
_EXCEPTION_EVENT_NAME = "exception"


def _sanitized_exception_attributes(exc: BaseException) -> dict[str, str]:
    """Return bounded ``exception.*`` attributes for *exc*.

    ``exception.type`` is the fully-qualified class name the SDK itself
    would record. ``db.sqlstate`` is added when one is available. The
    unbounded ``exception.message`` / ``exception.stacktrace`` pair is
    never included.
    """
    module = type(exc).__module__
    qualified = f"{module}.{type(exc).__name__}" if module else type(exc).__name__
    attributes = {"exception.type": qualified}
    fields = safe_error_fields(exc)
    sqlstate = fields.get("db_sqlstate")
    if sqlstate is not None:
        attributes["db.sqlstate"] = sqlstate
    return attributes


class SanitizingTracer:
    """Delegating tracer that bounds what a driver exception records.

    Wraps only ``start_as_current_span`` -- the single entry point the
    asyncpg instrumentor uses. ``start_span`` and anything else pass
    straight through to the wrapped tracer, so a future instrumentor
    change degrades to the default behavior rather than breaking.
    """

    def __init__(self, inner: Tracer) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    @contextmanager
    def start_as_current_span(self, name: str, *args: Any, **kwargs: Any) -> Iterator[Span]:
        """Open *name* with the SDK's auto-recording off, recording failures here.

        A non-driver exception keeps the SDK's shape (message and
        stacktrace on the event, a described status), since this codebase
        owns those strings. A driver-caused one gets the bounded event
        and a description-free ERROR status. Either way the exception
        propagates and the span ends.

        ``Exception``, not ``BaseException``: the SDK's own context exit
        ignores non-``Exception`` raises, and both the OTel SDK and the
        asyncpg instrumentor treat a cancellation or an interpreter
        shutdown as "not a database failure". Catching ``BaseException``
        here would stamp an ERROR status and a db failure event onto a
        span that was merely cancelled, inventing a failure the driver
        never reported. Those raises pass straight through, unrecorded.
        """
        kwargs["record_exception"] = False
        kwargs["set_status_on_exception"] = False
        with self._inner.start_as_current_span(name, *args, **kwargs) as span:
            try:
                yield span
            except Exception as exc:
                if is_driver_caused(exc):
                    span.add_event(
                        _EXCEPTION_EVENT_NAME,
                        attributes=_sanitized_exception_attributes(exc),
                    )
                    span.set_status(Status(StatusCode.ERROR))
                else:
                    span.record_exception(exc)
                    span.set_status(
                        Status(StatusCode.ERROR, description=f"{type(exc).__name__}: {exc}"),
                    )
                raise


def instrument_asyncpg_sanitized(instrumentor: Any) -> bool:
    """Instrument asyncpg so its client spans withhold driver text.

    Returns True when the sanitizing tracer is in place. Returns False
    ONLY when asyncpg is left un-instrumented, so a False answer never
    means "still leaking": the two states are sanitized-and-instrumented,
    or not-instrumented-at-all.

    ``AsyncPGInstrumentor`` is a singleton whose ``instrument()`` is a
    no-op once anything has already instrumented asyncpg (a bare
    ``opentelemetry-distro`` autoload, or a second ``configure_otel`` in
    one process), so the already-instrumented case swaps the tracer on
    the live instrumentor instead of re-instrumenting.

    Fail-closed: if the tracer cannot be replaced -- a future
    instrumentor that keeps its tracer somewhere else, or one whose
    ``_tracer`` attribute refuses assignment -- the instrumentation is
    REMOVED rather than left in its default, leaking shape. Losing
    per-statement client spans is a monitoring regression; keeping them
    exports the DETAIL of every failing audit write. The removal is
    verified against the driver modules, and a removal that cannot be
    confirmed raises so startup surfaces it.
    """
    try:
        return _install(instrumentor)
    except _SanitizerUnsafe:
        raise
    except Exception:
        # The wrap itself failed for an unanticipated reason. Do not
        # leave default instrumentation behind on the strength of a
        # guess about why.
        _logger.warning("asyncpg_sanitize_failed", exc_info=True)
        _force_uninstrument(instrumentor)
        return False


def _install(instrumentor: Any) -> bool:
    """Install the sanitizing tracer, or remove instrumentation entirely."""
    already = bool(getattr(instrumentor, "is_instrumented_by_opentelemetry", False))
    if not already:
        instrumentor.instrument()

    inner = getattr(instrumentor, "_tracer", None)
    if isinstance(inner, SanitizingTracer):
        return True

    if inner is None:
        # No tracer attribute to wrap: the instrumentor's internals have
        # moved, and whatever tracer it does use is un-sanitized.
        _logger.warning(
            "asyncpg_sanitize_unsupported",
            extra={"reason": "instrumentor exposed no tracer to wrap"},
        )
        _force_uninstrument(instrumentor)
        return False

    try:
        instrumentor._tracer = SanitizingTracer(inner)  # the instrumentor exposes no public setter
    except Exception:
        _logger.warning("asyncpg_sanitize_tracer_readonly", exc_info=True)
        _force_uninstrument(instrumentor)
        return False

    if not isinstance(getattr(instrumentor, "_tracer", None), SanitizingTracer):
        # The assignment was accepted but did not stick (a property, a
        # __slots__ shadow, a descriptor). Verify, never assume.
        _logger.warning(
            "asyncpg_sanitize_not_applied",
            extra={"reason": "tracer assignment did not take effect"},
        )
        _force_uninstrument(instrumentor)
        return False
    return True


def _force_uninstrument(instrumentor: Any) -> None:
    """Remove asyncpg instrumentation, raising if any wrapper survives.

    Called only on the unsafe paths above. A surviving wrapper means the
    process would keep emitting un-sanitized client spans, which is the
    condition this module exists to prevent -- so it is raised rather
    than logged.
    """
    with contextlib.suppress(Exception):
        instrumentor.uninstrument()
    with contextlib.suppress(Exception):
        instrumentor._is_instrumented_by_opentelemetry = False

    surviving = _instrumented_targets()
    if surviving:
        msg = (
            "asyncpg instrumentation could not be sanitized and could not be removed; "
            f"un-sanitized wrappers remain on {sorted(surviving)}. Failing startup "
            "rather than exporting driver message/DETAIL on audit-write client spans."
        )
        raise _SanitizerUnsafe(msg)
    _logger.warning(
        "asyncpg_instrumentation_removed",
        extra={"reason": "could not sanitize; per-statement client spans disabled"},
    )


def _instrumented_targets() -> set[str]:
    """Return the driver methods currently carrying an instrumentor wrapper.

    Detection is by wrapt's ``BoundFunctionWrapper``, which is what
    ``wrap_function_wrapper`` leaves on the class attribute. A
    ``__wrapped__`` check would be wrong here: several cursor methods
    carry an unrelated native decorator, so they look wrapped even on a
    clean interpreter.
    """
    surviving: set[str] = set()
    for module_name, class_name, attribute in _WRAPPED_TARGETS:
        try:
            module = importlib.import_module(module_name)
            owner = getattr(module, class_name)
        except (ImportError, AttributeError) as exc:
            # A target this asyncpg version does not have cannot be
            # carrying a wrapper, so it is not a leak -- but a target that
            # silently vanished means _WRAPPED_TARGETS has drifted from
            # the instrumentor, which the next reader needs to know.
            _logger.debug(
                "asyncpg_wrapper_target_absent",
                extra={"target": f"{module_name}.{class_name}.{attribute}", "error": str(exc)},
            )
            continue
        candidate = owner.__dict__.get(attribute, getattr(owner, attribute, None))
        if isinstance(candidate, wrapt.BoundFunctionWrapper | wrapt.FunctionWrapper):
            surviving.add(f"{class_name}.{attribute}")
    return surviving
