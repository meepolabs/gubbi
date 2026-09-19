"""Telemetry fields for an exception whose text cannot be trusted.

A PostgreSQL error raised by asyncpg carries the server's own message
plus its ``DETAIL`` / ``HINT`` blocks, and both can quote the failing
statement's values back. On the audit-write path those values are the
actor id, the originating IP and the User-Agent, so the default
telemetry shapes -- ``span.record_exception`` (which writes
``exception.message`` and ``exception.stacktrace``), a status
description built from ``str(exc)``, ``logger.exception``, or an
``error=str(exc)`` field -- export identity-bearing text to the trace
backend.

gubbi-common sanitizes its own ``audit.write`` span and re-raises. This
module is the caller-side half: the exception class name plus a
shape-validated SQLSTATE, which is enough to tell a privilege rejection
from a constraint violation, and nothing the server supplied as text.

Scope is deliberately narrow. Only a database error, or a wrapper that
carries one in its ``__cause__`` / ``__context__`` chain, is treated as
untrusted. Everything else -- an LLM provider error, a ``ValueError``
from our own validation -- keeps the SDK's default recording including
the exception message and a status description, because this codebase
owns those strings and an operator needs them.

The chain walk matters because a translated exception re-exports the
text it wrapped: ``DatabaseUnavailable(str(exc))`` in
:mod:`gubbi.storage.connection` interpolates the driver message into its
own ``str()``, so classifying only the outermost type would let it
through.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import asyncpg
from opentelemetry.trace import Status, StatusCode

if TYPE_CHECKING:
    from opentelemetry.trace import Span

__all__: list[str] = [
    "DB_ERROR_EVENT_NAME",
    "DRIVER_ERROR_TYPES",
    "is_driver_caused",
    "record_exception_sanitized",
    "safe_error_fields",
]

# A distinct event name keeps the sanitized record apart from the SDK's
# ``exception`` event in the backend: an ``exception`` event carries the
# message and stacktrace attributes, this one never can.
DB_ERROR_EVENT_NAME: Final[str] = "db.error"

# SQLSTATE is exactly five characters drawn from digits and uppercase
# ASCII letters (SQL standard, class + subclass). A value of any other
# shape is not a SQLSTATE and is dropped rather than forwarded: this
# attribute is the one place a driver-supplied string reaches telemetry,
# so its shape is validated instead of trusted. Mirrors the validation
# gubbi-common applies inside the ``audit.write`` span; the rule is
# duplicated rather than imported because upstream keeps it private.
_SQLSTATE_LENGTH: Final[int] = 5
_SQLSTATE_ALPHABET: Final[frozenset[str]] = frozenset("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# asyncpg exception classes whose ``str()`` can carry server- or
# driver-supplied text for the failing statement.
#
# ``PostgresError`` is the server-side set -- where SQLSTATE, DETAIL and
# HINT live -- and both connectivity classes the storage layer translates
# (``PostgresConnectionError``, ``CannotConnectNowError``) derive from it.
#
# ``InterfaceError`` and ``InternalClientError`` are included as a
# fail-closed measure rather than on evidence of a leak: measured against
# a live PostgreSQL 17 on the audit-write execute path their messages are
# fixed driver strings ("connection is closed", "the server expects 2
# arguments for this query, 1 was passed") and their stacktraces carry no
# statement text. They are driver-owned strings all the same, and a
# future driver version is free to start interpolating -- the cost of
# withholding them is one fixed string in a log line, the cost of a wrong
# guess is exported identity data.
#
# Public because the HTTP boundary must register a handler for exactly
# this set: a class classified here but not registered in
# ``gubbi.main.register_exception_handlers`` takes the catch-all route and
# reaches the server span un-sanitized. One source, no drift.
#
# ``PostgresWarning`` stays out: asyncpg constructs it as a connection log
# message (``PostgresLogMessage.new``) and never raises it on the execute
# path, so it cannot reach an audit-write except block at all. Read
# ``asyncpg.exceptions._base.PostgresLogMessage`` before adding it.
DRIVER_ERROR_TYPES: Final[tuple[type[Exception], ...]] = (
    asyncpg.PostgresError,
    asyncpg.InterfaceError,
    asyncpg.InternalClientError,
)

# How far to follow ``__cause__`` / ``__context__``. A translated
# exception is wrapped once or twice in practice. Reaching this bound
# means the chain could not be characterised, which is treated as
# driver-caused (see :func:`is_driver_caused`) rather than assumed
# clean -- an unbounded walk is refused, but so is a false negative.
_MAX_CHAIN_DEPTH: Final[int] = 10


def _walk_chain(exc: BaseException) -> tuple[list[BaseException], bool]:
    """Return *exc*'s cause/context chain and whether it terminated.

    The walk stops on a repeated link (a cycle) or at
    :data:`_MAX_CHAIN_DEPTH`. ``exhausted`` is True only in the second
    case: a cycle has been fully characterised, whereas a depth cut-off
    leaves links unexamined.
    """
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None:
        if id(current) in seen:
            return chain, False
        if len(chain) >= _MAX_CHAIN_DEPTH:
            return chain, True
        seen.add(id(current))
        chain.append(current)
        # Identity, not truthiness: an exception subclass may define
        # ``__bool__`` or ``__len__`` and evaluate falsey, and
        # ``cause or context`` would then skip past a REAL cause to the
        # context (or to None, truncating the walk). Measured: a falsey
        # link holding an asyncpg error as its own cause made the whole
        # chain read as non-driver. ``__cause__`` is either an exception
        # or None; None is the only value that means "no cause".
        cause = current.__cause__
        current = cause if cause is not None else current.__context__
    return chain, False


def _chain(exc: BaseException) -> list[BaseException]:
    """Return *exc* plus its cause/context ancestors, bounded and cycle-safe."""
    return _walk_chain(exc)[0]


def _sqlstate_of(exc: BaseException) -> str | None:
    """Return *exc*'s own SQLSTATE when it has the exact standard shape."""
    candidate = getattr(exc, "sqlstate", None)
    if not isinstance(candidate, str) or len(candidate) != _SQLSTATE_LENGTH:
        return None
    if not all(char in _SQLSTATE_ALPHABET for char in candidate):
        return None
    return candidate


def _sqlstate_or_none(exc: BaseException) -> str | None:
    """Return the first standard-shaped SQLSTATE in *exc*'s chain."""
    for link in _chain(exc):
        sqlstate = _sqlstate_of(link)
        if sqlstate is not None:
            return sqlstate
    return None


def is_driver_caused(exc: BaseException) -> bool:
    """Whether *exc* or anything it wraps can carry driver-supplied text.

    True for a database exception and for a wrapper that has one in its
    cause/context chain -- including a wrapper carrying no SQLSTATE at
    all, such as a connectivity translation, whose own message is still
    built from the driver's.

    Also true when the chain hit :data:`_MAX_CHAIN_DEPTH` before
    terminating: past the bound the remaining links are unexamined, so
    "no driver error found" is not a finding. Failing closed here costs a
    message on a pathologically-wrapped non-database error; failing open
    would export the driver text that motivated the bound.
    """
    chain, exhausted = _walk_chain(exc)
    if exhausted:
        return True
    return any(isinstance(link, DRIVER_ERROR_TYPES) for link in chain)


def safe_error_fields(exc: BaseException) -> dict[str, str]:
    """Return the log fields that describe *exc* without its own text.

    ``error_type`` is always present. ``db_sqlstate`` is present only
    when the exception or something it wraps carries a standard-shaped
    SQLSTATE.
    """
    fields = {"error_type": type(exc).__name__}
    sqlstate = _sqlstate_or_none(exc)
    if sqlstate is not None:
        fields["db_sqlstate"] = sqlstate
    return fields


def record_exception_sanitized(span: Span, exc: BaseException) -> None:
    """Mark *span* failed, withholding driver-supplied text.

    A driver-caused exception (see :func:`is_driver_caused`) is recorded
    as a :data:`DB_ERROR_EVENT_NAME` event holding only its class name
    and a shape-validated SQLSTATE where one exists, with an error
    status carrying no description.

    Anything else keeps the SDK's default recording -- the ``exception``
    event with its message and stacktrace, and a status description of
    ``type: message`` -- because this codebase owns those strings.

    The caller re-raises: this records, it never swallows.
    """
    if not is_driver_caused(exc):
        span.record_exception(exc)
        span.set_status(
            Status(StatusCode.ERROR, description=f"{type(exc).__name__}: {exc}"),
        )
        return

    attributes = {"exception.type": type(exc).__name__}
    sqlstate = _sqlstate_or_none(exc)
    if sqlstate is not None:
        attributes["db.sqlstate"] = sqlstate
    span.add_event(DB_ERROR_EVENT_NAME, attributes=attributes)
    span.set_status(Status(StatusCode.ERROR))
