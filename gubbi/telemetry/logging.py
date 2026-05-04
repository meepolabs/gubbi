"""Structured logging with JSON-per-line output and correlation_id support.

Provides:
    - StructuredLogFormatter: a logging.Formatter that emits JSON-per-line.
    - set_correlation_id / get_correlation_id: contextvar-based helpers
      for the current request's correlation_id.

Schema (one JSON object per line):
    timestamp (UTC ISO 8601)
    level
    service ("gubbi")
    correlation_id
    trace_id (when OTel span active)
    span_id
    event (string event name)
    attributes (nested object)
"""

from __future__ import annotations

import json
import logging
import os
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

from opentelemetry import trace

# ---------------------------------------------------------------------------
# Correlation ID context var
# ---------------------------------------------------------------------------

_correlation_id_var: ContextVar[str | None] = ContextVar("correlation_id", default=None)

logger = logging.getLogger(__name__)


def set_correlation_id(cid: str) -> None:
    """Set the request-scoped correlation_id in context."""
    _correlation_id_var.set(cid)


def get_correlation_id() -> str | None:
    """Return the current request's correlation_id, or None."""
    return _correlation_id_var.get()


# ---------------------------------------------------------------------------
# OTel trace helpers
# ---------------------------------------------------------------------------


def _get_otel_ids() -> tuple[str | None, str | None]:
    """Return (trace_id, span_id) from the current OTel span, if any."""
    try:
        span = trace.get_current_span()
        span_context = span.get_span_context()
        if span_context and span_context.is_valid:
            trace_id_hex = format(span_context.trace_id, "032x")
            span_id_hex = format(span_context.span_id, "016x")
            return trace_id_hex, span_id_hex
    except Exception:
        logger.debug("Failed to read OTel span context", exc_info=True)
    return None, None


# ---------------------------------------------------------------------------
# StructuredLogFormatter
# ---------------------------------------------------------------------------


class StructuredLogFormatter(logging.Formatter):
    """JSON-per-line formatter with a stable schema.

    The ``event`` field is populated from the log message string.
    Extra keyword arguments (via ``extra`` dict or ``**kwargs``) are
    placed in the ``attributes`` nested object.

    Usage::

        logger = logging.getLogger("gubbi")
        logger.info("tool.call", extra={"tool.name": "journal_append_entry"})

    This produces::

        {"timestamp": "2026-04-29T12:00:00Z", "level": "INFO",
         "service": "gubbi", "correlation_id": "...",
         "trace_id": "...", "span_id": "...",
         "event": "tool.call",
         "attributes": {"tool.name": "journal_append_entry"}}
    """

    def __init__(self) -> None:
        super().__init__()
        self._service_name = os.environ.get("OTEL_SERVICE_NAME", "gubbi")

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as a single JSON line."""
        timestamp = datetime.fromtimestamp(record.created, tz=UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        correlation_id = get_correlation_id()
        trace_id, span_id = _get_otel_ids()

        # Build the attributes dict from the record's extra data
        attributes: dict[str, Any] = {}
        # Copy all extra fields from the record
        for key, value in record.__dict__.items():
            if key in (
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "exc_info",
                "exc_text",
                "stack_info",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
            ):
                continue
            attributes[key] = value

        # If record.msg looks like an event name (no spaces, no %-formatting),
        # use it as 'event'. Otherwise put it in attributes.
        event: str = record.msg
        if record.args:
            # The message is a format string; let the base class format it
            try:
                event = record.getMessage()
            except Exception:
                event = record.msg

        entry: dict[str, Any] = {
            "timestamp": timestamp,
            "level": record.levelname,
            "service": self._service_name,
            "correlation_id": correlation_id,
            "trace_id": trace_id,
            "span_id": span_id,
            "event": event,
        }

        if attributes:
            entry["attributes"] = attributes

        return json.dumps(entry, default=str, ensure_ascii=False)


def configure_structured_logging() -> None:
    """Configure the root logger to use StructuredLogFormatter.

    Call once during app startup. Removes all existing handlers and
    adds a StreamHandler (stderr) with the JSON formatter.
    """
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Remove existing handlers
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler()
    handler.setFormatter(StructuredLogFormatter())
    root.addHandler(handler)
