"""Correlation ID middleware for journalctl (TASK-03.19).

Extracts ``X-Correlation-ID`` from inbound HTTP requests, generates a
UUID4 if absent, stores it in a ContextVar for structured logging and
span attributes, and echoes it back in the response header.

Also sets the ``correlation_id`` attribute on the current OTel span.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from opentelemetry import trace
from starlette.types import ASGIApp, Receive, Scope, Send

from journalctl.telemetry.attrs import SpanNames, safe_set_attributes
from journalctl.telemetry.logging import _correlation_id_var

logger = logging.getLogger(__name__)

_CORRELATION_ID_HEADER = "X-Correlation-ID"


class CorrelationIDMiddleware:
    """ASGI middleware that manages correlation_id propagation.

    Must be mounted early in the middleware stack so the correlation_id
    is available for logging and span attribution through the rest of
    the request lifecycle.

    This middleware uses raw ASGI (NOT BaseHTTPMiddleware) to avoid
    buffering SSE streaming responses.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        correlation_id = self._extract_or_generate(scope)
        token = _correlation_id_var.set(correlation_id)

        # Set the correlation_id on the current OTel span via safe_set_attributes
        try:
            span = trace.get_current_span()
            if span and span.is_recording():
                safe_set_attributes(
                    SpanNames.HTTP_REQUEST,
                    span,
                    {"correlation_id": correlation_id},
                )
        except Exception:
            logger.debug("Failed to set correlation_id on OTel span", exc_info=True)

        # Wrap send to inject the response header
        original_send = send

        async def send_wrapper(message: dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                headers = message.get("headers", [])
                # Check if header already set
                header_present = any(
                    k.lower() == _CORRELATION_ID_HEADER.lower() for k, v in headers
                )
                if not header_present:
                    # headers is a list of (bytes, bytes) tuples
                    new_headers = list(headers)
                    new_headers.append(
                        (_CORRELATION_ID_HEADER.encode("latin-1"), correlation_id.encode("latin-1"))
                    )
                    message["headers"] = new_headers
            await original_send(message)

        try:
            await self.app(scope, receive, send_wrapper)  # type: ignore[arg-type]
        finally:
            _correlation_id_var.reset(token)

    @staticmethod
    def _extract_or_generate(scope: Scope) -> str:
        """Extract X-Correlation-ID from request headers or generate a new UUID4."""
        headers = dict(scope.get("headers", []))
        raw: bytes | None = headers.get(_CORRELATION_ID_HEADER.lower().encode("latin-1"))
        if raw is not None:
            try:
                decoded = raw.decode("latin-1").strip()
                if decoded:
                    return decoded
            except (UnicodeDecodeError, ValueError):
                pass
        # Generate a new UUID4
        return str(uuid.uuid4())
