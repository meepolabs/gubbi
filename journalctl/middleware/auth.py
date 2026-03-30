"""Bearer token authentication for the MCP endpoint.

Validates two types of tokens:
1. Legacy static API key (for Claude CLI / Desktop)
2. OAuth 2.0 access tokens (for claude.ai web/mobile)

Uses a lightweight ASGI wrapper (NOT BaseHTTPMiddleware) to avoid
buffering responses — BaseHTTPMiddleware breaks SSE streaming
required by MCP's streamable HTTP transport.
"""

from __future__ import annotations

import secrets
from collections.abc import Callable

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from journalctl.config import get_settings
from journalctl.oauth.constants import MAX_BEARER_TOKEN_LEN


class BearerAuthMiddleware:
    """ASGI middleware that enforces Bearer token authentication.

    Validates tokens in two modes:
    1. Direct match against the legacy JOURNAL_API_KEY
    2. Delegated to an optional token_validator callable

    This is NOT BaseHTTPMiddleware — it passes through the raw
    ASGI interface without buffering, so SSE streaming works.
    """

    def __init__(
        self,
        app: ASGIApp,
        token_validator: Callable[[str], bool] | None = None,
    ) -> None:
        self.app = app
        self.token_validator = token_validator

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope)
        auth_header = request.headers.get("authorization", "")

        if not auth_header.startswith("Bearer "):
            response = JSONResponse(
                {"error": "Missing or invalid Authorization header"},
                status_code=401,
                headers={"WWW-Authenticate": 'Bearer error="invalid_token"'},
            )
            await response(scope, receive, send)
            return

        token = auth_header[7:]  # Strip "Bearer "

        if len(token) > MAX_BEARER_TOKEN_LEN:
            response = JSONResponse(
                {"error": "Invalid token"},
                status_code=401,
                headers={"WWW-Authenticate": 'Bearer error="invalid_token"'},
            )
            await response(scope, receive, send)
            return

        settings = get_settings()

        # Mode 1: Legacy API key (timing-safe comparison)
        if secrets.compare_digest(token, settings.api_key):
            await self.app(scope, receive, send)
            return

        # Mode 2: Delegated token validation (OAuth)
        if self.token_validator is not None and self.token_validator(token):
            await self.app(scope, receive, send)
            return

        response = JSONResponse(
            {"error": "Invalid or expired token"},
            status_code=401,
            headers={"WWW-Authenticate": 'Bearer error="invalid_token"'},
        )
        await response(scope, receive, send)
