"""Bearer token authentication for the MCP endpoint.

Validates two types of tokens:
1. Legacy static API key (for Claude CLI / Desktop)
2. Hydra OAuth 2.0 access tokens (for web/mobile clients)
3. Legacy OAuth access tokens via external token_validator callback

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

from journalctl.auth.hydra import HydraIntrospector, HydraInvalidToken, HydraUnreachable
from journalctl.core.auth_context import current_user_id
from journalctl.oauth.constants import MAX_BEARER_TOKEN_LEN


def _unauthorized(message: str) -> JSONResponse:
    return JSONResponse(
        {"error": message},
        status_code=401,
        headers={"WWW-Authenticate": 'Bearer error="invalid_token"'},
    )


def _forbidden(scope: str) -> JSONResponse:
    return JSONResponse(
        {"error": "insufficient_scope"},
        status_code=403,
        headers={"WWW-Authenticate": f'Bearer error="insufficient_scope",required_scope="{scope}"'},
    )


def _service_unavailable() -> JSONResponse:
    return JSONResponse(
        {"error": "auth service unavailable"},
        status_code=503,
        headers={"Retry-After": "5"},
    )


class BearerAuthMiddleware:
    """ASGI middleware that enforces Bearer token authentication.

    Validates tokens in three modes:
    1. Direct match against the legacy API key
    2. Hydra introspection for Ory Atlassian tokens
    3. Delegated token validation via legacy_token_validator callback

    This is NOT BaseHTTPMiddleware — it passes through the raw
    ASGI interface without buffering, so SSE streaming works.
    """

    def __init__(
        self,
        app: ASGIApp,
        api_key: str,
        introspector: HydraIntrospector | None = None,
        required_scope: str = "journal",
        legacy_token_validator: Callable[[str], bool] | None = None,
    ) -> None:
        self.app = app
        self.api_key = api_key
        self.introspector = introspector
        self.required_scope = required_scope
        self.legacy_token_validator = legacy_token_validator

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope)
        auth_header = request.headers.get("authorization", "")

        if not auth_header:
            await _unauthorized("Missing or invalid Authorization header")(scope, receive, send)
            return

        if not auth_header.lower().startswith("bearer "):
            await _unauthorized("Missing or invalid Authorization header")(scope, receive, send)
            return

        token = auth_header[7:]

        if len(token) > MAX_BEARER_TOKEN_LEN:
            await _unauthorized("Invalid token")(scope, receive, send)
            return

        # Mode 1: Legacy API key (timing-safe comparison)
        if secrets.compare_digest(token, self.api_key):
            await self.app(scope, receive, send)
            return

        # Mode 2: Hydra introspection (Ory access tokens)
        if token.startswith("ory_at_") and self.introspector is not None:
            try:
                claims = await self.introspector.introspect(token)
            except HydraUnreachable:
                await _service_unavailable()(scope, receive, send)
                return
            except HydraInvalidToken:
                await _unauthorized("Invalid or expired token")(scope, receive, send)
                return

            scopes = claims.scope.split()
            if self.required_scope not in scopes:
                await _forbidden(self.required_scope)(scope, receive, send)
                return

            token_reset = current_user_id.set(claims.sub)
            try:
                await self.app(scope, receive, send)
            finally:
                current_user_id.reset(token_reset)
            return

        # Mode 3: Legacy OAuth callback
        if self.legacy_token_validator is not None and self.legacy_token_validator(token):
            await self.app(scope, receive, send)
            return

        # None of the modes accepted the token
        await _unauthorized("Invalid or expired token")(scope, receive, send)
