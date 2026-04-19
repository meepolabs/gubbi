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
from uuid import UUID

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
        founder_user_id: UUID | None = None,
    ) -> None:
        self.app = app
        self.api_key = api_key
        self.introspector = introspector
        self.required_scope = required_scope
        self.legacy_token_validator = legacy_token_validator
        # Legacy API-key and legacy-OAuth paths pre-date multi-tenant auth; both
        # authenticate as "the founder". Binding their requests to this UUID lets
        # user_scoped_connection set app.current_user_id uniformly across all auth
        # modes. When None, legacy-authenticated requests reach DB code without a
        # user binding and MissingUserIdError surfaces as a 500.
        self.founder_user_id = founder_user_id

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
            await self._call_with_founder(scope, receive, send)
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

            # Defense-in-depth: TokenClaims.sub is typed UUID and parsed via UUID(sub_raw)
            # in HydraIntrospector, but mirror db_context's isinstance guard so a bypass
            # path (test mock, future cache deserializer) cannot stash a str into the ctxvar.
            if not isinstance(claims.sub, UUID):
                await _unauthorized("Invalid or expired token")(scope, receive, send)
                return

            token_reset = current_user_id.set(claims.sub)
            try:
                await self.app(scope, receive, send)
            finally:
                current_user_id.reset(token_reset)
            return

        # Mode 3: Legacy OAuth callback
        if self.legacy_token_validator is not None and self.legacy_token_validator(token):
            await self._call_with_founder(scope, receive, send)
            return

        # None of the modes accepted the token
        await _unauthorized("Invalid or expired token")(scope, receive, send)

    async def _call_with_founder(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Invoke the wrapped app with current_user_id bound to the founder UUID.

        Used by legacy auth modes (API key, legacy OAuth). When no founder UUID
        is configured the request passes through without binding; downstream
        DB code will then raise MissingUserIdError (→ 500). This is intentional
        fail-loud behaviour — the 500 flags a deployment misconfiguration, it
        is NOT a silent security bypass.
        """
        if self.founder_user_id is None:
            await self.app(scope, receive, send)
            return
        token_reset = current_user_id.set(self.founder_user_id)
        try:
            await self.app(scope, receive, send)
        finally:
            current_user_id.reset(token_reset)
