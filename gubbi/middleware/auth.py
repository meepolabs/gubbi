"""Bearer token authentication for the MCP endpoint.

Uses a strategy-based auth chain built at construction time.
Uses a lightweight ASGI wrapper (NOT BaseHTTPMiddleware) to avoid
buffering responses -- BaseHTTPMiddleware breaks SSE streaming
required by MCP's streamable HTTP transport.
"""

from __future__ import annotations

from gubbi_common.auth.bearer_challenge import build_bearer_challenge as _build_bearer_challenge
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from gubbi.auth.scope import SCOPE_GRANTS, check_scope
from gubbi.auth.strategies import AuthRejected, AuthResult, AuthStrategy
from gubbi.auth_context import current_token_scopes, current_user_id

__all__: list[str] = ["BearerAuthMiddleware"]


def _unauthorized(detail: str, resource_metadata_url: str | None = None) -> JSONResponse:
    """Return a 401 JSONResponse with RFC 6750 Bearer challenge."""
    return JSONResponse(
        {"error": detail},
        status_code=401,
        headers={
            "WWW-Authenticate": _build_bearer_challenge("invalid_token", resource_metadata_url),
        },
    )


def _forbidden(required_scope: str, resource_metadata_url: str | None = None) -> JSONResponse:
    """Return a 403 JSONResponse with Bearer challenge for scope denial."""
    return JSONResponse(
        {"error": "insufficient_scope"},
        status_code=403,
        headers={
            "WWW-Authenticate": _build_bearer_challenge(
                "insufficient_scope",
                resource_metadata_url,
                required_scope=required_scope,
            ),
        },
    )


def _service_unavailable(detail: str = "auth service unavailable") -> JSONResponse:
    return JSONResponse(
        {"error": detail},
        status_code=503,
        headers={"Retry-After": "5"},
    )


def _has_scope(token_scopes: frozenset[str], required_scope: str) -> bool:
    """Return True if token_scopes satisfy required_scope.

    Extends check_scope to handle tokens whose scopes are the expanded grants
    of the required scope (e.g. API-key tokens carry {"journal:read","journal:write"}
    rather than {"journal"}; both grant the "journal" requirement).
    """
    if check_scope(set(token_scopes), required_scope):
        return True
    expanded = SCOPE_GRANTS.get(required_scope)
    return expanded is not None and expanded.issubset(token_scopes)


class BearerAuthMiddleware:
    """ASGI middleware that enforces Bearer token authentication via strategy list."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        strategies: list[AuthStrategy],
        required_scope: str | None = None,
        protected_resource_metadata_url: str | None = None,
    ) -> None:
        self.app = app
        self.strategies = strategies
        self.required_scope = required_scope
        self.protected_resource_metadata_url = protected_resource_metadata_url

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope)
        result: AuthResult | None = None

        for strategy in self.strategies:
            try:
                result = await strategy.authenticate(request)
            except AuthRejected as exc:
                if exc.status == 503:
                    resp = _service_unavailable(exc.detail)
                elif exc.status == 403:
                    resp = _forbidden(
                        self.required_scope or "",
                        self.protected_resource_metadata_url,
                    )
                else:
                    resp = _unauthorized(exc.detail, self.protected_resource_metadata_url)
                await resp(scope, receive, send)
                return
            if result is not None:
                break

        if result is None:
            await _unauthorized("Invalid or expired token", self.protected_resource_metadata_url)(
                scope, receive, send
            )
            return

        if self.required_scope is not None and not _has_scope(result.scopes, self.required_scope):
            await _forbidden(self.required_scope, self.protected_resource_metadata_url)(
                scope, receive, send
            )
            return

        token_reset = current_user_id.set(result.user_id)
        scope_reset = current_token_scopes.set(result.scopes)
        try:
            await self.app(scope, receive, send)
        finally:
            current_user_id.reset(token_reset)
            current_token_scopes.reset(scope_reset)
