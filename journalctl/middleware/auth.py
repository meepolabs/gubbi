"""Bearer token authentication for the MCP endpoint.

Validates three types of tokens, reflecting the three supported deploy
shapes (see docs/deployment.md for the full matrix):

1. Static API key (Claude Code, Desktop, Cursor, other CLI clients).
   Enabled in Mode 1 (API-key-only) and Mode 2 (full self-host). Disabled
   in Mode 3 (multi-tenant hosted) -- lifespan passes api_key="" so the
   timing-safe compare never matches.
2. Self-host OAuth access tokens via external token_validator callback
   (Mode 2 -- single-user self-host via the MCP SDK's DCR-capable OAuth
   routes, activated when JOURNAL_PASSWORD_HASH is set).
3. Hydra OAuth 2.1 access tokens (Mode 3 -- multi-tenant hosted,
   activated when JOURNAL_HYDRA_ADMIN_URL is set).

Uses a lightweight ASGI wrapper (NOT BaseHTTPMiddleware) to avoid
buffering responses -- BaseHTTPMiddleware breaks SSE streaming
required by MCP's streamable HTTP transport.
"""

from __future__ import annotations

import secrets
from collections.abc import Callable
from uuid import UUID

from gubbi_common.auth.bearer_challenge import build_bearer_challenge as _build_bearer_challenge
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from journalctl.auth.hydra import (
    HydraIntrospector,
    HydraInvalidToken,
    HydraUnreachable,
)
from journalctl.core.auth_context import current_token_scopes, current_user_id
from journalctl.core.scope import check_scope
from journalctl.oauth.constants import MAX_BEARER_TOKEN_LEN

# RFC 6750 / RFC 9728 Bearer challenges are now built via
# gubbi_common.auth.bearer_challenge.build_bearer_challenge; the local
# alias above keeps existing call sites unchanged.


def _unauthorized(detail: str, resource_metadata_url: str | None = None) -> JSONResponse:
    """Return a 401 JSONResponse with RFC 6750 Bearer challenge.

    The ``detail`` param is the human-readable error in the JSON body; the
    WWW-Authenticate challenge always uses error="invalid_token" per RFC 6750.
    """
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


def _service_unavailable() -> JSONResponse:
    return JSONResponse(
        {"error": "auth service unavailable"},
        status_code=503,
        headers={"Retry-After": "5"},
    )


class BearerAuthMiddleware:
    """ASGI middleware that enforces Bearer token authentication.

    Validates tokens in three modes:
    1. Direct match against the static API key (Mode 1).
    2. Delegated token validation via selfhost_token_validator callback
       (Mode 2 -- single-user self-host).
    3. Hydra introspection for Ory access tokens (Mode 3 -- multi-tenant
       hosted).  Also supports a trust-gateway sub-mode (cloud-api
       forwarding) where all auth is skipped and X-Auth-User-Id is trusted.

    This is NOT BaseHTTPMiddleware -- it passes through the raw ASGI
    interface without buffering, so SSE streaming works.
    """

    def __init__(
        self,
        app: ASGIApp,
        api_key: str,
        introspector: HydraIntrospector | None = None,
        required_scope: str = "journal",
        selfhost_token_validator: Callable[[str], bool] | None = None,
        operator_user_id: UUID | None = None,
        protected_resource_metadata_url: str | None = None,
        trust_gateway: bool = False,
    ) -> None:
        self.app = app
        self.api_key = api_key
        self.introspector = introspector
        self.required_scope = required_scope
        self.selfhost_token_validator = selfhost_token_validator
        self.trust_gateway = trust_gateway
        # Static API key + self-host OAuth paths both authenticate as a single
        # operator. Binding their requests to this UUID lets
        # user_scoped_connection set app.current_user_id uniformly across all
        # auth modes. When None, operator-bound requests reach DB code without
        # a user binding and MissingUserIdError surfaces as a 500.
        self.operator_user_id = operator_user_id
        # URL of the OAuth protected-resource metadata document. Surfaced in
        # WWW-Authenticate on 401/403 per MCP spec 2025-11-25 so clients can
        # discover the authorization server. None disables the parameter
        # (appropriate for Mode 1 API-key-only deployments with no OAuth
        # routes to discover).
        self.protected_resource_metadata_url = protected_resource_metadata_url

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Trust gateway mode: skip all auth logic and trust the upstream
        # X-Auth-User-Id header set by cloud-api. This is used by hosted
        # Mode 3 deployments behind cloud-api.
        # DEPLOYMENT INVARIANT: this path trusts upstream cloud-api;
        # journalctl must not be directly internet-reachable when
        # trust_gateway=True.
        if self.trust_gateway:
            request = Request(scope)
            user_id_header = request.headers.get("x-auth-user-id", "")
            if not user_id_header:
                response = JSONResponse(
                    {"error": "Missing X-Auth-User-Id header"},
                    status_code=401,
                )
                await response(scope, receive, send)
                return
            try:
                user_uuid = UUID(user_id_header)
            except (ValueError, AttributeError):
                response = JSONResponse(
                    {"error": "Invalid X-Auth-User-Id header"},
                    status_code=401,
                )
                await response(scope, receive, send)
                return
            token_reset = current_user_id.set(user_uuid)
            scope_reset = current_token_scopes.set(frozenset({"journal"}))
            try:
                await self.app(scope, receive, send)
            finally:
                current_user_id.reset(token_reset)
                current_token_scopes.reset(scope_reset)
            return

        request = Request(scope)
        auth_header = request.headers.get("authorization", "")

        if not auth_header:
            await _unauthorized(
                "Missing or invalid Authorization header", self.protected_resource_metadata_url
            )(scope, receive, send)
            return

        if not auth_header.lower().startswith("bearer "):
            await _unauthorized(
                "Missing or invalid Authorization header", self.protected_resource_metadata_url
            )(scope, receive, send)
            return

        token = auth_header[7:]

        if len(token) > MAX_BEARER_TOKEN_LEN:
            await _unauthorized("Invalid token", self.protected_resource_metadata_url)(
                scope, receive, send
            )
            return

        # Mode 1: Static API key (timing-safe comparison).
        # Empty api_key disables the path entirely (Mode 3 passes ""); the
        # explicit truthiness check prevents an empty-vs-empty match.
        if self.api_key and secrets.compare_digest(token, self.api_key):
            await self._call_with_operator(scope, receive, send)
            return

        # Mode 3: Hydra introspection (Ory access tokens)
        if token.startswith("ory_at_") and self.introspector is not None:
            try:
                claims = await self.introspector.introspect(token)
            except HydraUnreachable:
                await _service_unavailable()(scope, receive, send)
                return
            except HydraInvalidToken:
                await _unauthorized(
                    "Invalid or expired token", self.protected_resource_metadata_url
                )(scope, receive, send)
                return

            scopes = claims.scope.split()
            if not check_scope(set(scopes), self.required_scope):
                await _forbidden(self.required_scope, self.protected_resource_metadata_url)(
                    scope, receive, send
                )
                return

            # Store scopes for @require_scope decorator (per-tool checks).
            token_scope_reset = current_token_scopes.set(frozenset(scopes))

            # Defense-in-depth: claims.sub is typed UUID and parsed via UUID(sub_raw)
            # in HydraIntrospector, but mirror gubbi_common.db.user_scoped
            # isinstance guard so a bypass
            # path (test mock, future cache deserializer) cannot stash a str into the ctxvar.
            if not isinstance(claims.sub, UUID):
                await _unauthorized(
                    "Invalid or expired token", self.protected_resource_metadata_url
                )(scope, receive, send)
                return

            sub = claims.sub
            token_reset = current_user_id.set(sub)
            try:
                await self.app(scope, receive, send)
            finally:
                current_user_id.reset(token_reset)
                current_token_scopes.reset(token_scope_reset)
            return

        # Mode 2: Self-host OAuth callback
        if self.selfhost_token_validator is not None and self.selfhost_token_validator(token):
            await self._call_with_operator(scope, receive, send)
            return

        # None of the modes accepted the token
        await _unauthorized("Invalid or expired token", self.protected_resource_metadata_url)(
            scope, receive, send
        )

    async def _call_with_operator(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Invoke the wrapped app with current_user_id bound to the operator UUID.

        Used by operator-identity auth modes (static API key, self-host OAuth).
        When no operator UUID is configured the request returns a 503. This is
        not a silent security bypass.
        """
        if self.operator_user_id is None:
            response = JSONResponse(
                {
                    "error": (
                        "operator not provisioned; app should auto-scaffold on Mode 1/2 startup"
                    ),
                },
                status_code=503,
            )
            await response(scope, receive, send)
            return
        token_reset = current_user_id.set(self.operator_user_id)
        scope_reset = current_token_scopes.set(frozenset({"journal"}))
        try:
            await self.app(scope, receive, send)
        finally:
            current_user_id.reset(token_reset)
            current_token_scopes.reset(scope_reset)
