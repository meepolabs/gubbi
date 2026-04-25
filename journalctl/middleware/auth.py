"""Bearer token authentication for the MCP endpoint.

Validates three types of tokens, reflecting the three supported deploy
shapes (see docs/deployment.md for the full matrix):

1. Static API key (Claude Code, Desktop, Cursor, other CLI clients).
   Enabled in Mode 1 (API-key-only) and Mode 2 (full self-host). Disabled
   in Mode 3 (multi-tenant hosted) -- lifespan passes api_key="" so the
   timing-safe compare never matches.
2. Hydra OAuth 2.1 access tokens (Mode 3 -- multi-tenant hosted,
   activated when JOURNAL_HYDRA_ADMIN_URL is set).
3. Self-host OAuth access tokens via external token_validator callback
   (Mode 2 -- single-user self-host via the MCP SDK's DCR-capable OAuth
   routes, activated when JOURNAL_PASSWORD_HASH is set).

Uses a lightweight ASGI wrapper (NOT BaseHTTPMiddleware) to avoid
buffering responses -- BaseHTTPMiddleware breaks SSE streaming
required by MCP's streamable HTTP transport.
"""

from __future__ import annotations

import secrets
from collections.abc import Callable
from uuid import UUID

import asyncpg
import structlog
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from journalctl.auth.hydra import (
    HydraIntrospector,
    HydraInvalidToken,
    HydraUnreachable,
    TokenClaims,
)
from journalctl.core.auth_context import current_user_id
from journalctl.oauth.constants import MAX_BEARER_TOKEN_LEN


def _build_bearer_challenge(
    error: str,
    resource_metadata_url: str | None,
    *,
    required_scope: str | None = None,
) -> str:
    """Build an RFC 6750 WWW-Authenticate Bearer challenge.

    Per MCP spec 2025-11-25, a protected MCP resource must include a
    ``resource_metadata`` parameter pointing to its OAuth protected-resource
    metadata document so clients can discover the authorization server.
    """
    parts = [f'error="{error}"']
    if required_scope is not None:
        parts.append(f'required_scope="{required_scope}"')
    if resource_metadata_url is not None:
        parts.append(f'resource_metadata="{resource_metadata_url}"')
    return "Bearer " + ", ".join(parts)


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
    1. Direct match against the static API key
    2. Hydra introspection for Ory access tokens (multi-tenant hosted)
    3. Delegated token validation via selfhost_token_validator callback
       (single-user self-host)

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
        jit_pool: asyncpg.Pool | None = None,
        hydra_public_url: str | None = None,
        protected_resource_metadata_url: str | None = None,
    ) -> None:
        self.app = app
        self.api_key = api_key
        self.introspector = introspector
        self.required_scope = required_scope
        self.selfhost_token_validator = selfhost_token_validator
        # Static API key + self-host OAuth paths both authenticate as a single
        # operator. Binding their requests to this UUID lets
        # user_scoped_connection set app.current_user_id uniformly across all
        # auth modes. When None, operator-bound requests reach DB code without
        # a user binding and MissingUserIdError surfaces as a 500.
        self.operator_user_id = operator_user_id
        # admin pool (BYPASSRLS) used for JIT user provisioning in Mode 3.
        # When None, JIT is disabled and the path degrades gracefully.
        self.jit_pool = jit_pool
        # Hydra public URL base for /userinfo endpoint. Fetched only on the
        # JIT path to source an email for the users row. Null disables
        # email fetching; provisioning skips without blocking the request.
        self.hydra_public_url = hydra_public_url
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

        # Mode 2: Hydra introspection (Ory access tokens)
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
            if self.required_scope not in scopes:
                await _forbidden(self.required_scope, self.protected_resource_metadata_url)(
                    scope, receive, send
                )
                return

            # Defense-in-depth: TokenClaims.sub is typed UUID and parsed via UUID(sub_raw)
            # in HydraIntrospector, but mirror db_context's isinstance guard so a bypass
            # path (test mock, future cache deserializer) cannot stash a str into the ctxvar.
            if not isinstance(claims.sub, UUID):
                await _unauthorized(
                    "Invalid or expired token", self.protected_resource_metadata_url
                )(scope, receive, send)
                return

            # JIT provision: idempotent UPSERT for users rows missing due to
            # failed Kratos webhook. Never blocks the request; logs warnings.
            await self._jit_provision(claims, token)

            token_reset = current_user_id.set(claims.sub)
            try:
                await self.app(scope, receive, send)
            finally:
                current_user_id.reset(token_reset)
            return

        # Mode 3: Self-host OAuth callback
        if self.selfhost_token_validator is not None and self.selfhost_token_validator(token):
            await self._call_with_operator(scope, receive, send)
            return

        # None of the modes accepted the token
        await _unauthorized("Invalid or expired token", self.protected_resource_metadata_url)(
            scope, receive, send
        )

    async def _jit_provision(self, claims: TokenClaims, token: str) -> None:
        """Idempotent UPSERT for users rows missing due to failed Kratos webhook.

        Calls Hydra /userinfo to fetch the email mapped to this token's identity,
        then inserts or ignores (ON CONFLICT DO NOTHING) a users row keyed on
        claims.sub.  If jit_pool is None, hydra_public_url is None, or any step
        errors, the method logs a warning and returns without blocking the request.

        Kratos webhook remains the fast-path for normal logins; this JIT path
        handles orphaned identities that slipped through.
        """
        if self.jit_pool is None:
            return

        if self.hydra_public_url is None:
            structlog.get_logger("jit").warning(
                "JIT provisioning disabled -- no hydra_public_url configured"
            )
            return

        try:
            if self.introspector is None or self.introspector.http_client is None:
                structlog.get_logger("jit").warning(
                    "JIT provisioning: no introspector HTTP client available"
                )
                return

            userinfo_resp = await self.introspector.http_client.get(
                f"{self.hydra_public_url}/userinfo",
                headers={"Authorization": f"Bearer {token}"},
            )
            if userinfo_resp.status_code != 200:
                structlog.get_logger("jit").warning(
                    "JIT provisioning: /userinfo returned non-200",
                    status_code=userinfo_resp.status_code,
                )
                return
            userinfo = userinfo_resp.json()
            email = userinfo.get("email")
            if not email:
                structlog.get_logger("jit").warning(
                    "JIT provisioning: no email in /userinfo response"
                )
                return

            async with self.jit_pool.acquire() as conn:
                # UUID binds to $1; asyncpg resolves Python UUID → pg UUID type automatically
                await conn.execute(
                    """
                    INSERT INTO users (id, email)
                    VALUES ($1, $2)
                    ON CONFLICT (id) DO NOTHING
                    """,
                    claims.sub,
                    email,
                )
        except Exception:
            structlog.get_logger("jit").warning(
                "JIT provisioning failed (non-blocking)",
                sub=str(claims.sub),
                exc_info=True,
            )

    async def _call_with_operator(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Invoke the wrapped app with current_user_id bound to the operator UUID.

        Used by operator-identity auth modes (static API key, self-host OAuth).
        When no operator UUID is configured the request returns a 503 with a
        message telling the operator how to provision a user row via the
        scaffold_self_host script. This is not a silent security bypass.
        """
        if self.operator_user_id is None:
            response = JSONResponse(
                {
                    "error": (
                        "operator not provisioned; run " "python deployment/scaffold_self_host.py"
                    ),
                },
                status_code=503,
            )
            await response(scope, receive, send)
            return
        token_reset = current_user_id.set(self.operator_user_id)
        try:
            await self.app(scope, receive, send)
        finally:
            current_user_id.reset(token_reset)
