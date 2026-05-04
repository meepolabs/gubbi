"""Shared FastAPI authentication dependencies for REST API routes.

Provides ``resolve_user_id`` and ``require_scope`` for use by
``/api/v1/ingest`` and ``/api/v1/extraction/progress`` endpoints.

Supports four auth modes in priority order:

a) Trust-gateway envelope (X-Auth-* headers, H-1 HMAC-verified)
b) Static API key (Authorization: Bearer <key>)
c) Hydra bearer (Authorization: Bearer ory_at_<token>)
d) Self-host OAuth (Authorization: Bearer <selfhost_token>)

The trust-gateway branch mirrors ``BearerAuthMiddleware``'s envelope
verification (see ``gubbi/middleware/auth.py``). Both paths consult
the same ``app.state.gubbi_gateway_secret`` and apply the same
legacy-vs-required-signature semantics.
"""

from __future__ import annotations

import hmac
import logging
from collections.abc import Awaitable, Callable
from uuid import UUID

from fastapi import HTTPException, Request
from gubbi_common.auth.gateway_signature import (
    GATEWAY_CONTRACT_VERSION,
    SignatureError,
    verify_signature,
)

from gubbi.oauth.constants import MAX_BEARER_TOKEN_LEN

logger = logging.getLogger(__name__)

_LEGACY_DEFAULT_SCOPES: frozenset[str] = frozenset({"journal:read", "journal:write"})


async def resolve_user_id(
    request: Request,
    scope: str | None = None,
) -> tuple[UUID, frozenset[str]]:
    """Authenticate request and return (user_id, granted_scopes).

    After resolving (user_id, scopes): if ``scope`` is not None,
    checks that ``scope`` is in ``scopes``. If not, raises HTTP 403.
    """
    app_ctx = request.app.state.app_ctx
    settings = app_ctx.settings

    # (a) Trust-gateway envelope (H-1 HMAC-verified envelope per
    # gubbi_common.auth.gateway_signature). Mirrors BearerAuthMiddleware's
    # verification path so REST and MCP enforce the same contract.
    if settings.auth.trust_gateway:
        user_id_str = request.headers.get("x-auth-user-id", "")
        if not user_id_str:
            raise HTTPException(status_code=401, detail="Missing X-Auth-User-Id header")
        try:
            user_uuid = UUID(user_id_str)
        except (ValueError, AttributeError):
            raise HTTPException(status_code=401, detail="Invalid X-Auth-User-Id header") from None

        contract_version = request.headers.get("x-auth-contract-version", "")
        scopes_header = request.headers.get("x-auth-scopes", "")
        timestamp_header = request.headers.get("x-auth-timestamp", "")
        signature_header = request.headers.get("x-auth-signature", "")

        sig_present = bool(signature_header)
        sig_required = settings.auth.gateway_require_signature

        if sig_present or sig_required:
            # Verification path: signature header present, OR enforced.
            if contract_version != str(GATEWAY_CONTRACT_VERSION):
                raise HTTPException(status_code=401, detail="Unsupported X-Auth-Contract-Version")
            gateway_secret = getattr(request.app.state, "gubbi_gateway_secret", None)
            if gateway_secret is None:
                # Deployment misconfiguration when gateway_require_signature=true.
                logger.warning(
                    "gateway_require_signature=true but secret not configured on app.state",
                    extra={"user_id": user_id_str},
                )
                raise HTTPException(status_code=503, detail="gateway secret not configured")
            try:
                verify_signature(
                    gateway_secret,
                    signature_header,
                    str(user_uuid),
                    scopes_header,
                    timestamp_header,
                    request.method.upper(),
                    request.url.path,
                )
            except SignatureError as exc:
                logger.warning(
                    "Gateway signature verification failed on REST route",
                    extra={"error_type": type(exc).__name__, "user_id": user_id_str},
                )
                raise HTTPException(status_code=401, detail="Invalid gateway signature") from None
        # else: legacy path -- no signature header AND not enforced.
        # Same semantics as BearerAuthMiddleware: accept bare X-Auth-User-Id.

        parsed_scopes = frozenset(s for s in scopes_header.split() if s)
        if not parsed_scopes:
            parsed_scopes = _LEGACY_DEFAULT_SCOPES

        user_id, granted_scopes = user_uuid, parsed_scopes

    else:
        auth_header = request.headers.get("authorization", "")
        if not auth_header.lower().startswith("bearer "):
            raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

        token = auth_header[7:]

        if len(token) > MAX_BEARER_TOKEN_LEN:
            raise HTTPException(status_code=401, detail="Invalid token")

        # (b) Static API key
        if settings.auth.api_key and hmac.compare_digest(token, settings.auth.api_key):
            operator_user_id = request.app.state.operator_user_id
            if operator_user_id is None:
                raise HTTPException(status_code=503, detail="Operator not provisioned")
            user_id = operator_user_id
            granted_scopes = frozenset(settings.auth.api_key_scopes)

        # (c) Hydra bearer
        elif token.startswith("ory_at_"):
            introspector = request.app.state.hydra_introspector
            if introspector is None:
                raise HTTPException(status_code=401, detail="Invalid or expired token")
            from gubbi.auth.hydra import HydraInvalidToken, HydraUnreachable

            try:
                claims = await introspector.introspect(token)
            except HydraUnreachable:
                raise HTTPException(status_code=503, detail="Auth service unavailable") from None
            except HydraInvalidToken:
                raise HTTPException(status_code=401, detail="Invalid or expired token") from None
            if not isinstance(claims.sub, UUID):
                raise HTTPException(status_code=401, detail="Invalid or expired token")
            user_id = claims.sub
            granted_scopes = frozenset(claims.scope.split())

        # (d) Self-host OAuth
        else:
            validator = request.app.state.selfhost_token_validator
            if validator is None:
                raise HTTPException(status_code=401, detail="Invalid or expired token")
            granted = validator(token)
            if granted is None:
                raise HTTPException(status_code=401, detail="Invalid or expired token")
            operator_user_id = request.app.state.operator_user_id
            if operator_user_id is None:
                raise HTTPException(status_code=503, detail="Operator not provisioned")
            user_id = operator_user_id
            granted_scopes = granted

    # Optional scope check
    if scope is not None and scope not in granted_scopes:
        raise HTTPException(status_code=403, detail="insufficient_scope")

    return (user_id, granted_scopes)


def require_scope(
    scope: str | None = None,
) -> Callable[[Request], Awaitable[tuple[UUID, frozenset[str]]]]:
    """FastAPI dependency factory that enforces a required OAuth scope.

    Returns a dependency that resolves (user_id, scopes) via resolve_user_id
    and optionally checks for scope membership.

    Usage::

        @router.get("/endpoint")
        async def endpoint(
            auth: Annotated[tuple[UUID, frozenset[str]], Depends(require_scope("journal:write"))],
        ):
            user_id, scopes = auth
    """

    async def _dep(request: Request) -> tuple[UUID, frozenset[str]]:
        return await resolve_user_id(request, scope=scope)

    return _dep
