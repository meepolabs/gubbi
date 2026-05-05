"""Shared authentication strategy registry and concrete implementations.

Defines ::

  * ``AuthStrategy`` -- a Protocol for auth strategies that attempt to
    authenticate an ASGI/FastAPI request and return :class:`AuthResult` or `None`.
  * ``AuthResult(user_id, scopes)`` -- a frozen dataclass representing a
    successful authentication.
  * ``AuthRejected`` -- raised by strategies when credentials are present
    but invalid (maps to 401/503 in the middleware adapter).

Four concrete strategies are provided:

  * ``TrustGatewayStrategy`` -- HMAC-verified X-Auth-* envelope (deploy-time switch only).
  * ``ApiKeyStrategy`` -- static API-key bearer token comparison.
  * ``HydraStrategy`` -- Ory Hydra introspection for ``ory_at_*`` tokens.
  * ``SelfHostStrategy`` -- self-host OAuth callback token validation.

Deployment posture (D3 from Task CO.17a -- locked by Lead):

  When ``settings.auth.trust_gateway is True``, the lifespan builds a list
  containing ONLY ``TrustGatewayStrategy``. Direct Hydra tokens, API keys
  etc MUST NOT pass because no other strategy sees them (the loop reaches
  "all returned None" and produces 401).

  When ``trust_gateway is False``, the list contains ``ApiKeyStrategy``,
  ``HydraStrategy``, and ``SelfHostStrategy`` (only those whose backing
  data is non-None; empty ones are excluded).
"""

from __future__ import annotations

import logging
import secrets
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from gubbi_common.auth.gateway_signature import (
    GATEWAY_CONTRACT_VERSION,
    SignatureError,
    verify_signature,
)
from starlette.requests import Request

from gubbi.auth.hydra import HydraIntrospector, HydraInvalidToken, HydraUnreachable
from gubbi.oauth.constants import MAX_BEARER_TOKEN_LEN

_logger = logging.getLogger("gubbi.auth.strategies")

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class AuthRejected(Exception):
    """Raised by a strategy when credentials are present but invalid.

    The middleware/FastAPI adapters translate these into appropriate JSON
    error responses with the given ``status`` code and ``detail`` string.
    """

    def __init__(self, detail: str, status: int = 401) -> None:
        super().__init__(detail)
        self.detail = detail
        self.status = status


@dataclass(frozen=True)
class AuthResult:
    """Frozen result of a successful authentication attempt."""

    user_id: UUID
    scopes: frozenset[str]


class AuthStrategy(Protocol):
    """Protocol for protocol-based authentication strategies.

    Each strategy attempts to authenticate the request. Returns
    ``AuthResult`` on success, ``None`` to defer to the next strategy,
    or raises ``AuthRejected`` when credentials are present but invalid
    (the adapter will immediately translate this into an error response).
    """

    name: str  # for logging / spans, e.g. "trust_gateway", "api_key", "hydra", "selfhost"

    async def authenticate(self, request: Request) -> AuthResult | None: ...


# ---------------------------------------------------------------------------
# Module-level helpers (not strategies themselves)
# ---------------------------------------------------------------------------

_DEFAULT_SCOPES: frozenset[str] = frozenset({"journal:read", "journal:write"})


def _resolve_scopes(scopes_header: str) -> frozenset[str]:
    """Parse X-Auth-Scopes header into a frozenset, falling back to defaults."""
    parsed = frozenset(s for s in scopes_header.split() if s)
    if not parsed:
        _logger.debug("Empty X-Auth-Scopes -- falling back to default scopes")
        return _DEFAULT_SCOPES
    return parsed


def _extract_bearer_token(request: Request) -> str:
    """Extract and validate the Bearer token from the Authorization header.

    Returns the raw token string on success. Raises ``AuthRejected(401)`` when
    no authorization header is present or when it is not a bearer scheme, and
    also raises on tokens exceeding ``MAX_BEARER_TOKEN_LEN``.
    """
    auth_header: str = request.headers.get("authorization", "")
    if not auth_header:
        raise AuthRejected(detail="Missing or invalid Authorization header", status=401)
    lower = auth_header.lower()
    if not lower.startswith("bearer "):
        raise AuthRejected(detail="Missing or invalid Authorization header", status=401)
    token = auth_header[7:]  # strip 'Bearer '
    if len(token) > MAX_BEARER_TOKEN_LEN:
        raise AuthRejected(detail="Invalid token", status=401)
    return token


# ---------------------------------------------------------------------------
# Concrete strategies
# ---------------------------------------------------------------------------


class TrustGatewayStrategy:
    """X-Auth-* HMAC envelope verification.

    Reads X-Auth-User-Id (and optionally signature fields) from request headers.
    Returns None when the user-id header is absent; raises AuthRejected on
    present-but-invalid values.
    """

    def __init__(
        self,
        gateway_secret: bytes | None,
        gateway_require_signature: bool = False,
    ) -> None:
        self.name = "trust_gateway"
        self.gateway_secret = gateway_secret
        self.gateway_require_signature = gateway_require_signature

    async def authenticate(self, request: Request) -> AuthResult | None:
        user_id_header = request.headers.get("x-auth-user-id", "")
        if not user_id_header:
            raise AuthRejected(detail="Missing X-Auth-User-Id header", status=401)
        try:
            user_uuid = UUID(user_id_header)
        except (ValueError, AttributeError):
            raise AuthRejected(detail="Invalid X-Auth-User-Id header", status=401) from None

        contract_version = request.headers.get("x-auth-contract-version", "")
        scopes_header = request.headers.get("x-auth-scopes", "")
        timestamp_header = request.headers.get("x-auth-timestamp", "")
        signature_header = request.headers.get("x-auth-signature", "")
        token_fp = request.headers.get("x-auth-token-fp", "")

        sig_present = bool(signature_header)
        sig_required = self.gateway_require_signature

        if sig_present or sig_required:
            if contract_version != str(GATEWAY_CONTRACT_VERSION):
                raise AuthRejected(detail="Unsupported X-Auth-Contract-Version", status=401)
            if self.gateway_secret is None:
                _logger.warning("gateway_require_signature=true but secret not configured")
                raise AuthRejected(detail="gateway secret not configured", status=503)
            try:
                verify_signature(
                    self.gateway_secret,
                    signature_header,
                    str(user_uuid),
                    scopes_header,
                    timestamp_header,
                    request.method.upper(),
                    request.url.path,
                )
            except SignatureError as exc:
                _logger.warning(
                    "Gateway signature verification failed",
                    extra={
                        "error_type": type(exc).__name__,
                        "user_id": user_id_header,
                        "token_fp": token_fp,
                    },
                )
                raise AuthRejected(detail="Invalid gateway signature", status=401) from None

        resolved_scopes = _resolve_scopes(scopes_header)
        return AuthResult(user_id=user_uuid, scopes=resolved_scopes)


class ApiKeyStrategy:
    """Static API-key bearer token (timing-safe comparison)."""

    def __init__(
        self,
        api_key: str,
        api_key_scopes: tuple[str, ...],
        operator_user_id: UUID | None,
    ) -> None:
        self.name = "api_key"
        self.api_key = api_key
        self.api_key_scopes: frozenset[str] = frozenset(api_key_scopes)
        self.operator_user_id = operator_user_id

    async def authenticate(self, request: Request) -> AuthResult | None:
        token = _extract_bearer_token(request)
        if not self.api_key or not secrets.compare_digest(token, self.api_key):
            return None
        if self.operator_user_id is None:
            raise AuthRejected(detail="Auto-scaffold auth -- operator not provisioned", status=503)
        return AuthResult(user_id=self.operator_user_id, scopes=self.api_key_scopes)


class HydraStrategy:
    """Ory Hydra introspection for ory_at_* tokens."""

    def __init__(self, introspector: HydraIntrospector | None) -> None:
        self.name = "hydra"
        self.introspector = introspector

    async def authenticate(self, request: Request) -> AuthResult | None:
        if self.introspector is None:
            return None
        token = _extract_bearer_token(request)
        if not token.startswith("ory_at_"):
            return None
        try:
            claims = await self.introspector.introspect(token)
        except HydraUnreachable:
            raise AuthRejected(detail="Auth service unavailable", status=503) from None
        except HydraInvalidToken:
            raise AuthRejected(detail="Invalid or expired token", status=401) from None
        if not isinstance(claims.sub, UUID):
            raise AuthRejected(detail="Invalid or expired token", status=401) from None
        return AuthResult(user_id=claims.sub, scopes=frozenset(claims.scope.split()))


class SelfHostStrategy:
    """Self-host OAuth callback token validation."""

    def __init__(
        self,
        token_validator: Callable[[str], frozenset[str] | None] | None,
        operator_user_id: UUID | None,
    ) -> None:
        self.name = "selfhost"
        self.token_validator = token_validator
        self.operator_user_id = operator_user_id

    async def authenticate(self, request: Request) -> AuthResult | None:
        if self.token_validator is None:
            return None
        token = _extract_bearer_token(request)
        granted = self.token_validator(token)
        if granted is None:
            return None
        if self.operator_user_id is None:
            raise AuthRejected(detail="Operator not provisioned", status=503)
        return AuthResult(user_id=self.operator_user_id, scopes=granted)
