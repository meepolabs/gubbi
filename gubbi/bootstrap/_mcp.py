"""MCP middleware assembly.

Builds the ASGI middleware chain that protects the MCP endpoint:

    OriginValidationMiddleware(
        BearerAuthMiddleware(mcp_http, ...)
    )
"""

from __future__ import annotations

from starlette.types import ASGIApp

from gubbi.auth.strategies import AuthStrategy
from gubbi.config import REQUIRED_OAUTH_SCOPE
from gubbi.middleware import (
    BearerAuthMiddleware,
    OriginValidationMiddleware,
)


def build_mcp_middleware(
    mcp_http: ASGIApp,
    *,
    strategies: list[AuthStrategy],
    required_scope: str | None = None,
    protected_resource_metadata_url: str | None,
    allowed_origins: frozenset[str],
) -> ASGIApp:
    """Build the auth-protected MCP ASGI chain.

    Parameters
    ----------
    mcp_http:
        The raw ``FastMCP.streamable_http_app()`` handler.
    strategies:
        Pre-computed list of ``AuthStrategy`` instances (trust-gateway or
        composited api-key / Hydra / self-host).
    required_scope:
        Scope gate passed through to BearerAuthMiddleware; defaults to
        ``REQUIRED_OAUTH_SCOPE`` ("journal").
    protected_resource_metadata_url:
        RFC 9728 metadata doc URL for OAuth discoverability (may be None).
    allowed_origins:
        Allowed host values from ``ALLOWED_ORIGINS`` config.

    Returns
    -------
    The fully-wrapped ``ASGIApp`` ready to be mounted on the FastAPI app.

    Middleware shape (outermost first):

    1. ``OriginValidationMiddleware`` -- DNS-rebinding guard
    2. ``BearerAuthMiddleware`` -- strategy-based JWT / HMAC auth
    3. ``mcp_http`` -- raw MCP streamable HTTP handler
    """
    bearer = BearerAuthMiddleware(
        mcp_http,
        strategies=strategies,
        required_scope=required_scope or REQUIRED_OAUTH_SCOPE,
        protected_resource_metadata_url=protected_resource_metadata_url,
    )
    return OriginValidationMiddleware(bearer, allowed_origins)
