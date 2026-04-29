"""Origin header validation for MCP streamable HTTP endpoint.

Prevents DNS-rebinding attacks by checking the ``Origin`` header against
an allowlist. Required by the MCP spec for streamable HTTP transports.

Used as:

.. code-block:: python

    authed_mcp = OriginValidationMiddleware(
        authed_mcp,
        allowed_origins=["https://claude.ai", "https://chatgpt.com", ...],
    )

Requests without an ``Origin`` header (curl, SDK clients) pass through.
Only requests that carry an ``Origin`` header are validated.
"""

from __future__ import annotations

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send


class OriginValidationMiddleware:
    """ASGI middleware that validates the Origin header against an allowlist.

    Non-HTTP scopes (e.g. websocket) and requests without an Origin header
    pass through.  Loopback origins are automatically allowed so local dev
    tools (MCP Inspector, Claude Desktop) work without configuration.
    """

    LOOPBACK_ORIGINS: frozenset[str] = frozenset(
        {
            "http://localhost",
            "http://127.0.0.1",
            "http://[::1]",
        }
    )

    def __init__(
        self,
        app: ASGIApp,
        allowed_origins: frozenset[str] | None = None,
    ) -> None:
        self.app = app
        # Pre-compute lowered allowlist once at __init__ instead of per-request.
        self.allowed_origins: frozenset[str] = frozenset(
            o.lower() for o in (allowed_origins or frozenset())
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Extract Origin header from ASGI scope
        origin: str | None = None
        for key, value in scope.get("headers", []):
            if key.lower() == b"origin":
                origin = value.decode("latin-1")
                break

        if origin is None:
            # No Origin header -- allow (curl, SDK clients don't send it)
            await self.app(scope, receive, send)
            return

        origin_lower = origin.lower()

        # Allow loopback origins for dev. Match exact base or base+":" so
        # "http://localhost.evil.com" cannot bypass via prefix-match
        # (DNS-rebinding mitigation).
        for loopback in self.LOOPBACK_ORIGINS:
            if origin_lower == loopback or origin_lower.startswith(loopback + ":"):
                await self.app(scope, receive, send)
                return

        # Check against pre-lowered allowlist
        if origin_lower not in self.allowed_origins:
            response = JSONResponse(
                {"error": "origin not allowed"},
                status_code=403,
            )
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)
