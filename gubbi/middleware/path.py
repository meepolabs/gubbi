"""Path normalization middleware for the MCP endpoint.

Starlette's Mount("/mcp") returns a 307 redirect for requests to
/mcp (without trailing slash).  HTTP clients following the 307
often change POST->GET, which breaks the MCP streamable HTTP
transport — the initialize POST arrives as a GET and opens an
SSE stream instead.

This raw ASGI middleware rewrites the path before it reaches the
router, avoiding the redirect entirely.  Unlike BaseHTTPMiddleware,
it does NOT buffer responses, so SSE streaming works correctly.
"""

from starlette.types import ASGIApp, Receive, Scope, Send


class MCPPathNormalizer:
    """Normalize /mcp -> /mcp/ at the ASGI level."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope["path"] == "/mcp":
            scope = dict(scope)
            scope["path"] = "/mcp/"
        await self.app(scope, receive, send)
