"""Bearer token authentication for the MCP endpoint.

Uses a lightweight ASGI wrapper (NOT BaseHTTPMiddleware) to avoid
buffering responses — BaseHTTPMiddleware breaks SSE streaming
required by MCP's streamable HTTP transport.

The wrapper checks the Authorization header before the request
reaches the MCP sub-app. This works across all gunicorn workers
since the API key comes from environment variables.
"""

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from journalctl.config import get_settings


class BearerAuthMiddleware:
    """ASGI middleware that enforces Bearer token authentication.

    Wraps an ASGI app (the MCP server) and rejects requests
    without a valid Authorization: Bearer <key> header.

    This is NOT BaseHTTPMiddleware — it passes through the raw
    ASGI interface without buffering, so SSE streaming works.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope)
        auth_header = request.headers.get("authorization", "")

        if not auth_header.startswith("Bearer "):
            response = JSONResponse(
                {"error": "Missing or invalid Authorization header"},
                status_code=401,
            )
            await response(scope, receive, send)
            return

        token = auth_header[7:]  # Strip "Bearer "
        settings = get_settings()

        if token != settings.api_key:
            response = JSONResponse(
                {"error": "Invalid API key"},
                status_code=401,
            )
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)
