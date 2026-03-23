"""ASGI middleware for the journal MCP server.

All middleware here uses raw ASGI (NOT BaseHTTPMiddleware) to avoid
buffering responses — BaseHTTPMiddleware breaks SSE streaming required
by MCP's streamable HTTP transport.
"""

from journalctl.middleware.auth import BearerAuthMiddleware
from journalctl.middleware.path import MCPPathNormalizer

__all__ = ["BearerAuthMiddleware", "MCPPathNormalizer"]
