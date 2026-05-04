"""ASGI middleware for the journal MCP server.

All middleware here uses raw ASGI (NOT BaseHTTPMiddleware) to avoid
buffering responses — BaseHTTPMiddleware breaks SSE streaming required
by MCP's streamable HTTP transport.
"""

from gubbi.middleware.auth import BearerAuthMiddleware
from gubbi.middleware.correlation import CorrelationIDMiddleware
from gubbi.middleware.origin import OriginValidationMiddleware
from gubbi.middleware.path import MCPPathNormalizer

__all__ = [
    "BearerAuthMiddleware",
    "CorrelationIDMiddleware",
    "MCPPathNormalizer",
    "OriginValidationMiddleware",
]
