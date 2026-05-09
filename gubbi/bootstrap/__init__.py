"""Bootstrap helpers — extracted from gubbi.main.lifespan.

Each helper encapsulates one startup concern so the lifespan reads as a
sequence of named calls.  Values are passed through explicit parameters.
"""

from __future__ import annotations

from gubbi.bootstrap._mcp import build_mcp_middleware
from gubbi.bootstrap._oauth import setup_oauth
from gubbi.bootstrap._secret import decode_gateway_secret

__all__ = [
    "build_mcp_middleware",
    "decode_gateway_secret",
    "setup_oauth",
]
