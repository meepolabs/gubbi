"""Bootstrap helpers -- extracted from gubbi.main.lifespan.

Each helper encapsulates one startup concern so the lifespan reads as a
sequence of named calls.  Values are passed through explicit parameters.
"""

from __future__ import annotations

from gubbi.bootstrap._gateway import check_trust_gateway_bind_address
from gubbi.bootstrap._mcp import build_mcp_middleware
from gubbi.bootstrap._oauth import setup_oauth
from gubbi.bootstrap._secret import decode_gateway_secret
from gubbi.bootstrap._teardown import teardown_lifespan_resources
from gubbi.bootstrap.probes import (
    BindAddressProbe,
    ReplicaCountWarnProbe,
)

__all__ = [
    "BindAddressProbe",
    "ReplicaCountWarnProbe",
    "build_mcp_middleware",
    "check_trust_gateway_bind_address",
    "decode_gateway_secret",
    "setup_oauth",
    "teardown_lifespan_resources",
]
