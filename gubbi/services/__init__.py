"""Shared application services used by more than one delivery surface.

A service module holds a pipeline that both the MCP tool layer
(``gubbi/tools/*``) and the web REST layer (``gubbi/api/v1/web/*``) call, so
the behavior stays identical across surfaces. Services depend on the storage
repositories and the crypto/embedding helpers, never on FastAPI or the MCP
server; each caller adapts the service result to its own response contract.
"""

from __future__ import annotations

__all__: list[str] = []
