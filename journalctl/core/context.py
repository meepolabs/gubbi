"""Application-level context passed to all MCP tool register() functions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    import asyncpg

    from journalctl.config import Settings
    from journalctl.storage.embedding_service import EmbeddingService


@dataclass
class AppContext:
    """Holds app-scoped resources shared across all MCP tools.

    Created once during lifespan startup and captured in tool closures.
    Mirrors the Context pattern from fastapi_template: the pool is the
    app-level resource; individual tools acquire a connection from it
    per operation.
    """

    pool: asyncpg.Pool
    embedding_service: EmbeddingService
    settings: Settings
    logger: structlog.BoundLogger
