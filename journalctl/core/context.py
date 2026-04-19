"""Application-level context passed to all MCP tool register() functions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import UUID

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

    ``pool`` is the runtime pool (``journal_app`` role — NOT BYPASSRLS).
    Every tenant-scoped tool call wraps its DB work with
    ``core.db_context.user_scoped_connection(pool)``.

    ``admin_pool`` is the optional cross-tenant pool (``journal_admin`` role,
    BYPASSRLS). Reserved for admin/worker paths like ``journal_reindex`` that
    must read across all tenants. ``None`` = single-tenant dev fallback.

    ``founder_user_id`` is the UUID bound into the ``current_user_id``
    ContextVar when a legacy API-key request is authorised. Resolved during
    lifespan from ``JOURNAL_FOUNDER_USER_ID`` or a DB lookup by
    ``JOURNAL_FOUNDER_EMAIL``. ``None`` disables the legacy-key tenant path.
    """

    pool: asyncpg.Pool
    embedding_service: EmbeddingService
    settings: Settings
    logger: structlog.stdlib.AsyncBoundLogger
    admin_pool: asyncpg.Pool | None = None
    founder_user_id: UUID | None = None
