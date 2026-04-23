"""Application-level context passed to all MCP tool register() functions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import UUID

import structlog

if TYPE_CHECKING:
    import asyncpg

    from journalctl.config import Settings
    from journalctl.core.crypto import ContentCipher
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

    ``operator_user_id`` is the UUID bound into the ``current_user_id``
    ContextVar when an operator-identity auth request is authorised (static
    API key or self-host OAuth). Resolved during lifespan via DB lookup by
    ``JOURNAL_OPERATOR_EMAIL``. ``None`` disables the operator-identity
    tenant binding.

    ``cipher`` is the app-layer AES-256-GCM content cipher (TASK-02.11).
    Built from ``JOURNAL_ENCRYPTION_MASTER_KEY_V*`` env vars at startup.
    ``None`` = no master key configured; required once TASK-02.13 wires
    repository encrypt/decrypt.
    """

    pool: asyncpg.Pool
    embedding_service: EmbeddingService
    settings: Settings
    logger: structlog.stdlib.AsyncBoundLogger
    admin_pool: asyncpg.Pool | None = None
    operator_user_id: UUID | None = None
    cipher: ContentCipher | None = None
