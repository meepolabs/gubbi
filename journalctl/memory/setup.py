"""Memory service setup — environment configuration and initialization factory.

Mirrors the oauth/setup.py pattern: one module owns the lifecycle of the
optional mcp-memory-service integration so main.py stays thin.
"""

import logging
import os
from typing import cast

from journalctl.config import Settings
from journalctl.memory.protocol import MemoryServiceProtocol

logger = logging.getLogger(__name__)


def configure_env() -> None:
    """Set mcp-memory-service env vars before its first import.

    mcp-memory-service reads these at import time via its config module,
    so this must be called before any ``import mcp_memory_service`` statement.

    Defaults choose the lightweight path suitable for a single-user GCP instance:

    - ONNX over sentence-transformers: ~80 MB vs ~2.7 GB
    - sqlite_vec: single file, zero external dependencies
    - auto_split disabled: content is already chunked via journal_append
    - quality_boost disabled: baseline embeddings sufficient, avoids background tasks
    """
    os.environ.setdefault("MCP_MEMORY_USE_ONNX", "true")
    os.environ.setdefault("MCP_MEMORY_STORAGE_BACKEND", "sqlite_vec")
    os.environ.setdefault("MCP_ENABLE_AUTO_SPLIT", "false")
    os.environ.setdefault("MCP_QUALITY_BOOST_ENABLED", "false")


async def init_service(settings: Settings) -> MemoryServiceProtocol | None:
    """Initialize and return a MemoryService, or None if disabled / unavailable.

    ``configure_env()`` must be called before this function.

    Args:
        settings: Application settings (checks memory_enabled, memory_db_path).

    Returns:
        Initialized MemoryService, or None on failure / disabled.
    """
    if not settings.memory_enabled:
        return None

    try:
        # Late import — mcp_memory_service is optional (--no-deps install)
        from mcp_memory_service.services.memory_service import (
            MemoryService,  # type: ignore[import-not-found]  # noqa: PLC0415
        )
        from mcp_memory_service.storage.sqlite_vec import (
            SqliteVecMemoryStorage,  # type: ignore[import-not-found]  # noqa: PLC0415
        )

        storage = SqliteVecMemoryStorage(db_path=str(settings.memory_db_path))
        await storage.initialize()
        return cast(MemoryServiceProtocol, MemoryService(storage))
    except ImportError:
        logger.warning("Memory service unavailable: mcp-memory-service not installed")
        return None
    except Exception:
        logger.exception("Memory service initialization failed, continuing without it")
        return None
