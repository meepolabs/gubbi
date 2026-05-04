"""Arq worker for conversation extraction jobs.

Sets up PostgreSQL pool, content cipher, extraction service, and Redis
for the extraction job to use.
"""

from __future__ import annotations

import logging
import os
import threading

import redis.asyncio as aioredis
from arq.connections import RedisSettings

from journalctl.config import get_settings
from journalctl.core.crypto import ContentCipher, load_master_keys_from_env
from journalctl.extraction.context import ExtractionContext
from journalctl.extraction.health import app as health_app
from journalctl.extraction.jobs.extract_conversation import extract_conversation
from journalctl.extraction.llm.anthropic_provider import AnthropicProvider
from journalctl.extraction.service import ExtractionService
from journalctl.storage.pg_setup import init_pool

logger = logging.getLogger(__name__)


def _redis_url() -> str:
    return os.environ.get("JOURNAL_REDIS_URL", "redis://localhost:6379")


def _build_redis_settings() -> RedisSettings:
    return RedisSettings.from_dsn(_redis_url())


def _build_content_cipher() -> ContentCipher | None:
    """Build ContentCipher from JOURNAL_ENCRYPTION_MASTER_KEY_V* env vars.

    Returns None if no key is configured. Raises on malformed key material.
    """
    master_keys = load_master_keys_from_env()
    if not master_keys:
        logger.warning(
            "Content cipher disabled -- set JOURNAL_ENCRYPTION_MASTER_KEY_V1 "
            "to enable app-layer encryption"
        )
        return None
    return ContentCipher(master_keys)


async def startup(ctx: ExtractionContext) -> None:
    # Health server thread (existing behaviour).
    health_thread = threading.Thread(
        target=_run_health_server,
        daemon=True,
    )
    health_thread.start()
    ctx["health_thread"] = health_thread

    # Load settings.
    settings = get_settings()

    # PostgreSQL pool.
    pool = await init_pool(settings.db.app_url)
    ctx["pool"] = pool
    logger.info("Extraction worker PG pool ready")

    # Content cipher.
    cipher = _build_content_cipher()
    ctx["cipher"] = cipher

    # Extraction service.
    extraction_service = ExtractionService(AnthropicProvider(settings.llm))
    ctx["extraction_service"] = extraction_service

    # Redis pub/sub client.
    redis_url = _redis_url()
    redis_client = await aioredis.from_url(redis_url)
    ctx["redis"] = redis_client
    logger.info("Extraction worker Redis client ready")


async def shutdown(ctx: ExtractionContext) -> None:
    pool = ctx.get("pool")
    if pool is not None:
        await pool.close()
        logger.info("Extraction worker PG pool closed")

    redis_client = ctx.get("redis")
    if redis_client is not None:
        await redis_client.aclose()
        logger.info("Extraction worker Redis client closed")


def _run_health_server() -> None:
    import uvicorn  # noqa: PLC0415

    public = os.environ.get("JOURNAL_EXTRACTION_HEALTH_BIND_PUBLIC", "").lower() == "true"
    host = "0.0.0.0" if public else "127.0.0.1"  # noqa: S104
    uvicorn.run(health_app, host=host, port=8201, log_level="info")


class WorkerSettings:
    redis_settings = _build_redis_settings()
    functions = [extract_conversation]
    on_startup = startup
    on_shutdown = shutdown
    max_jobs = 10
    job_timeout = 600
    keep_result = 86400
    poll_delay = 0.5
