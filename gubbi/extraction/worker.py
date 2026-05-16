"""Arq worker for conversation extraction jobs.

Sets up PostgreSQL pool, content cipher, extraction service, and Redis
for the extraction job to use.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from collections.abc import Callable
from contextlib import suppress

import redis.asyncio as aioredis
import structlog
from arq.connections import RedisSettings
from gubbi_common.bootstrap.pg_log_probe import probe_pg_log_settings
from gubbi_common.budget import PRE_CHARGE_LUA, BudgetHelper

from gubbi.config import Settings, get_settings
from gubbi.constants import ARQ_JOB_TIMEOUT_SECS
from gubbi.crypto.cipher import ContentCipher, load_master_keys_from_env
from gubbi.extraction.context import ExtractionContext
from gubbi.extraction.health import app as health_app
from gubbi.extraction.jobs.extract_conversation import extract_conversation
from gubbi.extraction.llm.anthropic_provider import AnthropicProvider
from gubbi.extraction.llm.fake_provider import FakeLLMProvider
from gubbi.extraction.llm.provider import LLMProvider
from gubbi.extraction.service import ExtractionService
from gubbi.storage.pg_setup import init_pool
from gubbi.telemetry.logger import initialize_logger

# ``logger`` is the canonical async-context logger (used inside the
# async ``startup`` / ``shutdown`` Arq hooks). ``_sync_log`` covers the
# one sync helper (``_build_content_cipher``); ``structlog.AsyncBoundLogger``
# emits return coroutines that cannot be used from sync callers.
logger = structlog.get_logger(__name__)
_sync_log = logging.getLogger(__name__)


# Registry mapping ``JOURNAL_LLM_PROVIDER`` env values to provider
# factories. Each factory takes the resolved Settings and returns an
# LLMProvider implementation. Open/closed: append a new (name, factory)
# pair to register a new provider; the worker switch logic itself does
# not change.
#
# Default key is ``"anthropic"`` so an unset env var in prod yields the
# real provider. Testbench D-tier compose sets ``JOURNAL_LLM_PROVIDER:
# fake`` to opt into the in-process stub.
_PROVIDER_FACTORIES: dict[str, Callable[[Settings], LLMProvider]] = {
    "anthropic": lambda s: AnthropicProvider(s.llm),
    "fake": lambda _s: FakeLLMProvider(),
}


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
        # Sync emit -- routes through stdlib ``logging`` because
        # ``_build_content_cipher`` is a sync helper called from
        # ``startup`` before the ``await`` chain begins.
        _sync_log.warning(
            "Content cipher disabled -- set JOURNAL_ENCRYPTION_MASTER_KEY_V1 "
            "to enable app-layer encryption"
        )
        return None
    return ContentCipher(master_keys)


async def startup(ctx: ExtractionContext) -> None:
    """Arq worker startup hook: init logging, health server, PG pool, and Redis client."""
    # Load settings.
    settings = get_settings()

    # Configure structured logging FIRST -- emits below would crash with
    # "AttributeError: 'NoneType' object has no attribute 'msg'" otherwise,
    # because Arq workers don't run the FastAPI lifespan that initializes
    # structlog in the HTTP server. Tests pass via conftest's autouse
    # session-scoped fixture, masking the production gap.
    initialize_logger("gubbi-extraction-worker", log_dir=str(settings.log_dir))

    # Health server thread (existing behaviour).
    health_thread = threading.Thread(
        target=_run_health_server,
        daemon=True,
    )
    health_thread.start()
    ctx["health_thread"] = health_thread

    # PostgreSQL pool.
    pool = await init_pool(settings.db.app_url)
    ctx["pool"] = pool
    await logger.info("Extraction worker PG pool ready")

    # Postgres log-settings probe (mirrors gubbi.main lifespan): refuse to
    # start when the cluster would capture statement text or bound
    # parameters in its log -- the worker hits the same encrypted INSERT
    # path as the HTTP API via ``extract_conversation``. Mode is read
    # from JOURNAL_PG_LOG_PROBE_MODE (strict|warn|off; default strict).
    pg_log_probe_mode = os.environ.get("JOURNAL_PG_LOG_PROBE_MODE", "strict")
    try:
        await probe_pg_log_settings(pool, mode=pg_log_probe_mode)
    except BaseException:
        # Catch BaseException (not Exception) so CancelledError /
        # KeyboardInterrupt during the probe still close the pool
        # before unwinding. Best-effort teardown; original error or
        # cancellation must propagate.
        # Suppress asyncio.CancelledError from close() explicitly --
        # CancelledError is a BaseException (not Exception) since
        # Python 3.8, so a bare ``suppress(Exception)`` would let a
        # cancelled close() clobber the original cancellation we're
        # about to ``raise``.
        with suppress(Exception, asyncio.CancelledError):
            await pool.close()
        raise

    # Content cipher.
    cipher = _build_content_cipher()
    ctx["cipher"] = cipher

    # Extraction service.
    # The LLM provider is selected by the ``JOURNAL_LLM_PROVIDER`` env
    # var (default ``anthropic`` so a missing env in prod stays safe).
    # Testbench D-tier compose sets ``JOURNAL_LLM_PROVIDER=fake`` so the
    # worker boots against the in-process FakeLLMProvider stub instead
    # of dialling api.anthropic.com -- keeps D-tier hermetic and avoids
    # accidental spend if a leaked ANTHROPIC_API_KEY is in scope.
    # PRD: llm_context/tasks/milestone-04.5-verification-suite.md TASK-04.5.07.
    # Adding a new provider is a one-line append to ``_PROVIDER_FACTORIES``.
    provider_name = os.environ.get("JOURNAL_LLM_PROVIDER", "anthropic").lower()
    factory = _PROVIDER_FACTORIES.get(provider_name)
    if factory is None:
        raise ValueError(
            f"Unknown JOURNAL_LLM_PROVIDER={provider_name!r}; expected "
            f"one of {sorted(_PROVIDER_FACTORIES)}"
        )
    llm_provider: LLMProvider = factory(settings)
    extraction_service = ExtractionService(llm_provider)
    ctx["extraction_service"] = extraction_service

    # Redis pub/sub client.
    redis_url = _redis_url()
    redis_pool = aioredis.ConnectionPool.from_url(redis_url)
    redis_client = aioredis.Redis(connection_pool=redis_pool)
    ctx["redis"] = redis_client
    ctx["redis_pool"] = redis_pool

    # BudgetHelper -- worker invokes only record_actual_cost, but per D7 we
    # register the Lua script anyway (cheapest option; no API split).
    if settings.llm.journal_llm_budget_enabled:
        pre_charge_script = redis_client.register_script(PRE_CHARGE_LUA)
        ctx["budget_helper"] = BudgetHelper(
            redis=redis_client,  # type: ignore[arg-type]  # duck-typed Protocol vs aioredis.Redis
            pre_charge_script=pre_charge_script,
        )
        await logger.info("Extraction worker BudgetHelper ready")
    else:
        await logger.info("Extraction worker BudgetHelper disabled (budget_enabled=False)")

    await logger.info("Extraction worker Redis client ready")


async def shutdown(ctx: ExtractionContext) -> None:
    """Arq worker shutdown hook: close PG pool and Redis client/connection pool."""
    pool = ctx.get("pool")
    if pool is not None:
        await pool.close()
        await logger.info("Extraction worker PG pool closed")

    redis_client = ctx.get("redis")
    if redis_client is not None:
        await redis_client.aclose()
        await logger.info("Extraction worker Redis client closed")

    # redis_client.aclose() does NOT drain an externally-supplied
    # ConnectionPool; close the pool explicitly to avoid leaking
    # pooled connections across worker restarts.
    redis_pool = ctx.get("redis_pool")
    if redis_pool is not None:
        await redis_pool.aclose()
        await logger.info("Extraction worker Redis pool closed")


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
    job_timeout = ARQ_JOB_TIMEOUT_SECS
    keep_result = 86400
    poll_delay = 0.5
