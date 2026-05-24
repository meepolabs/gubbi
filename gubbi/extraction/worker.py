"""Arq worker for conversation extraction jobs.

Sets up PostgreSQL pool, content cipher, extraction service, and Redis
for the extraction job to use.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from contextlib import suppress
from typing import TYPE_CHECKING

import redis.asyncio as aioredis
import structlog
from arq.connections import RedisSettings
from gubbi_common.bootstrap import StartupProbe, StartupRunner
from gubbi_common.budget import PRE_CHARGE_LUA, BudgetHelper

from gubbi.bootstrap import PgLogProbe, WorkerReplicaCountWarnProbe
from gubbi.config import Settings, get_settings
from gubbi.constants import ARQ_JOB_TIMEOUT_SECS
from gubbi.crypto.cipher import ContentCipher, load_master_keys_from_env
from gubbi.extraction.health import app as health_app
from gubbi.extraction.jobs.extract_conversation import extract_conversation
from gubbi.extraction.llm.anthropic_provider import AnthropicProvider
from gubbi.extraction.llm.fake_provider import FakeLLMProvider
from gubbi.extraction.service import ExtractionService
from gubbi.storage.pg_setup import init_pool
from gubbi.telemetry import rebind_metrics_after_configure
from gubbi.telemetry.logger import initialize_logger
from gubbi.telemetry.metrics import record_startup_probe_outcome

if TYPE_CHECKING:
    from collections.abc import Callable

    import asyncpg

    from gubbi.extraction.context import ExtractionContext
    from gubbi.extraction.llm.provider import LLMProvider

# ``logger`` is the canonical async-context logger (used inside the
# async ``startup`` / ``shutdown`` Arq hooks). ``_sync_log`` covers the
# one sync helper (``_build_content_cipher``); ``structlog.AsyncBoundLogger``
# emits return coroutines that cannot be used from sync callers.
logger = structlog.get_logger(__name__)
_sync_log = logging.getLogger(__name__)

# One-shot guard for ``_configure_worker_telemetry``. OTel's
# ``set_tracer_provider`` / ``set_meter_provider`` are Once-guarded inside the
# SDK: a second call is rejected with an "Overriding ... Provider is not
# allowed" warning AFTER the new providers (and their BatchSpanProcessor +
# PeriodicExportingMetricReader background threads) have already been built --
# leaking exporter threads on every re-entry. A second ``startup()`` in the
# same process (Arq worker restart-in-place, or a test that drives ``startup``
# twice) would trip exactly that. This module global makes the wiring run at
# most once per process; tests reset it via the ``_reset_worker_telemetry_guard``
# helper (or by monkeypatching this name directly).
_WORKER_TELEMETRY_CONFIGURED: bool = False


def _reset_worker_telemetry_guard() -> None:
    """Reset the one-shot telemetry guard (test-support hook).

    Production never calls this -- the guard is meant to stay latched for
    the life of the process. Tests that exercise ``startup`` more than once
    in a single interpreter use it to re-arm ``_configure_worker_telemetry``.
    """
    global _WORKER_TELEMETRY_CONFIGURED
    _WORKER_TELEMETRY_CONFIGURED = False


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


def _configure_worker_telemetry(settings: Settings) -> None:
    """Wire FULL OTel (traces + metrics) for the Arq worker process.

    The worker does NOT run the FastAPI lifespan, so unlike the HTTP server
    it never reaches ``gubbi.telemetry.configure_otel(app)``. Without this
    call the worker would be BOTH span-dark (no TracerProvider -> spans
    created against the global NoOp tracer and dropped) AND metric-NoOp
    (every counter ``.add`` silently discarded). We intentionally wire the
    worker as a first-class observable service: gubbi-common's
    ``configure_otel`` installs a ``TracerProvider`` with OTLP span export
    AND a ``MeterProvider`` with a periodic OTLP metric reader. (It stays
    FastAPI-decoupled, so no FastAPI / asyncpg / redis auto-instrumentors
    are wired here -- those need the ASGI app and live on the HTTP path.)
    After wiring, ``rebind_metrics_after_configure()`` is invoked so the
    ``gateway.replica_count_warning`` counter -- and the three B5
    orphan-counter factories (orphan_cleanup, extract_conversation,
    anthropic_provider) -- bind to the live MeterProvider and actually
    export.

    Deploy dependency: because this enables OTLP export, the worker's
    deploy must set ``OTEL_EXPORTER_OTLP_ENDPOINT`` (and may set
    ``OTEL_ENABLED=false`` to opt out) -- the SAME dependency cloud-api and
    the gubbi HTTP server already carry. With ``OTEL_ENABLED=false`` the
    SDK providers are installed with no exporter, so spans/metrics are
    created and dropped with zero IO.

    One-shot per process: guarded by ``_WORKER_TELEMETRY_CONFIGURED`` so a
    second ``startup()`` in the same interpreter does not rebuild the
    exporters/readers (and leak their background threads) only to have
    OTel's ``set_*_provider`` Once-guards reject the second registration.

    Best-effort: a configure-time failure must never block worker boot, so
    the exception is logged and swallowed. ``record_replica_count_warning``
    is independently guarded against a missing instrument, so the paired
    WARNING log still fires even if this no-ops.
    """
    global _WORKER_TELEMETRY_CONFIGURED
    if _WORKER_TELEMETRY_CONFIGURED:
        return

    from gubbi_common.telemetry.otel import configure_otel as _common_configure_otel

    from gubbi import __version__ as _gubbi_version

    service_name = os.environ.get("OTEL_SERVICE_NAME", "gubbi-extraction-worker")
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    enabled = os.environ.get("OTEL_ENABLED", "true").strip().lower() in ("true", "1", "yes")
    try:
        _common_configure_otel(
            service_name,
            endpoint,
            enabled=enabled,
            service_version=_gubbi_version,
            deployment_environment=settings.app_env,
        )
        # Re-prime ALL metric instruments against the freshly installed
        # MeterProvider. Goes through ``rebind_metrics_after_configure`` --
        # the same hook the FastAPI lifespan uses -- so the canonical
        # ``initialize_metrics`` lru_cache AND the three B5 orphan-counter
        # factories (orphan_cleanup, extract_conversation, anthropic_provider)
        # all bind to the SDK provider rather than a stale NoOp handle.
        # The cache_clear + re-prime sequence is NOT atomic, but it is safe
        # here only because the health-server thread started later in
        # ``startup`` does not call any metric helper during boot --
        # nothing races this swap. If a startup-time metric emitter is
        # added to that thread (or any other), this re-prime needs a lock.
        rebind_metrics_after_configure()
        # Latch only after a clean wiring: a failed configure leaves the
        # guard unset so a later retry (e.g. a re-armed test) can try again.
        _WORKER_TELEMETRY_CONFIGURED = True
    except Exception:
        # Per the OTel SDK, a bad OTEL_EXPORTER_OTLP_ENDPOINT does NOT raise
        # here: the OTLP exporters + Batch/Periodic readers are constructed
        # eagerly but only fail LATER, on background export, surfacing as
        # background exporter errors (not caught at this call site). So this
        # swallow covers configure-time / import-time failures only. Log at
        # ERROR, not WARNING: a configure failure means total telemetry loss
        # (traces AND metrics) for the worker process. Operators who want
        # to deliberately disable telemetry without seeing this ERROR log
        # should set OTEL_ENABLED=false (the clean opt-out path), rather
        # than relying on configure failure as the kill switch.
        _sync_log.error("Worker OTel wiring failed; traces+metrics lost", exc_info=True)


async def startup(ctx: ExtractionContext) -> None:
    """Arq worker startup hook: init logging, health server, PG pool, and Redis client.

    Uses the same ``StartupRunner``-driven probe sequence as the gubbi
    HTTP lifespan: the worker's two probes (PgLog + WorkerReplicaCount)
    run through the runner so their structured-log + counter emission
    flows through the canonical pipeline. Required-probe failure
    escalates as :class:`gubbi_common.bootstrap.ProbeFailure`; the outer
    ``finally`` closes the pool best-effort so the same teardown path
    covers pre-yield init failure and clean shutdown alike.
    """
    settings = get_settings()

    # Configure structured logging FIRST -- emits below would crash with
    # "AttributeError: 'NoneType' object has no attribute 'msg'" otherwise,
    # because Arq workers don't run the FastAPI lifespan that initializes
    # structlog in the HTTP server. Tests pass via conftest's autouse
    # session-scoped fixture, masking the production gap.
    initialize_logger("gubbi-extraction-worker", log_dir=str(settings.log_dir))
    worker_logger = structlog.get_logger("gubbi-extraction-worker")

    # Wire FULL OTel -- TracerProvider (OTLP span export) + MeterProvider --
    # for the worker process. The worker does not run the FastAPI lifespan,
    # so it never reaches configure_otel(app); without this it would be both
    # span-dark and metric-NoOp (the replica-count counter, and any future
    # worker metric/span, silently dropped). Requires
    # OTEL_EXPORTER_OTLP_ENDPOINT in the deploy. One-shot + best-effort:
    # never blocks boot. See _configure_worker_telemetry.
    _configure_worker_telemetry(settings)

    pool: asyncpg.Pool | None = None

    try:
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

        # Worker probe sequence. PgLogProbe refuses to start when the
        # cluster would capture statement text or bound parameters (the
        # worker hits the same encrypted INSERT path as the HTTP API via
        # ``extract_conversation``). WorkerReplicaCountWarnProbe flags the
        # single-worker deploy-policy violation (M4 #138 worker variant)
        # and increments the alertable ``gateway.replica_count_warning``
        # counter. Mode for PgLogProbe is consumed from
        # ``settings.pg_log_probe_mode`` (was ``JOURNAL_PG_LOG_PROBE_MODE``
        # in the pre-StartupRunner shape); replica count is consumed from
        # ``settings.replica_count`` (was the worker's now-deleted
        # ``_validate_replica_count`` helper, which Settings.replica_count
        # ge=1 supersedes at construction time).
        probes: list[StartupProbe] = [
            PgLogProbe(pool=pool, mode=settings.pg_log_probe_mode),
            WorkerReplicaCountWarnProbe(
                replica_count=settings.replica_count,
                pool_max_per_pod=pool.get_max_size(),
            ),
        ]
        runner = StartupRunner(
            app_env=settings.app_env,
            outcome_counter=record_startup_probe_outcome,
        )
        await runner.run(probes, logger=worker_logger)

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
        if settings.llm.llm_budget_enabled:
            pre_charge_script = redis_client.register_script(PRE_CHARGE_LUA)
            ctx["budget_helper"] = BudgetHelper(
                redis=redis_client,  # type: ignore[arg-type]  # duck-typed Protocol vs aioredis.Redis
                pre_charge_script=pre_charge_script,
            )
            await logger.info("Extraction worker BudgetHelper ready")
        else:
            await logger.info("Extraction worker BudgetHelper disabled (budget_enabled=False)")

        await logger.info("Extraction worker Redis client ready")
    except BaseException:
        # Catch BaseException (not Exception) so CancelledError /
        # KeyboardInterrupt during init still close the pool before
        # unwinding. Best-effort teardown; original error or cancellation
        # must propagate.
        # Suppress asyncio.CancelledError from close() explicitly --
        # CancelledError is a BaseException (not Exception) since
        # Python 3.8, so a bare ``suppress(Exception)`` would let a
        # cancelled close() clobber the original cancellation we're
        # about to ``raise``.
        if pool is not None:
            with suppress(Exception, asyncio.CancelledError):
                await pool.close()
        raise


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
    import uvicorn

    public = os.environ.get("JOURNAL_EXTRACTION_HEALTH_BIND_PUBLIC", "").lower() == "true"
    host = "0.0.0.0" if public else "127.0.0.1"  # noqa: S104
    uvicorn.run(health_app, host=host, port=8201, log_level="info")


class WorkerSettings:
    redis_settings = _build_redis_settings()
    # arq reads this attribute on the class (never instantiates WorkerSettings),
    # so the list is functionally a configuration constant, not a mutable
    # instance attribute. Annotating with ClassVar is overkill here.
    functions = [extract_conversation]  # noqa: RUF012
    on_startup = startup
    on_shutdown = shutdown
    max_jobs = 10
    job_timeout = ARQ_JOB_TIMEOUT_SECS
    keep_result = 86400
    poll_delay = 0.5
