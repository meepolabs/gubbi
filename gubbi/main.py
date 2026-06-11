"""Journal MCP Server -- FastAPI application entry point.

Serves the MCP protocol over streamable HTTP (production) or
stdio (local development). Application-scoped resources (pools,
cipher, MCP server, auth strategies) are written to ``app.state`` by
the lifespan and read back through typed accessors in
``gubbi.app_state``.
"""

from __future__ import annotations

import asyncio
import textwrap
from contextlib import asynccontextmanager, suppress
from typing import TYPE_CHECKING, Any
from uuid import UUID

import asyncpg
import httpx
import redis.asyncio as aioredis
import structlog
from arq import create_pool as arq_create_pool
from arq.connections import RedisSettings as ArqRedisSettings
from fastapi import FastAPI, Request, Response, status
from fastapi.responses import JSONResponse
from gubbi_common.auth.prm import build_prm_metadata_url
from gubbi_common.bootstrap import StartupProbe, StartupRunner
from gubbi_common.bootstrap.probes import PgLogProbe, RedisPingProbe
from gubbi_common.budget import PRE_CHARGE_LUA, BudgetHelper
from redis.exceptions import RedisError
from starlette.types import ASGIApp  # noqa: TC002 (used in runtime variable annotation)

from gubbi.app_context import AppContext
from gubbi.app_state import get_optional_app_ctx, get_optional_redis_client
from gubbi.auth.hydra import HydraIntrospector, InMemoryHydraCache
from gubbi.auth.strategies import (
    ApiKeyStrategy,
    AuthStrategy,
    HydraStrategy,
    SelfHostStrategy,
    TrustGatewayStrategy,
)
from gubbi.bootstrap import (
    BindAddressProbe,
    ReplicaCountWarnProbe,
    build_mcp_middleware,
    decode_gateway_secret,
    setup_oauth,
    teardown_lifespan_resources,
)
from gubbi.config import (
    ALLOWED_ORIGINS,
    HYDRA_INTROSPECT_TIMEOUT_SECS,
    REQUIRED_OAUTH_SCOPE,
    Settings,
    get_settings,
)
from gubbi.constants import (
    DB_HEALTH_ACQUIRE_TIMEOUT_SECS,
    DB_HEALTH_QUERY_TIMEOUT_SECS,
    REDIS_HEALTH_PING_TIMEOUT_SECS,
)
from gubbi.crypto.cipher import ContentCipher, load_master_keys_from_env
from gubbi.extraction.orphan_cleanup import run_orphan_cleanup
from gubbi.mcp_validation import JournalFastMCP
from gubbi.middleware import (
    CorrelationIDMiddleware,
    MCPPathNormalizer,
)
from gubbi.storage.embedding_service import EmbeddingService
from gubbi.storage.exceptions import DatabaseUnavailable
from gubbi.storage.pg_setup import init_pool
from gubbi.telemetry import configure_otel
from gubbi.telemetry.logger import initialize_logger
from gubbi.telemetry.metrics import record_startup_probe_outcome
from gubbi.tools.registry import register_tools
from gubbi.users.bootstrap import scaffold_operator

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from arq.connections import ArqRedis
    from mcp.server.fastmcp import FastMCP

    from gubbi.oauth.storage import OAuthStorage

__all__: list[str] = [
    "create_mcp_server",
    "database_unavailable_handler",
    "general_exception_handler",
    "lifespan",
    "main",
    "mcp_health",
    "server",
]


async def _build_content_cipher(
    logger: structlog.stdlib.AsyncBoundLogger,
) -> ContentCipher | None:
    """Build ContentCipher from JOURNAL_ENCRYPTION_MASTER_KEY_V* env vars.

    Returns None if no key is configured -- acceptable during Track B
    pre-02.13; once the repo layer depends on it, a missing cipher
    surfaces as an explicit startup failure from that wiring, not here.
    Raises on malformed key material so a misconfigured deploy fails
    loudly at startup rather than at first encrypt call.
    """
    try:
        master_keys = load_master_keys_from_env()
    except ValueError as exc:
        await logger.error(
            "Content cipher startup failed -- malformed JOURNAL_ENCRYPTION_MASTER_KEY_V*",
            error=str(exc),
        )
        raise
    if not master_keys:
        await logger.warning(
            "Content cipher disabled -- set JOURNAL_ENCRYPTION_MASTER_KEY_V1 "
            "to enable app-layer encryption (required once 02.13 ships)"
        )
        return None
    try:
        cipher = ContentCipher(master_keys)
    except (TypeError, ValueError) as exc:
        await logger.error(
            "Content cipher rejected master key material",
            error=str(exc),
        )
        raise
    await logger.info(
        "Content cipher ready",
        versions=sorted(master_keys.keys()),
        active_version=cipher.active_version,
    )
    return cipher


async def _resolve_operator_user_id(
    settings: Settings,
    pool: asyncpg.Pool,
    logger: structlog.stdlib.AsyncBoundLogger,
) -> UUID | None:
    """Resolve the operator UUID for operator-identity auth modes.

    Used by the static API key path and the self-host OAuth callback --
    both represent a single operator identity and bind requests to this
    UUID. The UUID is derived by looking up users.email =
    JOURNAL_OPERATOR_EMAIL. None is a valid outcome; callers treat it as
    "operator binding absent" and fail loud on DB code paths.
    """
    if not settings.auth.operator_email:
        return None
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id FROM users WHERE email = $1 AND deleted_at IS NULL",
            settings.auth.operator_email,
        )
    if row is None:
        await logger.warning(
            "Operator email not found in users table -- operator-identity auth will be unbound",
            email=settings.auth.operator_email,
        )
        return None
    resolved = row["id"]
    await logger.info(
        "Resolved operator_user_id from DB",
        email=settings.auth.operator_email,
        operator_user_id=str(resolved),
    )
    return resolved if isinstance(resolved, UUID) else UUID(str(resolved))


def create_mcp_server(app_ctx: AppContext) -> FastMCP:
    """Create and configure the MCP server with all tools."""
    mcp = JournalFastMCP(
        "Personal Journal & Lifelong Memory",
        instructions=textwrap.dedent("""\
            Journal is the user's persistent memory layer across conversations.
            It records events, decisions, reflections, and conversations with full-text
            and semantic search.

            DATA MODEL
            Topic -- A category or area of life (e.g. 'project/mcp', 'cars/toyota').
                    Topics are containers. All entries and conversations live under a topic.
                    Topic paths are permanent, lowercase, max 2 levels deep.
            Entry -- A dated record within a topic: a decision, event, milestone, or reflection.
                    Has content (the headline) and optional reasoning (the why).
                    Created with journal_append_entry, read with journal_read_topic.
            Conversation -- A saved chat transcript within a topic.
                    Has messages, a summary, and a title.
                    Created with journal_save_conversation, browsed with journal_list_conversations.

            Hierarchy: Topic contains -> Entries + Conversations
            journal_search spans both topics and conversations.
            journal_read_topic returns all entries the topic.
            journal_list_conversations returns all conversations of the topic.

            STARTUP
            Call journal_briefing before responding to the user's first message.
            Every conversation. No exceptions.

            PROACTIVE JOURNALING
            When the user shares a decision, milestone, life event, progress update, plan, setback,
            or idea worth preserving call journal_append_entry. Do not wait for 'remember this.'
            At the end of substantive conversations, offer to save with journal_save_conversation.

            TOPIC SAFETY
            Before writing, confirm the topic exists (check briefing)"""),
        stateless_http=True,
        streamable_http_path="/",
        host=app_ctx.settings.server.host,
    )
    register_tools(mcp, app_ctx)
    return mcp


async def _build_app_ctx(
    settings: Settings,
    logger: structlog.stdlib.AsyncBoundLogger,
) -> tuple[AppContext, asyncpg.Pool, asyncpg.Pool | None, FastMCP]:
    """Build shared startup state used by both lifespan and _run_stdio.

    Returns (app_ctx, pool, admin_pool, mcp). Cleanup (pool.close()) is the
    caller's responsibility since lifecycle differs between HTTP and stdio.
    """
    admin_pool: asyncpg.Pool | None = None
    if settings.db.admin_url:
        admin_pool = await init_pool(settings.db.admin_url)
        await logger.info("Admin PG pool ready (BYPASSRLS)")

    pool = await init_pool(settings.db.app_url)
    await logger.info("PostgreSQL pool ready")

    hydra_on = bool(settings.auth.hydra_admin_url)
    if not hydra_on:
        pool_for_scaffold = admin_pool or pool
        await scaffold_operator(pool_for_scaffold, settings.auth.operator_email, settings.timezone)
        await logger.info("Auto-scaffold operator row complete (Mode 1/2)")
    else:
        await logger.info("Skipping auto-scaffold -- Mode 3 (cloud-api provisions)")

    embedding_service = EmbeddingService()
    await logger.info("EmbeddingService ready")

    settings.conversations_json_dir.mkdir(parents=True, exist_ok=True)

    if admin_pool is None and settings.auth.operator_email:
        await logger.warning(
            "Operator lookup will use app pool -- safe only while users table "
            "has no RLS policy. Configure JOURNAL_DB_ADMIN_URL for safety."
        )
    operator_user_id = await _resolve_operator_user_id(settings, admin_pool or pool, logger)

    cipher = await _build_content_cipher(logger)

    app_ctx = AppContext(
        pool=pool,
        embedding_service=embedding_service,
        settings=settings,
        logger=logger,
        admin_pool=admin_pool,
        operator_user_id=operator_user_id,
        cipher=cipher,
    )
    mcp = create_mcp_server(app_ctx)
    return app_ctx, pool, admin_pool, mcp


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: startup and shutdown.

    Init phase opens long-lived resources (DB pools, OAuth storage,
    optional Hydra HTTP client, Redis client/pool) and then drives
    ``StartupRunner`` with the canonical probe sequence in TWO PHASES:

    * Phase 1 (config-only) runs BEFORE any wiring -- currently just
      ``BindAddressProbe``.  The bind-address contract violation is a
      deploy-time misconfiguration; failing here means we abort before
      allocating pools / OAuth storage / Redis client / Hydra HTTP
      client, matching the original spec's "fail before resources" intent.
    * Phase 2 (resources) runs AFTER pools / Redis / OAuth are wired and
      covers ``PgLogProbe`` -> ``RedisPingProbe`` -> ``ReplicaCountWarnProbe``
      in the canonical order pinned by ``test_lifespan_probe_trace.py``.

    Required-probe failure escalates as ``ProbeFailure``; the outer
    ``finally`` routes through :func:`teardown_lifespan_resources` so the
    same helper covers pre-yield init failure and clean shutdown alike.
    The same ``StartupRunner`` instance is reused across both phases --
    it has no per-invocation state; the per-call timeout budget applies
    to each phase independently.
    """
    settings = get_settings()

    initialize_logger("gubbi", log_dir=str(settings.log_dir))
    logger = structlog.get_logger("gubbi")

    configure_otel(app)

    await logger.info("Server starting up")

    pool: asyncpg.Pool | None = None
    admin_pool: asyncpg.Pool | None = None
    redis_pool_handle: aioredis.ConnectionPool | None = None
    redis_client: aioredis.Redis | None = None
    oauth_storage: OAuthStorage | None = None
    hydra_http_client: httpx.AsyncClient | None = None
    arq_pool: ArqRedis | None = None
    cron_task: asyncio.Task[None] | None = None

    runner = StartupRunner(
        app_env=settings.app_env,
        outcome_counter=record_startup_probe_outcome,
    )

    try:
        # Phase 1 (config-only probes).  Runs BEFORE wiring so a
        # trust-gateway misconfiguration aborts boot before we open
        # pools, OAuth storage, Redis client, or the Hydra HTTP client.
        # The original lifespan-probe spec called for fail-before-
        # resources; the ordering pin in
        # ``tests/unit/test_lifespan_probe_trace.py`` enforces it.
        config_probes: list[StartupProbe] = [
            BindAddressProbe(
                host=settings.server.host,
                trust_gateway=settings.auth.trust_gateway,
            ),
        ]
        await runner.run(config_probes, logger=logger)

        # Core startup: pools, operator scaffold, caching, cipher.
        app_ctx, pool, admin_pool, mcp = await _build_app_ctx(settings, logger)
        app.state.app_ctx = app_ctx
        operator_user_id = app_ctx.operator_user_id

        # OAuth -- storage, routes, expired-token cleanup.
        oauth_storage, token_validator = await setup_oauth(app, settings)

        # Hydra introspector -- optional, activated when JOURNAL_HYDRA_ADMIN_URL is set.
        introspector: HydraIntrospector | None = None
        if settings.auth.hydra_admin_url:
            hydra_http_client = httpx.AsyncClient(timeout=HYDRA_INTROSPECT_TIMEOUT_SECS)
            introspector = HydraIntrospector(
                admin_url=settings.auth.hydra_admin_url,
                http_client=hydra_http_client,
                logger=logger,
                cache=InMemoryHydraCache(),
                timeout_seconds=HYDRA_INTROSPECT_TIMEOUT_SECS,
            )
            await logger.info("Hydra introspector ready", admin_url=settings.auth.hydra_admin_url)

        # Shared Redis client for SSE pub/sub (extraction progress).
        redis_pool_handle = aioredis.ConnectionPool.from_url(str(settings.redis_url))
        redis_client = aioredis.Redis(connection_pool=redis_pool_handle)
        app.state.redis_client = redis_client

        # Phase 2 (resource probes).  Runs AFTER wiring; consumes the
        # DB pool, the Redis client, and the live pool max sizes for
        # the replica-count budget warning.  Order is pinned by
        # ``tests/unit/test_lifespan_probe_trace.py``: pg_log ->
        # redis_ping -> replica_count.  Adding / reordering probes
        # forces that test to update.
        resource_probes: list[StartupProbe] = [
            PgLogProbe(pool=pool, mode=settings.pg_log_probe_mode),
            RedisPingProbe(client=redis_client),
            ReplicaCountWarnProbe(
                replica_count=settings.replica_count,
                pool_max_per_pod=pool.get_max_size()
                + (admin_pool.get_max_size() if admin_pool is not None else 0),
            ),
        ]
        await runner.run(resource_probes, logger=logger)

        # Gateway HMAC secret (three warning branches preserved verbatim).
        app.state.gubbi_gateway_secret = await decode_gateway_secret(
            settings.auth.gateway_secret,
            require_signature=settings.auth.gateway_require_signature,
            trust_gateway=settings.auth.trust_gateway,
            logger=logger,
        )

        # Expose auth dependencies on app.state for REST API routes.
        app.state.hydra_introspector = introspector
        app.state.selfhost_token_validator = token_validator  # may be None
        app.state.operator_user_id = operator_user_id  # may be None

        # Arq pool for background job enqueue (separate from the SSE aioredis client).
        arq_pool = await arq_create_pool(ArqRedisSettings.from_dsn(str(settings.redis_url)))
        app.state.arq_pool = arq_pool
        app_ctx.arq_pool = arq_pool

        # BudgetHelper -- shared facade over Redis pre-charge + delta writes.
        # Disabled in self-host mode (Mode 1/2); constructed only when the
        # operator has opted into LLM budget enforcement.
        if settings.llm.llm_budget_enabled:
            pre_charge_script = redis_client.register_script(PRE_CHARGE_LUA)
            budget_helper = BudgetHelper(
                redis=redis_client,  # type: ignore[arg-type]  # duck-typed Protocol vs aioredis.Redis
                pre_charge_script=pre_charge_script,
            )
            app.state.budget_helper = budget_helper
            app_ctx.budget_helper = budget_helper
            await logger.info("BudgetHelper ready (lifespan)")
        else:
            app.state.budget_helper = None
            app_ctx.budget_helper = None
            await logger.info("BudgetHelper disabled (llm_budget_enabled=False)")

        # Orphan cleanup cron: marks stale pending extraction_jobs rows as failed.
        # Requires admin_pool (BYPASSRLS) for cross-tenant sweep; skipped when no
        # admin pool is configured (single-tenant dev fallback).
        app.state.background_tasks = set()
        if admin_pool is not None:
            cron_task = asyncio.create_task(
                run_orphan_cleanup(
                    admin_pool,
                    threshold_minutes=settings.llm.orphan_cleanup_threshold_minutes,
                    budget_helper=app.state.budget_helper,
                ),
            )
            app.state.background_tasks.add(cron_task)

        # Mode 3 (hosted) disables the shared static API key path -- operators
        # authenticate via Hydra like any user. Pass api_key="" so the timing-safe
        # compare in the middleware can never match (every token is >= one char).
        effective_api_key = "" if introspector is not None else settings.auth.api_key

        # Trust-gateway mode authenticates via gateway-signed headers only --
        # an API key left in env is dead config. Fail-fast so a Mode-2 -> Mode-3
        # cutover that forgot to unset JOURNAL_API_KEY surfaces at boot rather
        # than silently dropping the key and confusing operators.
        if settings.auth.trust_gateway and effective_api_key:
            await logger.error("auth_strategy_conflict")
            raise ValueError(
                "JOURNAL_API_KEY is set but JOURNAL_TRUST_GATEWAY is enabled. "
                "API key auth is disabled in trust-gateway mode. "
                "Either unset JOURNAL_API_KEY or unset JOURNAL_TRUST_GATEWAY."
            )

        # Point clients at the OAuth protected-resource metadata doc so they can
        # discover the authorization server (MCP spec 2025-11-25). Only surface
        # the URL when OAuth is actually wired -- pure Mode 1 API-key deployments
        # have no metadata endpoint to advertise.
        protected_resource_metadata_url: str | None = None
        if introspector is not None or token_validator is not None:
            server_base = settings.server.url.rstrip("/")
            protected_resource_metadata_url = build_prm_metadata_url(
                f"{server_base}/mcp", legacy_suffix=True
            )

        # Build auth strategy list. Trust-gateway deployments use ONLY
        # TrustGatewayStrategy; non-trust builds compose ApiKey + Hydra + SelfHost.
        if settings.auth.trust_gateway:
            auth_strategies: list[AuthStrategy] = [
                TrustGatewayStrategy(
                    gateway_secret=app.state.gubbi_gateway_secret,
                    gateway_require_signature=settings.auth.gateway_require_signature,
                ),
            ]
        else:
            _raw_strategies: list[AuthStrategy | None] = [
                (
                    ApiKeyStrategy(
                        api_key=effective_api_key,
                        api_key_scopes=tuple(settings.auth.api_key_scopes),
                        operator_user_id=operator_user_id,
                    )
                )
                if effective_api_key
                else None,
                HydraStrategy(introspector=introspector) if introspector is not None else None,
                SelfHostStrategy(
                    token_validator=token_validator,
                    operator_user_id=operator_user_id,
                )
                if token_validator is not None
                else None,
            ]
            auth_strategies = [s for s in _raw_strategies if s is not None]

        app.state.auth_strategies = auth_strategies

        # Assemble MCP middleware chain and mount at /mcp.
        mcp_http = mcp.streamable_http_app()
        origin_validated_mcp = build_mcp_middleware(
            mcp_http,
            strategies=auth_strategies,
            required_scope=REQUIRED_OAUTH_SCOPE,
            protected_resource_metadata_url=protected_resource_metadata_url,
            allowed_origins=ALLOWED_ORIGINS,
        )
        app.mount("/mcp", origin_validated_mcp)

        async with mcp.session_manager.run():
            yield
    finally:
        await logger.info("Server shutting down")
        if cron_task is not None:
            cron_task.cancel()
            with suppress(asyncio.CancelledError):
                await cron_task
        if arq_pool is not None:
            with suppress(Exception, asyncio.CancelledError):
                await arq_pool.close()
        # Single helper covers pre-yield init failure and clean shutdown
        # alike: each parameter is None-safe so partially-initialized
        # state at the failure point still tears down what was opened.
        # Replaces the 8-9 duplicated try/except teardown blocks the
        # pre-StartupRunner lifespan body carried.
        await teardown_lifespan_resources(
            pool=pool,
            admin_pool=admin_pool,
            redis_client=redis_client,
            redis_pool=redis_pool_handle,
            oauth_storage=oauth_storage,
            hydra_http_client=hydra_http_client,
        )


# Construct the FastAPI app with an EMPTY middleware list. All custom
# middleware -- including ones that don't strictly need to be outside
# the OTel server span -- is composed at the ASGI layer below, mirroring
# gubbi-cloud's pattern.
#
# Why all-ASGI rather than the hybrid model (some via app.add_middleware,
# some via ASGI wrap):
#
#   1. SSE safety. FastMCP's streamable-http transport returns SSE
#      response bodies on /mcp/. A future ``@app.middleware("http")``
#      decorator would create a Starlette BaseHTTPMiddleware which
#      buffers both request and response bodies, silently breaking
#      SSE streaming. Keeping ``user_middleware`` empty closes that
#      tripwire structurally.
#   2. CorrelationIDMiddleware MUST run before any OTel layer because
#      CorrelationSpanProcessor.on_start reads the request-scoped
#      ContextVar this middleware sets. FastAPIInstrumentor patches
#      build_middleware_stack so OpenTelemetryMiddleware lands OUTSIDE
#      anything in user_middleware -- the only place state-setting
#      middleware can live is OUTSIDE the FastAPI app.
#   3. Symmetry with gubbi-cloud's gateway shape (one mental model
#      across the two services).
#
# MCPPathNormalizer ends up outside the OTel server span as a
# consequence. The cosmetic effect is that the span's http.target
# attribute reflects the rewritten path (``/mcp/``) rather than the
# original (``/mcp``). The rewrite is a sub-microsecond dict mutation
# that cannot raise; nothing meaningful is lost by being outside OTel.
app: FastAPI = FastAPI(
    title="gubbi",
    description="Personal journal MCP server",
    version="0.2.0",
    lifespan=lifespan,
)


# Register REST API routers
from gubbi.api.v1.extraction import router as extraction_router  # noqa: E402
from gubbi.api.v1.ingest import router as ingest_router  # noqa: E402
from gubbi.api.v1.web.conversations import router as web_conversations_router  # noqa: E402
from gubbi.api.v1.web.search import router as web_search_router  # noqa: E402
from gubbi.api.v1.web.stats import router as web_stats_router  # noqa: E402
from gubbi.api.v1.web.topic_admin import router as web_topic_admin_router  # noqa: E402
from gubbi.api.v1.web.topics import router as web_topics_router  # noqa: E402

app.include_router(ingest_router, prefix="/api/v1")
app.include_router(extraction_router, prefix="/api/v1")
app.include_router(web_topics_router, prefix="/api/v1")
app.include_router(web_conversations_router, prefix="/api/v1")
app.include_router(web_search_router, prefix="/api/v1")
app.include_router(web_stats_router, prefix="/api/v1")
app.include_router(web_topic_admin_router, prefix="/api/v1")


@app.exception_handler(DatabaseUnavailable)
async def database_unavailable_handler(
    request: Request,
    exc: DatabaseUnavailable,
) -> JSONResponse:
    """Map transient DB errors to HTTP 503 with Retry-After header."""
    return JSONResponse(
        status_code=503,
        content={"detail": "database temporarily unavailable"},
        headers={"Retry-After": "5"},
    )


@app.exception_handler(Exception)
async def general_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Handle unhandled exceptions."""
    logger = structlog.get_logger("gubbi")
    await logger.error(
        "Unhandled exception",
        exc_info=exc,
        path=request.url.path,
        method=request.method,
    )
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"},
    )


@app.get("/health")
async def mcp_health() -> dict[str, Any]:
    """Liveness probe for Docker health checks.

    Always 200 once uvicorn is bound -- it does NOT inspect lifespan
    state or pool reachability.  Use ``/health/ready`` for readiness
    (drained when ``app_ctx`` / pool is unset).

    NOTE: do NOT add @app.get("/mcp/") here -- it shadows the
    FastMCP streamable-http app mounted at /mcp via app.mount(...).
    Claude.ai opens a GET to /mcp/ to start the SSE handshake; if
    this route intercepts it, the client receives application/json
    instead of text/event-stream and bails with "Authorization
    failed" (a misleading client-side error). Bug caught during
    a deploy on 2026-04-30.
    """
    return {"status": "ok"}


@app.get("/health/ready")
async def mcp_health_ready(request: Request) -> Response:
    """Readiness probe -- 200 only when the lifespan finished startup.

    Returns 503 during cold-start (before lifespan attaches
    ``app.state.app_ctx``) and during shutdown (after the pool is
    closed) so an upstream load balancer drains traffic off this
    instance at the right moments.  Symmetric with cloud-api's
    ``/health/ready`` so HEALTHCHECK and orchestrator readiness probes
    share a single contract across both services.

    Order of checks (each independently observable in the response):

    1. ``app_ctx`` not initialised -> 503 ``not_ready``.
    2. Redis PING timeout / driver error -> 503 ``redis_unreachable``.
    3. DB acquire / SELECT 1 driver error -> 503 ``db_unreachable``.

    Each failure envelope additionally carries an ``error_class`` field
    holding the underlying exception class name (no message text), so
    pool-exhaustion vs socket-reset vs query-timeout vs connection-
    refused are distinguishable at a glance on the on-call dashboard
    without leaking driver detail into the body.

    Short timeouts (2s acquire, 1s SELECT, 1s Redis PING) prevent a
    single readiness poll from holding a slot a real request wants
    under pool saturation -- mirrors gubbi-cloud's split.
    """
    app_ctx = get_optional_app_ctx(request)
    if app_ctx is None:
        return Response(
            content='{"status":"not_ready"}',
            media_type="application/json",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )
    redis_client = get_optional_redis_client(request)
    if redis_client is not None:
        try:
            async with asyncio.timeout(REDIS_HEALTH_PING_TIMEOUT_SECS):
                await redis_client.ping()
        except (RedisError, OSError, TimeoutError) as exc:
            error_class = type(exc).__name__
            return Response(
                content=f'{{"status":"redis_unreachable","error_class":"{error_class}"}}',
                media_type="application/json",
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
    try:
        async with app_ctx.pool.acquire(timeout=DB_HEALTH_ACQUIRE_TIMEOUT_SECS) as conn:
            await conn.fetchval("SELECT 1", timeout=DB_HEALTH_QUERY_TIMEOUT_SECS)
    except (asyncpg.PostgresError, OSError, TimeoutError) as exc:
        error_class = type(exc).__name__
        return Response(
            content=f'{{"status":"db_unreachable","error_class":"{error_class}"}}',
            media_type="application/json",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )
    return Response(
        content='{"status":"ok"}',
        media_type="application/json",
        status_code=status.HTTP_200_OK,
    )


# ASGI-layer middleware composition. Order, outermost first:
#
#   CorrelationIDMiddleware -> MCPPathNormalizer -> FastAPI app
#
# CorrelationIDMiddleware is OUTSIDE the FastAPI app so the request-
# scoped correlation_id ContextVar is populated before any OTel layer
# runs. CorrelationSpanProcessor.on_start (registered via
# configure_otel) reads the ContextVar when the FastAPI server span
# opens; with the wrap order above, the value is always present. See
# gubbi-common's CorrelationIDMiddleware docstring for the contract.
#
# MCPPathNormalizer rewrites /mcp -> /mcp/ before FastAPI's router
# sees the request. Outside-OTel placement is incidental, not required
# (it is a sub-microsecond dict mutation with no failure mode), but
# keeping the FastAPI middleware list empty closes the SSE-streaming
# tripwire described above.
#
# The deployment target (uvicorn / gunicorn / `gubbi.main:server`) is
# this wrapper. `app` remains the FastAPI handle for routes,
# decorators, and tests that need state inspection.
server: ASGIApp = CorrelationIDMiddleware(MCPPathNormalizer(app))


def main() -> None:
    """Entry point for running the server."""
    settings = get_settings()

    if settings.server.transport == "stdio":

        async def _run_stdio() -> None:
            initialize_logger("gubbi", log_dir=str(settings.log_dir))
            logger = structlog.get_logger("gubbi")
            _app_ctx, pool, admin_pool, mcp = await _build_app_ctx(settings, logger)
            try:
                mcp.run(transport="stdio")
            finally:
                if admin_pool is not None:
                    await admin_pool.close()
                await pool.close()

        asyncio.run(_run_stdio())
    else:
        import uvicorn

        uvicorn.run(
            "gubbi.main:server",
            host=settings.server.host,
            port=settings.server.port,
            reload=False,
        )


if __name__ == "__main__":
    main()
