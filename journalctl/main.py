"""Journal MCP Server — FastAPI application entry point.

Serves the MCP protocol over streamable HTTP (production) or
stdio (local development). Based on fastapi_template patterns:
CustomFastAPI subclass, lifespan, AppContext, structlog.
"""

import asyncio
import ipaddress
import socket
import textwrap
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from uuid import UUID

import asyncpg
import httpx
import redis.asyncio as aioredis
import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from mcp.server.fastmcp import FastMCP
from starlette.middleware import Middleware

from journalctl.auth.hydra import HydraIntrospector, InMemoryHydraCache
from journalctl.config import (
    ALLOWED_ORIGINS,
    HYDRA_INTROSPECT_TIMEOUT_SECS,
    REQUIRED_OAUTH_SCOPE,
    Settings,
    get_settings,
)
from journalctl.core.context import AppContext
from journalctl.core.crypto import ContentCipher, load_master_keys_from_env
from journalctl.core.logger import initialize_logger
from journalctl.middleware import (
    BearerAuthMiddleware,
    CorrelationIDMiddleware,
    MCPPathNormalizer,
    OriginValidationMiddleware,
)
from journalctl.oauth.router import register_oauth_routes
from journalctl.oauth.storage import OAuthStorage
from journalctl.storage.embedding_service import EmbeddingService
from journalctl.storage.pg_setup import init_pool
from journalctl.telemetry import configure_otel
from journalctl.tools.registry import register_tools
from journalctl.users.bootstrap import scaffold_operator


class CustomFastAPI(FastAPI):
    """Extended FastAPI with journal-specific attributes."""

    logger: structlog.stdlib.AsyncBoundLogger
    pool: asyncpg.Pool
    admin_pool: asyncpg.Pool | None
    embedding_service: EmbeddingService
    settings: Settings
    cipher: ContentCipher | None
    mcp: FastMCP


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
    mcp = FastMCP(
        "Personal Journal & Lifelong Memory",
        instructions=textwrap.dedent("""\
            Journal is the user's persistent memory layer across conversations.
            It records events, decisions, reflections, and conversations with full-text
            and semantic search.

            DATA MODEL
            Topic — A category or area of life (e.g. 'project/mcp', 'cars/toyota').
                    Topics are containers. All entries and conversations live under a topic.
                    Topic paths are permanent, lowercase, max 2 levels deep.
            Entry — A dated record within a topic: a decision, event, milestone, or reflection.
                    Has content (the headline) and optional reasoning (the why).
                    Created with journal_append_entry, read with journal_read_topic.
            Conversation — A saved chat transcript within a topic.
                    Has messages, a summary, and a title.
                    Created with journal_save_conversation, browsed with journal_list_conversations.

            Hierarchy: Topic contains → Entries + Conversations
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


async def _check_trust_gateway_bind_address(
    host: str,
    trust_gateway: bool,
    logger: structlog.stdlib.AsyncBoundLogger,
) -> None:
    """Fail fast when trust_gateway is paired with a public-routable bind address."""
    if not trust_gateway:
        return

    addresses: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []

    # Try parsing the bind address as a literal IP first.
    try:
        addresses.append(ipaddress.ip_address(host))
    except ValueError:
        # Hostname -- resolve before classifying.
        loop = asyncio.get_running_loop()
        try:
            resolved = await loop.run_in_executor(
                None,
                socket.getaddrinfo,
                host,
                None,
                socket.AF_UNSPEC,
                socket.SOCK_STREAM,
            )
        except socket.gaierror:
            raise RuntimeError(
                f"JOURNAL_TRUST_GATEWAY=true -- bind address '{host}' failed to resolve. "
                "Set JOURNAL_HOST to a resolvable address."
            ) from None

        for _family, _type, _proto, _canonname, sockaddr in resolved:
            addresses.append(ipaddress.ip_address(sockaddr[0]))

    has_unspecified = False
    for addr in addresses:
        if addr.is_unspecified:
            has_unspecified = True
            continue

        if addr.is_loopback or addr.is_private or addr.is_link_local:
            continue

        # Public-routable -- fail fast.
        raise RuntimeError(
            f"JOURNAL_TRUST_GATEWAY=true is incompatible with bind address "
            f"'{host}' -- set JOURNAL_HOST to a loopback or private-network address."
        )

    if has_unspecified:
        await logger.warning(
            "JOURNAL_TRUST_GATEWAY=true with bind address '%s' -- "
            "exposure depends on network/proxy layer",
            host,
        )


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

    settings.knowledge_dir.mkdir(parents=True, exist_ok=True)
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
async def lifespan(app: CustomFastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: startup and shutdown."""
    settings = get_settings()

    initialize_logger("journalctl", log_dir=str(settings.log_dir))
    app.logger = structlog.get_logger("journalctl")

    configure_otel(app)

    await app.logger.info("Server starting up")

    app.settings = settings

    await _check_trust_gateway_bind_address(
        settings.server.host, settings.auth.trust_gateway, app.logger
    )

    app_ctx, pool, admin_pool, mcp = await _build_app_ctx(settings, app.logger)
    app.pool = pool
    app.admin_pool = admin_pool
    app.embedding_service = app_ctx.embedding_service
    app.cipher = app_ctx.cipher
    app.mcp = mcp
    app.state.app_ctx = app_ctx

    operator_user_id = app_ctx.operator_user_id

    # OAuth (stays SQLite -- own connection, out of scope for PG migration)
    oauth_storage = OAuthStorage(settings.oauth_db_path)
    _ = oauth_storage.conn
    expired = oauth_storage.cleanup_expired()
    if expired:
        await app.logger.info("OAuth cleanup", expired_tokens=expired)

    token_validator = register_oauth_routes(app, oauth_storage, settings)
    if token_validator:
        await app.logger.info("OAuth endpoints registered")

    # Gateway HMAC secret: decode the hex-encoded shared secret from config.
    # If empty or invalid, stash None (verification will fall through to
    # legacy path or 503 depending on gateway_require_signature).
    gateway_secret: bytes | None = None
    if settings.auth.gateway_secret:
        try:
            decoded = bytes.fromhex(settings.auth.gateway_secret)
            if len(decoded) >= 32:
                gateway_secret = decoded
            else:
                await app.logger.warning(
                    "JOURNAL_GATEWAY_SECRET decodes to less than 32 bytes "
                    "-- gateway signature verification disabled"
                )
        except ValueError:
            await app.logger.warning(
                "JOURNAL_GATEWAY_SECRET is not valid hex "
                "-- gateway signature verification disabled"
            )
    if settings.auth.gateway_require_signature and gateway_secret is None:
        await app.logger.warning(
            "JOURNAL_GATEWAY_REQUIRE_SIGNATURE=true but gateway secret "
            "is missing or invalid -- signed requests will fail with 503"
        )
    if settings.auth.trust_gateway and not settings.auth.gateway_require_signature:
        await app.logger.warning(
            "trust_gateway=true but gateway_require_signature=false -- "
            "requests are accepted without HMAC verification"
        )
    app.state.journalctl_gateway_secret = gateway_secret

    # Expose auth dependencies on app.state for REST API routes
    app.state.hydra_introspector = None  # may be replaced below
    app.state.selfhost_token_validator = token_validator  # may be None
    app.state.operator_user_id = operator_user_id  # may be None

    mcp_http = app.mcp.streamable_http_app()

    # Hydra introspector -- optional, activated when JOURNAL_HYDRA_ADMIN_URL is set.
    # When on, the static API key path is disabled (hosted mode is OAuth-only).
    introspector: HydraIntrospector | None = None
    hydra_http_client: httpx.AsyncClient | None = None
    if settings.auth.hydra_admin_url:
        hydra_http_client = httpx.AsyncClient(timeout=HYDRA_INTROSPECT_TIMEOUT_SECS)
        introspector = HydraIntrospector(
            admin_url=settings.auth.hydra_admin_url,
            http_client=hydra_http_client,
            logger=app.logger,
            cache=InMemoryHydraCache(),
            timeout_seconds=HYDRA_INTROSPECT_TIMEOUT_SECS,
        )
        await app.logger.info("Hydra introspector ready", admin_url=settings.auth.hydra_admin_url)
        app.state.hydra_introspector = introspector

    # Shared Redis client for SSE pub/sub (extraction progress).
    redis_client = aioredis.from_url(str(settings.redis_url))
    app.state.redis_client = redis_client

    # Mode 3 (hosted) disables the shared static API key path -- operators
    # authenticate via Hydra like any user. Pass api_key="" so the timing-safe
    # compare in the middleware can never match (every token is >= one char).
    effective_api_key = "" if introspector is not None else settings.auth.api_key

    # Point clients at the OAuth protected-resource metadata doc so they can
    # discover the authorization server (MCP spec 2025-11-25). Only surface
    # the URL when OAuth is actually wired -- pure Mode 1 API-key deployments
    # have no metadata endpoint to advertise.
    #
    # RFC 9728 mounts the metadata at <.well-known>/oauth-protected-resource
    # + the resource path, so for resource <server_url>/mcp the SDK serves
    # the doc at /.well-known/oauth-protected-resource/mcp. Must match the
    # resource_url passed to create_protected_resource_routes in router.py
    # (which uses the same /mcp suffix); a mismatch breaks discovery.
    protected_resource_metadata_url: str | None = None
    if introspector is not None or token_validator is not None:
        server_base = settings.server.url.rstrip("/")
        protected_resource_metadata_url = f"{server_base}/.well-known/oauth-protected-resource/mcp"
        # Guard: resource_metadata must be an absolute URI (RFC 8414 s3).
        if protected_resource_metadata_url and not any(
            protected_resource_metadata_url.startswith(pre) for pre in ("http://", "https://")
        ):
            protected_resource_metadata_url = None

    authed_mcp = BearerAuthMiddleware(
        mcp_http,
        api_key=effective_api_key,
        introspector=introspector,
        required_scope=REQUIRED_OAUTH_SCOPE,
        selfhost_token_validator=token_validator,
        operator_user_id=operator_user_id,
        protected_resource_metadata_url=protected_resource_metadata_url,
        trust_gateway=settings.auth.trust_gateway,
        gateway_secret=gateway_secret,
        gateway_require_signature=settings.auth.gateway_require_signature,
        api_key_scopes=frozenset(settings.auth.api_key_scopes),
    )
    # Origin validation: prevents DNS-rebinding attacks on the MCP endpoint.
    origin_validated_mcp = OriginValidationMiddleware(authed_mcp, ALLOWED_ORIGINS)
    app.mount("/mcp", origin_validated_mcp)

    try:
        async with app.mcp.session_manager.run():
            yield
    finally:
        await app.logger.info("Server shutting down")
        if hydra_http_client is not None:
            await hydra_http_client.aclose()
        if app.admin_pool is not None:
            await app.admin_pool.close()
        await app.pool.close()
        oauth_storage.close()
        await redis_client.aclose()


# Create FastAPI app
server = CustomFastAPI(
    title="journalctl",
    description="Personal journal MCP server",
    version="0.2.0",
    lifespan=lifespan,
    middleware=[
        Middleware(MCPPathNormalizer),
        Middleware(CorrelationIDMiddleware),
    ],
)


# Register REST API routers
from journalctl.api.v1.extraction import router as extraction_router  # noqa: E402
from journalctl.api.v1.ingest import router as ingest_router  # noqa: E402

server.include_router(ingest_router, prefix="/api/v1")
server.include_router(extraction_router, prefix="/api/v1")


@server.exception_handler(Exception)
async def general_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Handle unhandled exceptions."""
    logger = structlog.get_logger("journalctl")
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


@server.get("/health")
async def mcp_health() -> dict:
    """Liveness probe for Docker health checks.

    NOTE: do NOT add @server.get("/mcp/") here -- it shadows the
    FastMCP streamable-http app mounted at /mcp via app.mount(...).
    Claude.ai opens a GET to /mcp/ to start the SSE handshake; if
    this route intercepts it, the client receives application/json
    instead of text/event-stream and bails with "Authorization
    failed" (a misleading client-side error). Bug caught during
    M3 deploy on bunsamosa 2026-04-30.
    """
    return {"status": "ok"}


def main() -> None:
    """Entry point for running the server."""
    settings = get_settings()

    if settings.server.transport == "stdio":

        async def _run_stdio() -> None:
            initialize_logger("journalctl", log_dir=str(settings.log_dir))
            logger = structlog.get_logger("journalctl")
            _app_ctx, pool, admin_pool, mcp = await _build_app_ctx(settings, logger)
            try:
                mcp.run(transport="stdio")
            finally:
                if admin_pool is not None:
                    await admin_pool.close()
                await pool.close()

        asyncio.run(_run_stdio())
    else:
        import uvicorn  # noqa: PLC0415

        uvicorn.run(
            "journalctl.main:server",
            host=settings.server.host,
            port=settings.server.port,
            reload=False,
        )


if __name__ == "__main__":
    main()
