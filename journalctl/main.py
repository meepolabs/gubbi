"""Journal MCP Server — FastAPI application entry point.

Serves the MCP protocol over streamable HTTP (production) or
stdio (local development). Based on fastapi_template patterns:
CustomFastAPI subclass, lifespan, AppContext, structlog.
"""

import asyncio
import textwrap
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from uuid import UUID

import asyncpg
import httpx
import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from mcp.server.fastmcp import FastMCP
from starlette.middleware import Middleware

from journalctl.auth.hydra import HydraIntrospector, InMemoryHydraCache
from journalctl.config import Settings, get_settings
from journalctl.core.context import AppContext
from journalctl.core.crypto import ContentCipher, load_master_keys_from_env
from journalctl.core.logger import initialize_logger
from journalctl.middleware import BearerAuthMiddleware, MCPPathNormalizer
from journalctl.oauth.router import register_oauth_routes
from journalctl.oauth.storage import OAuthStorage
from journalctl.storage.embedding_service import EmbeddingService
from journalctl.storage.pg_setup import init_pool
from journalctl.tools.registry import register_tools


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


async def _resolve_founder_user_id(
    settings: Settings,
    pool: asyncpg.Pool,
    logger: structlog.stdlib.AsyncBoundLogger,
) -> UUID | None:
    """Resolve the founder UUID for legacy-auth requests.

    Precedence: JOURNAL_FOUNDER_USER_ID env override > DB lookup by
    JOURNAL_FOUNDER_EMAIL > None. None is a valid outcome; callers must
    treat it as "legacy auth unbound".
    """
    if settings.founder_user_id is not None:
        await logger.info("Using founder_user_id from env override")
        return settings.founder_user_id
    if not settings.founder_email:
        return None
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id FROM users WHERE email = $1 AND deleted_at IS NULL",
            settings.founder_email,
        )
    if row is None:
        await logger.warning(
            "Founder email not found in users table — legacy auth will be unbound",
            email=settings.founder_email,
        )
        return None
    resolved = row["id"]
    await logger.info(
        "Resolved founder_user_id from DB",
        email=settings.founder_email,
        founder_user_id=str(resolved),
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
        host=app_ctx.settings.host,
    )
    register_tools(mcp, app_ctx)
    return mcp


@asynccontextmanager
async def lifespan(app: CustomFastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: startup and shutdown."""
    settings = get_settings()

    initialize_logger("journalctl", log_dir=str(settings.log_dir))
    app.logger = structlog.get_logger("journalctl")
    await app.logger.info("Server starting up")

    app.settings = settings
    settings.knowledge_dir.mkdir(parents=True, exist_ok=True)
    settings.conversations_json_dir.mkdir(parents=True, exist_ok=True)

    # PostgreSQL pools -- each gunicorn worker creates its own (no --preload).
    # Schema is owned by alembic; run `alembic upgrade head` before first boot.
    # The runtime pool is journal_app (RLS-enforced); the admin pool is
    # journal_admin (BYPASSRLS) for cross-tenant worker paths like
    # journal_reindex and cleanup. User-scoped tool calls must never touch
    # the admin pool. Empty admin DSN falls back to the runtime pool, which
    # is fine in single-tenant dev before RLS is live.
    app.admin_pool = None
    if settings.database_url_admin:
        app.admin_pool = await init_pool(settings.database_url_admin)
        await app.logger.info("Admin PG pool ready (BYPASSRLS)")

    app.pool = await init_pool(settings.database_url)
    await app.logger.info("PostgreSQL pool ready")

    # EmbeddingService — ONNX model loaded here before workers fork.
    # entrypoint.sh pre-downloads the model to disk; this just loads it.
    app.embedding_service = EmbeddingService()
    await app.logger.info("EmbeddingService ready")

    # Founder UUID — binds legacy API-key + legacy-OAuth paths to a concrete
    # tenant so user_scoped_connection works uniformly. Look up against the
    # admin pool when available so the query bypasses RLS (once 02.05 ships,
    # the app pool would return zero rows with app.current_user_id unset).
    # users has no RLS today (migration 0005 only enables it on the 5 tenant
    # tables), so the app-pool fallback works — warn loudly so the assumption
    # is visible if users ever gets an RLS policy.
    if app.admin_pool is None and settings.founder_email:
        await app.logger.warning(
            "Founder lookup will use app pool — safe only while users table "
            "has no RLS policy. Configure JOURNAL_DATABASE_URL_ADMIN for safety."
        )
    founder_user_id = await _resolve_founder_user_id(
        settings,
        app.admin_pool or app.pool,
        app.logger,
    )

    app.cipher = await _build_content_cipher(app.logger)

    # OAuth (stays SQLite — own connection, out of scope for PG migration)
    oauth_storage = OAuthStorage(settings.oauth_db_path)
    _ = oauth_storage.conn
    expired = oauth_storage.cleanup_expired()
    if expired:
        await app.logger.info("OAuth cleanup", expired_tokens=expired)

    token_validator = register_oauth_routes(app, oauth_storage, settings)
    if token_validator:
        await app.logger.info("OAuth endpoints registered")

    app_ctx = AppContext(
        pool=app.pool,
        embedding_service=app.embedding_service,
        settings=settings,
        logger=app.logger,
        admin_pool=app.admin_pool,
        founder_user_id=founder_user_id,
        cipher=app.cipher,
    )

    app.mcp = create_mcp_server(app_ctx)
    mcp_http = app.mcp.streamable_http_app()

    # Hydra introspector — optional, activated when JOURNAL_HYDRA_ADMIN_URL is set
    introspector: HydraIntrospector | None = None
    hydra_http_client: httpx.AsyncClient | None = None
    if settings.hydra_admin_url:
        hydra_http_client = httpx.AsyncClient(timeout=settings.hydra_introspect_timeout)
        introspector = HydraIntrospector(
            admin_url=settings.hydra_admin_url,
            http_client=hydra_http_client,
            logger=app.logger,
            cache=InMemoryHydraCache(),
            timeout_seconds=settings.hydra_introspect_timeout,
        )
        await app.logger.info("Hydra introspector ready", admin_url=settings.hydra_admin_url)

    authed_mcp = BearerAuthMiddleware(
        mcp_http,
        api_key=settings.api_key,
        introspector=introspector,
        required_scope=settings.required_oauth_scope,
        legacy_token_validator=token_validator,
        founder_user_id=founder_user_id,
    )
    app.mount("/mcp", authed_mcp)

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


# Create FastAPI app
server = CustomFastAPI(
    title="journalctl",
    description="Personal journal MCP server",
    version="0.2.0",
    lifespan=lifespan,
    middleware=[Middleware(MCPPathNormalizer)],
)


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
@server.get("/mcp/")
async def mcp_health() -> dict:
    """Liveness probe for Docker health checks."""
    return {"status": "ok"}


def main() -> None:
    """Entry point for running the server."""
    settings = get_settings()

    if settings.transport == "stdio":

        async def _run_stdio() -> None:
            logger = structlog.get_logger("journalctl")
            pool = await init_pool(settings.database_url)
            admin_pool: asyncpg.Pool | None = None
            if settings.database_url_admin:
                admin_pool = await init_pool(settings.database_url_admin)
            embedding_service = EmbeddingService()
            settings.knowledge_dir.mkdir(parents=True, exist_ok=True)
            settings.conversations_json_dir.mkdir(parents=True, exist_ok=True)
            founder_user_id = await _resolve_founder_user_id(settings, admin_pool or pool, logger)
            cipher = await _build_content_cipher(logger)
            app_ctx = AppContext(
                pool=pool,
                embedding_service=embedding_service,
                settings=settings,
                logger=logger,
                admin_pool=admin_pool,
                founder_user_id=founder_user_id,
                cipher=cipher,
            )
            mcp = create_mcp_server(app_ctx)
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
            host=settings.host,
            port=settings.port,
            reload=False,
        )


if __name__ == "__main__":
    main()
