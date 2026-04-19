"""Journal MCP Server — FastAPI application entry point.

Serves the MCP protocol over streamable HTTP (production) or
stdio (local development). Based on fastapi_template patterns:
CustomFastAPI subclass, lifespan, AppContext, structlog.
"""

import asyncio
import textwrap
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

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
from journalctl.core.logger import initialize_logger
from journalctl.middleware import BearerAuthMiddleware, MCPPathNormalizer
from journalctl.oauth.router import register_oauth_routes
from journalctl.oauth.storage import OAuthStorage
from journalctl.storage.embedding_service import EmbeddingService
from journalctl.storage.pg_setup import init_pool, setup_schema
from journalctl.tools.registry import register_tools


class CustomFastAPI(FastAPI):
    """Extended FastAPI with journal-specific attributes."""

    logger: structlog.stdlib.AsyncBoundLogger
    pool: asyncpg.Pool
    embedding_service: EmbeddingService
    settings: Settings
    mcp: FastMCP


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

    # PostgreSQL pool — each gunicorn worker creates its own pool (no --preload)
    app.pool = await init_pool(settings.database_url)
    await setup_schema(app.pool)
    await app.logger.info("PostgreSQL pool ready")

    # EmbeddingService — ONNX model loaded here before workers fork.
    # entrypoint.sh pre-downloads the model to disk; this just loads it.
    app.embedding_service = EmbeddingService()
    await app.logger.info("EmbeddingService ready")

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
    )
    app.mount("/mcp", authed_mcp)

    try:
        async with app.mcp.session_manager.run():
            yield
    finally:
        await app.logger.info("Server shutting down")
        if hydra_http_client is not None:
            await hydra_http_client.aclose()
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
            pool = await init_pool(settings.database_url)
            await setup_schema(pool)
            embedding_service = EmbeddingService()
            settings.knowledge_dir.mkdir(parents=True, exist_ok=True)
            settings.conversations_json_dir.mkdir(parents=True, exist_ok=True)
            app_ctx = AppContext(
                pool=pool,
                embedding_service=embedding_service,
                settings=settings,
                logger=structlog.get_logger("journalctl"),
            )
            mcp = create_mcp_server(app_ctx)
            try:
                mcp.run(transport="stdio")
            finally:
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
