"""Journal MCP Server — FastAPI application entry point.

Serves the MCP protocol over streamable HTTP (production) or
stdio (local development). Based on fastapi_template
patterns: CustomFastAPI subclass, lifespan, structlog.
"""

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from mcp.server.fastmcp import FastMCP
from starlette.middleware import Middleware

from journalctl.config import Settings, get_settings
from journalctl.core.logger import initialize_logger
from journalctl.memory.bootstrap import configure_env, init_service
from journalctl.memory.client import MemoryServiceProtocol
from journalctl.middleware import BearerAuthMiddleware, MCPPathNormalizer
from journalctl.oauth.router import register_oauth_routes
from journalctl.oauth.storage import OAuthStorage
from journalctl.storage.database import DatabaseStorage
from journalctl.storage.search_index import SearchIndex
from journalctl.tools.registry import register_tools


class CustomFastAPI(FastAPI):
    """Extended FastAPI with journal-specific attributes."""

    logger: structlog.BoundLogger
    storage: DatabaseStorage
    index: SearchIndex
    settings: Settings
    mcp: FastMCP
    memory_service: MemoryServiceProtocol


def create_mcp_server(
    storage: DatabaseStorage,
    index: SearchIndex,
    settings: Settings,
    memory_service: MemoryServiceProtocol,
) -> FastMCP:
    """Create and configure the MCP server with all tools."""
    mcp = FastMCP(
        "Personal Journal & Lifelong Memory",
        instructions=(
            "Persistent personal journal and memory — records events, decisions, "
            "reflections, and conversations with full-text and semantic search. "
            "Call journal_briefing first to load the user's identity and context. "
            "Use journal_append proactively when the user shares life updates."
        ),
        stateless_http=True,
        streamable_http_path="/",
        host=settings.host,
    )
    register_tools(mcp, storage, index, settings, memory_service=memory_service)
    return mcp


@asynccontextmanager
async def lifespan(app: CustomFastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: startup and shutdown."""
    settings = get_settings()

    # Initialize logger
    initialize_logger("journalctl", log_dir=str(settings.log_dir))
    app.logger = structlog.get_logger("journalctl")
    await app.logger.info("Server starting up")

    # Initialize storage
    app.settings = settings
    app.storage = DatabaseStorage(settings.db_path, settings.journal_root)
    app.index = SearchIndex(settings.db_path)

    # Ensure knowledge directory exists (still file-based)
    settings.knowledge_dir.mkdir(parents=True, exist_ok=True)

    # Force schema init on both storage and index
    _ = app.storage.conn
    _ = app.index.conn
    await app.logger.info("Storage initialized", db_path=str(settings.db_path))

    # Initialize OAuth storage and register routes
    oauth_storage = OAuthStorage(settings.oauth_db_path)
    _ = oauth_storage.conn  # Force schema init
    expired = oauth_storage.cleanup_expired()
    if expired:
        await app.logger.info("OAuth cleanup", expired_tokens=expired)

    token_validator = register_oauth_routes(app, oauth_storage, settings)
    if token_validator:
        await app.logger.info("OAuth endpoints registered")

    # Initialize memory service
    configure_env()
    memory_service = await init_service(settings)
    if memory_service is None:
        raise RuntimeError(
            "Memory service failed to initialize — check mcp-memory-service is installed"
        )
    app.memory_service = memory_service
    await app.logger.info("Memory service initialized", db_path=str(settings.memory_db_path))

    # Create MCP server and mount on FastAPI
    app.mcp = create_mcp_server(app.storage, app.index, settings, memory_service=app.memory_service)
    mcp_http = app.mcp.streamable_http_app()
    authed_mcp = BearerAuthMiddleware(mcp_http, token_validator=token_validator)
    app.mount("/mcp", authed_mcp)

    # session_manager must be entered AFTER streamable_http_app()
    try:
        async with app.mcp.session_manager.run():
            yield
    finally:
        await app.logger.info("Server shutting down")
        if hasattr(app.memory_service, "close"):
            await app.memory_service.close()
        app.index.close()
        oauth_storage.close()


# Create FastAPI app
server = CustomFastAPI(
    title="journalctl",
    description="Personal journal MCP server",
    version="0.1.0",
    lifespan=lifespan,
    middleware=[Middleware(MCPPathNormalizer)],
)


# Global exception handler
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


# Health check (unprotected)
@server.get("/health")
async def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok", "service": "journalctl"}


def main() -> None:
    """Entry point for running the server."""
    settings = get_settings()

    if settings.transport == "stdio":
        configure_env()

        storage = DatabaseStorage(settings.db_path, settings.journal_root)
        idx = SearchIndex(settings.db_path)
        try:
            settings.knowledge_dir.mkdir(parents=True, exist_ok=True)
            _ = storage.conn
            _ = idx.conn
            mem_service = asyncio.run(init_service(settings))
            if mem_service is None:
                raise RuntimeError(
                    "Memory service failed to initialize — check mcp-memory-service is installed"
                )

            mcp = create_mcp_server(storage, idx, settings, memory_service=mem_service)
            mcp.run(transport="stdio")
        finally:
            idx.close()
            storage.close()
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
