"""Journal MCP Server — FastAPI application entry point.

Serves the MCP protocol over streamable HTTP (production) or
stdio (local development). Based on fastapi_template
patterns: CustomFastAPI subclass, lifespan, structlog.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from mcp.server.fastmcp import FastMCP
from starlette.middleware import Middleware
from starlette.types import ASGIApp, Receive, Scope, Send

from journalctl.auth import BearerAuthMiddleware
from journalctl.config import Settings, get_settings
from journalctl.core.logger import initialize_logger
from journalctl.import_tools import import_tools
from journalctl.oauth.setup import register_oauth_routes
from journalctl.oauth.storage import OAuthStorage
from journalctl.storage.index import SearchIndex
from journalctl.storage.markdown import MarkdownStorage


class MCPPathNormalizer:
    """Normalize /mcp → /mcp/ at the ASGI level.

    Starlette's Mount("/mcp") returns a 307 redirect for requests to
    /mcp (without trailing slash).  HTTP clients following the 307
    often change POST→GET, which breaks the MCP streamable HTTP
    transport — the initialize POST arrives as a GET and opens an
    SSE stream instead.

    This raw ASGI middleware rewrites the path before it reaches the
    router, avoiding the redirect entirely.  Unlike BaseHTTPMiddleware,
    it does NOT buffer responses, so SSE streaming works correctly.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope["path"] == "/mcp":
            scope = dict(scope)
            scope["path"] = "/mcp/"
        await self.app(scope, receive, send)


class CustomFastAPI(FastAPI):
    """Extended FastAPI with journal-specific attributes."""

    logger: structlog.BoundLogger
    storage: MarkdownStorage
    index: SearchIndex
    settings: Settings
    mcp: FastMCP


def create_mcp_server(
    storage: MarkdownStorage,
    index: SearchIndex,
    settings: Settings,
) -> FastMCP:
    """Create and configure the MCP server with all tools."""
    mcp = FastMCP(
        "journalctl",
        stateless_http=True,
        streamable_http_path="/",
        host="0.0.0.0",  # noqa: S104 — accept any Host header behind reverse proxy
    )
    import_tools(mcp, storage, index, settings)
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
    app.storage = MarkdownStorage(settings.journal_root)
    app.index = SearchIndex(settings.db_path, settings.journal_root)

    # Ensure directories exist
    settings.topics_dir.mkdir(parents=True, exist_ok=True)
    settings.conversations_dir.mkdir(parents=True, exist_ok=True)
    settings.knowledge_dir.mkdir(parents=True, exist_ok=True)

    # Incremental index rebuild on startup
    count = app.index.incremental_rebuild()
    await app.logger.info(
        "Index rebuild complete",
        files_updated=count,
    )

    # Initialize OAuth storage and register routes
    oauth_storage = OAuthStorage(settings.oauth_db_path)
    _ = oauth_storage.conn  # Force schema init
    expired = oauth_storage.cleanup_expired()
    if expired:
        await app.logger.info("OAuth cleanup", expired_tokens=expired)

    token_validator = register_oauth_routes(app, oauth_storage, settings)
    if token_validator:
        await app.logger.info("OAuth endpoints registered")

    # Create MCP server and mount on FastAPI
    app.mcp = create_mcp_server(app.storage, app.index, settings)
    mcp_http = app.mcp.streamable_http_app()
    authed_mcp = BearerAuthMiddleware(mcp_http, token_validator=token_validator)
    app.mount("/mcp", authed_mcp)

    # session_manager must be entered AFTER streamable_http_app()
    try:
        async with app.mcp.session_manager.run():
            yield
    finally:
        await app.logger.info("Server shutting down")
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
        # stdio mode: run MCP directly without FastAPI
        storage = MarkdownStorage(settings.journal_root)
        idx = SearchIndex(settings.db_path, settings.journal_root)

        settings.topics_dir.mkdir(parents=True, exist_ok=True)
        settings.conversations_dir.mkdir(parents=True, exist_ok=True)
        settings.knowledge_dir.mkdir(parents=True, exist_ok=True)

        idx.incremental_rebuild()
        mcp = create_mcp_server(storage, idx, settings)
        mcp.run(transport="stdio")
    else:
        import uvicorn

        uvicorn.run(
            "journalctl.main:server",
            host=settings.host,
            port=settings.port,
            reload=False,
        )


if __name__ == "__main__":
    main()
