"""Protected-resource metadata route registration."""

from __future__ import annotations

import logging
from collections.abc import Sequence

from fastapi import FastAPI
from mcp.server.auth.routes import create_protected_resource_routes
from pydantic import AnyHttpUrl

from journalctl.config import Settings

_logger = logging.getLogger("journalctl.oauth.wellknown")


def register(
    app: FastAPI,
    settings: Settings,
    authorization_servers: Sequence[AnyHttpUrl],
) -> None:
    """Register the /.well-known/oauth-protected-resource/mcp route."""
    pr_routes = create_protected_resource_routes(
        resource_url=AnyHttpUrl(f"{settings.server.url.rstrip('/')}/mcp"),
        authorization_servers=list(authorization_servers),
        scopes_supported=["journal", "offline_access", "openid", "email"],
        resource_documentation=AnyHttpUrl("https://docs.meepolabs.com/mcp"),
    )
    for route in pr_routes:
        app.routes.insert(0, route)
    _logger.info("Registered protected-resource routes")
