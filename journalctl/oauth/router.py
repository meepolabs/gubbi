"""OAuth route registration for the FastAPI application.

A single function wires all OAuth endpoints onto the app,
keeping main.py:lifespan() clean.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from collections.abc import Callable
from typing import Any

from fastapi import FastAPI
from mcp.server.auth.routes import create_auth_routes, create_protected_resource_routes
from mcp.server.auth.settings import ClientRegistrationOptions, RevocationOptions
from pydantic import AnyHttpUrl
from starlette.requests import Request as StarletteRequest
from starlette.responses import JSONResponse
from starlette.routing import Route

from journalctl.config import OAUTH_AUTH_CODE_TTL_SECS, Settings
from journalctl.oauth.constants import REGISTER_MAX_ATTEMPTS, REGISTER_WINDOW_SECS
from journalctl.oauth.forms import client_ip, create_login_handler
from journalctl.oauth.provider import JournalOAuthProvider
from journalctl.oauth.storage import OAuthStorage

_logger = logging.getLogger("journalctl.oauth.router")


def _make_token_validator(oauth_storage: OAuthStorage) -> Callable[[str], bool]:
    """Build a closure that validates OAuth access tokens."""

    def validate(token: str) -> bool:
        try:
            at = oauth_storage.get_access_token(token)
            return at is not None and (at.expires_at is None or at.expires_at > int(time.time()))
        except (sqlite3.Error, ValueError, KeyError):
            _logger.warning("Token validation failed", exc_info=True)
            return False
        except Exception:
            _logger.exception("Unexpected error during token validation")
            return False

    return validate


def _wrap_register_rate_limit(
    route: Route,
    oauth_storage: OAuthStorage,
) -> None:
    """Wrap a Starlette Route's ASGI app to rate-limit by client IP.

    The SDK's /register route stores a CORSMiddleware ASGI app as route.app,
    so the endpoint cannot be called as a simple Starlette handler. We replace
    route.app with a new ASGI callable that checks rate limits then delegates
    to the original ASGI app.
    """
    original_app = route.app

    async def rate_limited_app(
        scope: dict[str, Any],
        receive: Any,
        send: Any,
    ) -> None:
        request = StarletteRequest(scope)
        ip = client_ip(request)
        event_key = f"register:{ip}"
        try:
            count = oauth_storage.count_rate_limit_events(event_key, REGISTER_WINDOW_SECS)
        except (sqlite3.Error, ValueError):
            # Fail safe -- reject on storage error rather than allow unrestricted registration
            response = JSONResponse(
                {"error": "rate_limit_unavailable"},
                status_code=503,
            )
            await response(scope, receive, send)
            return
        if count >= REGISTER_MAX_ATTEMPTS:
            response = JSONResponse(
                {"error": "rate_limit_exceeded"},
                status_code=429,
            )
            await response(scope, receive, send)
            return
        oauth_storage.record_rate_limit_event(event_key)
        await original_app(scope, receive, send)

    route.app = rate_limited_app  # type: ignore[assignment]


def register_oauth_routes(
    app: FastAPI,
    oauth_storage: OAuthStorage,
    settings: Settings,
) -> Callable[[str], bool] | None:
    """Register OAuth endpoints on the FastAPI app if configured.

    Dispatches across three deploy shapes:
    - Mode 1 (neither password_hash nor hydra): no routes, returns None.
    - Mode 2 (password_hash set): registers self-host OAuth + protected-resource.
    - Mode 3 (hydra_admin_url set): registers only a protected-resource endpoint
      pointing at the external Hydra issuer. Returns None because Hydra
      introspection middleware owns token validation.

    Returns a token_validator callable for BearerAuthMiddleware,
    or None if OAuth is disabled.
    """
    # Mode 3: Hydra-backed multi-tenant. Only advertise the resource
    # metadata route so MCP clients can discover the Hydra authorization
    # server. Token validation is handled by Hydra introspection middleware.
    if settings.hydra_admin_url:
        issuer_url = AnyHttpUrl(settings.hydra_public_issuer_url)
        pr_routes = create_protected_resource_routes(
            resource_url=AnyHttpUrl(f"{settings.server_url.rstrip('/')}/mcp"),
            authorization_servers=[issuer_url],
            scopes_supported=["journal", "offline_access", "openid", "email"],
            resource_documentation=AnyHttpUrl("https://docs.meepolabs.com/mcp"),
        )
        for route in pr_routes:
            app.routes.insert(0, route)
        _logger.info("Registered protected-resource routes (Mode 3: Hydra)")
        return None

    # Mode 1: no OAuth path configured.
    if not settings.password_hash:
        return None

    provider = JournalOAuthProvider(
        storage=oauth_storage,
        server_url=settings.server_url,
    )

    issuer_url = AnyHttpUrl(settings.server_url)

    # SDK auth routes: discovery, authorize, token, register, revoke
    auth_routes = create_auth_routes(
        provider=provider,
        issuer_url=issuer_url,
        client_registration_options=ClientRegistrationOptions(enabled=True),
        revocation_options=RevocationOptions(enabled=True),
    )
    for route in auth_routes:
        if isinstance(route, Route) and route.path == "/register":
            _wrap_register_rate_limit(route, oauth_storage)
        app.routes.insert(0, route)

    # Protected resource metadata
    pr_routes = create_protected_resource_routes(
        resource_url=AnyHttpUrl(f"{settings.server_url.rstrip('/')}/mcp"),
        authorization_servers=[issuer_url],
        scopes_supported=["journal", "offline_access", "openid", "email"],
        resource_documentation=AnyHttpUrl("https://docs.meepolabs.com/mcp"),
    )
    for route in pr_routes:
        app.routes.insert(0, route)

    # Custom login page
    login_handler = create_login_handler(
        storage=oauth_storage,
        password_hash=settings.password_hash,
        auth_code_ttl=OAUTH_AUTH_CODE_TTL_SECS,
        secure_cookies=settings.server_url.startswith("https"),
    )
    app.routes.insert(
        0,
        Route("/login", endpoint=login_handler, methods=["GET", "POST"]),
    )

    return _make_token_validator(oauth_storage)
