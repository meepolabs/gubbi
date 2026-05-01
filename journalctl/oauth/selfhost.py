"""Self-host OAuth route registration."""

from __future__ import annotations

import logging
import sqlite3
import time
from collections.abc import Callable
from typing import Any

from fastapi import FastAPI
from mcp.server.auth.routes import create_auth_routes
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
from journalctl.oauth.wellknown import register as register_wellknown

_logger = logging.getLogger("journalctl.oauth.selfhost")


def _make_token_validator(oauth_storage: OAuthStorage) -> Callable[[str], bool]:
    """Return a closure that validates OAuth access tokens."""

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
    """Wrap /register route with per-IP rate limiting."""
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


def register(
    app: FastAPI,
    storage: OAuthStorage,
    settings: Settings,
) -> Callable[[str], bool]:
    """Register self-host OAuth routes and return token validator."""
    provider = JournalOAuthProvider(
        storage=storage,
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
            _wrap_register_rate_limit(route, storage)
        app.routes.insert(0, route)

    register_wellknown(app, settings, authorization_servers=[issuer_url])

    login_handler = create_login_handler(
        storage=storage,
        password_hash=settings.password_hash,
        auth_code_ttl=OAUTH_AUTH_CODE_TTL_SECS,
        secure_cookies=settings.server_url.startswith("https"),
    )
    app.routes.insert(
        0,
        Route("/login", endpoint=login_handler, methods=["GET", "POST"]),
    )

    return _make_token_validator(storage)
