"""OAuth route registration for the FastAPI application.

A single function wires all OAuth endpoints onto the app,
keeping main.py:lifespan() clean.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Callable

from fastapi import FastAPI
from mcp.server.auth.routes import create_auth_routes, create_protected_resource_routes
from mcp.server.auth.settings import ClientRegistrationOptions, RevocationOptions
from pydantic import AnyHttpUrl
from starlette.routing import Route

from journalctl.config import Settings
from journalctl.oauth.forms import create_login_handler
from journalctl.oauth.provider import JournalOAuthProvider
from journalctl.oauth.storage import OAuthStorage


def _make_token_validator(oauth_storage: OAuthStorage) -> Callable[[str], bool]:
    """Build a closure that validates OAuth access tokens."""
    import logging

    _logger = logging.getLogger("journalctl.oauth.validator")

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


def register_oauth_routes(
    app: FastAPI,
    oauth_storage: OAuthStorage,
    settings: Settings,
) -> Callable[[str], bool] | None:
    """Register OAuth endpoints on the FastAPI app if configured.

    Returns a token_validator callable for BearerAuthMiddleware,
    or None if OAuth is disabled.
    """
    if not settings.owner_password_hash:
        return None

    provider = JournalOAuthProvider(
        storage=oauth_storage,
        server_url=settings.server_url,
        settings=settings,
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
        app.routes.insert(0, route)

    # Protected resource metadata
    pr_routes = create_protected_resource_routes(
        resource_url=AnyHttpUrl(f"{settings.server_url.rstrip('/')}/mcp"),
        authorization_servers=[issuer_url],
    )
    for route in pr_routes:
        app.routes.insert(0, route)

    # Custom login page
    login_handler = create_login_handler(
        storage=oauth_storage,
        owner_password_hash=settings.owner_password_hash,
        auth_code_ttl=settings.oauth_auth_code_ttl,
        secure_cookies=settings.server_url.startswith("https"),
    )
    app.routes.insert(
        0,
        Route("/login", endpoint=login_handler, methods=["GET", "POST"]),
    )

    return _make_token_validator(oauth_storage)
