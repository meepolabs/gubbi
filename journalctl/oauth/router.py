"""OAuth route registration dispatcher."""

from __future__ import annotations

import logging
from collections.abc import Callable

from fastapi import FastAPI
from pydantic import AnyHttpUrl

from journalctl.config import Settings
from journalctl.oauth.disabled import register as register_disabled
from journalctl.oauth.selfhost import register as register_selfhost
from journalctl.oauth.storage import OAuthStorage
from journalctl.oauth.wellknown import register as register_wellknown

_logger = logging.getLogger("journalctl.oauth.router")


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
    if settings.auth.hydra_admin_url:
        issuer_url = AnyHttpUrl(settings.auth.hydra_public_issuer_url)
        register_wellknown(app, settings, authorization_servers=[issuer_url])
        _logger.info("Registered protected-resource routes (Mode 3: Hydra)")
        return None

    # Mode 1: no OAuth path configured.
    if not settings.auth.password_hash:
        register_disabled(app)
        return None

    # Mode 2: self-host OAuth.
    token_validator = register_selfhost(app, oauth_storage, settings)
    _logger.info("Registered self-host OAuth routes (Mode 2)")
    return token_validator
