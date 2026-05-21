"""OAuth storage and route wiring.

Constructs the SQLite-backed ``OAuthStorage``, registers OAuth routes on
the FastAPI app, and runs a one-shot expired-token cleanup.  Returns the
storage handle so that the lifespan's ``finally:`` block can close it.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from gubbi.oauth.router import register_oauth_routes
from gubbi.oauth.storage import OAuthStorage

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from fastapi import FastAPI

    from gubbi.config import Settings

logger = logging.getLogger(__name__)


async def setup_oauth(
    app: FastAPI,
    settings: Settings,
) -> tuple[OAuthStorage, Callable[[str], Awaitable[frozenset[str] | None]] | None]:
    """Set up OAuth storage and routes.

    Performs the three steps from the lifespan body (lines 331-339):

    1. Create ``OAuthStorage`` backed by ``settings.oauth_db_path``.
    2. Run ``cleanup_expired()`` and log if any rows were removed.
    3. Register OAuth routes; return ``(storage, token_validator)``.

    Does NOT close the storage -- that stays in the lifespan's ``finally:``
    block so the caller retains control over resource lifecycle.
    """
    oauth_storage = OAuthStorage(settings.oauth_db_path)
    await oauth_storage.initialize()

    expired = await oauth_storage.cleanup_expired()
    if expired > 0:
        logger.info("OAuth cleanup expired_tokens=%d", expired)

    token_validator = await register_oauth_routes(app, oauth_storage, settings)
    if token_validator:
        logger.info("OAuth endpoints registered")

    return oauth_storage, token_validator
