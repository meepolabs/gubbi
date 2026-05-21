"""Protected-resource metadata route registration."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from urllib.parse import urlparse

from fastapi import FastAPI
from mcp.server.auth.routes import create_protected_resource_routes
from pydantic import AnyHttpUrl

from gubbi.config import Settings

# Stays on stdlib ``logging`` because ``register`` runs sync at lifespan
# setup. ``structlog.AsyncBoundLogger`` emits return coroutines that must
# be awaited; sync callers cannot use it.
logger = logging.getLogger(__name__)

# URL prefixes the metadata document may legitimately emit. https is the
# security target; localhost loopbacks are admitted because the test
# fixture (``TEST_SERVER_URL = "http://localhost:8100"``) and any local
# dev box round-trip those URLs through the document. The deployed Mode-3
# validator below uses a STRICTER scheme-only check (https, no loopback)
# because hosted production never legitimately points ``authorization_servers``
# at a loopback address -- doing so would let a deploy mistake or
# compromised config route credential traffic through a local interceptor.
TLS_OR_LOOPBACK_PREFIXES: tuple[str, ...] = (
    "https://",
    "http://localhost",
    "http://127.0.0.1",
    "http://[::1]",
)

# Public documentation URL emitted in /.well-known/oauth-protected-resource/mcp
# under ``resource_documentation``. Module-level so the TLS validator can audit
# it alongside ``authorization_servers``, and so tests can patch it without
# threading an extra parameter through ``register``.
RESOURCE_DOCUMENTATION_URL: str = "https://gubbi.ai/docs/mcp"


def _is_https(url: str) -> bool:
    """Return True iff ``url`` parses with the ``https`` scheme.

    Uses :func:`urllib.parse.urlparse` rather than ``str.startswith`` so the
    check is robust against Pydantic's ``AnyHttpUrl`` normalisation
    (which appends a trailing slash to bare origins) and against any
    future scheme-prefix coincidence (e.g. a hostname literally beginning
    with ``https``-like text).
    """
    return urlparse(url).scheme == "https"


def _validate_tls_for_deployed_hosted(
    settings: Settings,
    authorization_servers: Sequence[AnyHttpUrl],
    resource_documentation_url: str,
) -> None:
    """Reject non-TLS credential-bearing URLs in deployed Mode-3 hosted setups.

    Defence-in-depth check that runs at lifespan setup, IFF both:

    - ``settings.is_deployed`` is True (``app_env`` in ``staging``/``production``)
    - ``settings.auth.hydra_admin_url`` is non-empty (Mode-3 hosted multi-tenant)

    Inside the gate we require **https only** -- no loopback bypass.
    Legitimate hosted Mode-3 deployments never run Hydra on loopback
    (Hydra is a separate host or sidecar bound to a non-loopback
    interface), so loopback inside this gate is pure attack surface --
    a deploy mistake or compromised config could pin ``authorization_servers``
    at ``http://localhost`` and route the client through a local
    interceptor.

    Self-host Modes 1 (API-key only) and 2 (full self-host with password)
    bypass the check unconditionally -- those operators may legitimately
    run on a LAN URL without TLS, and breaking their deploy on startup
    would be a regression. Non-deployed envs (``dev``, ``ci``,
    testbench) likewise bypass: testbench drives Hydra at
    ``http://localhost:4444`` and any local-Hydra dev box does the same.

    Raises:
        ValueError: When a credential-bearing URL fails the https check
            while the validator is active. The message names the
            offending URL, why it's rejected, and how to fix.
    """
    if not (settings.is_deployed and settings.auth.hydra_admin_url):
        return

    for idx, server in enumerate(authorization_servers):
        server_str = str(server)
        if not _is_https(server_str):
            raise ValueError(
                f"authorization_servers[{idx}]={server_str!r} is non-https "
                "in deployed Mode-3 hosted. https:// is required -- a non-TLS "
                "or loopback authorization server in a hosted Mode-3 deploy "
                "is a downgrade vector (credentials discovered from this "
                "metadata document would transit cleartext, or be routed "
                "through a local interceptor in the loopback case). "
                "Loopback is intentionally rejected here even though it is "
                "kernel-loopback safe -- legitimate hosted Mode-3 never "
                "points authorization_servers at a loopback address."
            )

    if not _is_https(resource_documentation_url):
        raise ValueError(
            f"resource_documentation={resource_documentation_url!r} is "
            "non-https in deployed Mode-3 hosted. https:// is required -- a "
            "non-TLS doc URL could host malicious content under a "
            "phishing-friendly hostname that clients reach from the "
            "published metadata."
        )


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
        resource_documentation=AnyHttpUrl(RESOURCE_DOCUMENTATION_URL),
    )
    _validate_tls_for_deployed_hosted(settings, authorization_servers, RESOURCE_DOCUMENTATION_URL)
    for route in pr_routes:
        app.routes.insert(0, route)
    logger.info("Registered protected-resource routes")
