"""Bootstrap helper: trust-gateway bind-address safety check.

Lifted from gubbi.main lifespan as part of CO.44a (Phase 4 of the
code-organization review). Public symbol so tests + callers do not
import a private name from ``gubbi.main``.
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import structlog

__all__ = ["check_trust_gateway_bind_address"]


async def check_trust_gateway_bind_address(
    host: str,
    trust_gateway: bool,
    logger: structlog.stdlib.AsyncBoundLogger,
) -> None:
    """Fail fast when trust_gateway is paired with a public-routable bind address."""
    if not trust_gateway:
        return

    addresses: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []

    # Try parsing the bind address as a literal IP first.
    try:
        addresses.append(ipaddress.ip_address(host))
    except ValueError:
        # Hostname -- resolve before classifying.
        loop = asyncio.get_running_loop()
        try:
            resolved = await loop.run_in_executor(
                None,
                socket.getaddrinfo,
                host,
                None,
                socket.AF_UNSPEC,
                socket.SOCK_STREAM,
            )
        except socket.gaierror:
            raise RuntimeError(
                f"JOURNAL_TRUST_GATEWAY=true -- bind address '{host}' failed to resolve. "
                "Set JOURNAL_HOST to a resolvable address."
            ) from None

        for _family, _type, _proto, _canonname, sockaddr in resolved:
            addresses.append(ipaddress.ip_address(sockaddr[0]))

    has_unspecified = False
    for addr in addresses:
        if addr.is_unspecified:
            has_unspecified = True
            continue

        if addr.is_loopback or addr.is_private or addr.is_link_local:
            continue

        # Public-routable -- fail fast.
        raise RuntimeError(
            f"JOURNAL_TRUST_GATEWAY=true is incompatible with bind address "
            f"'{host}' -- set JOURNAL_HOST to a loopback or private-network address."
        )

    if has_unspecified:
        await logger.warning(
            "JOURNAL_TRUST_GATEWAY=true with bind address '%s' -- "
            "exposure depends on network/proxy layer",
            host,
        )
