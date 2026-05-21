"""Gateway HMAC secret decoder.

Decodes ``JOURNAL_GUBBI_GATEWAY_SECRET`` from hex and emits the same
warnings as the original lifespan body, verbatim.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import structlog.stdlib

__all__ = ["decode_gateway_secret"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def decode_gateway_secret(
    raw_hex: str | None,
    *,
    require_signature: bool,
    trust_gateway: bool,
    logger: structlog.stdlib.AsyncBoundLogger,
) -> bytes | None:
    """Decode the gateway HMAC secret and warn on misconfiguration.

    Mirrors the lifespan body verbatim (lines 341-369 of ``main.py``):

    * Decode hex → reject if < 32 bytes (warning logged).
    * Reject non-hex input (warning logged).
    * Flag ``require_signature`` without a usable secret.
    * Flag ``trust_gateway`` with ``gateway_require_signature`` disabled.
    """
    gateway_secret: bytes | None = None

    if raw_hex:
        try:
            decoded = bytes.fromhex(raw_hex)
            if len(decoded) >= 32:
                gateway_secret = decoded
            else:
                await logger.warning(
                    "JOURNAL_GUBBI_GATEWAY_SECRET decodes to less than 32 bytes "
                    "-- gateway signature verification disabled"
                )
        except ValueError:
            await logger.warning(
                "JOURNAL_GUBBI_GATEWAY_SECRET is not valid hex "
                "-- gateway signature verification disabled"
            )

    if require_signature and gateway_secret is None:
        await logger.warning(
            "JOURNAL_GATEWAY_REQUIRE_SIGNATURE=true but JOURNAL_GUBBI_GATEWAY_SECRET "
            "is missing or invalid -- signed requests will fail with 503"
        )

    if trust_gateway and not require_signature:
        await logger.warning(
            "trust_gateway=true but gateway_require_signature=false -- "
            "requests are accepted without HMAC verification"
        )

    return gateway_secret
