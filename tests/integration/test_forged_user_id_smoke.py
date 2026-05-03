"""CI smoke test: forged X-Auth-User-Id with no signature is rejected.

This test exists because of H-1 audit prescription: the trust-gateway path
must reject unauthenticated identity headers. When REQUIRE_SIGNATURE=true,
a request with a forged X-Auth-User-Id and NO signature headers must
return 401.

Constructs BearerAuthMiddleware directly (the ASGI app's lifespan does not
run in pytest), then sends a forged identity header to verify the middleware
rejects it.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

import httpx
import pytest

from journalctl.middleware.auth import BearerAuthMiddleware

VICTIM_UUID = UUID("99999999-8888-7777-6666-555555555555")
TEST_GATEWAY_SECRET = bytes.fromhex("b" * 64)  # 32 bytes


def _asgi_app() -> Any:
    """Minimal ASGI app that returns 200."""

    async def _app(scope: dict, receive: Any, send: Any) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    return _app


@pytest.mark.integration
async def test_forged_user_id_no_signature_returns_401() -> None:
    """Forged X-Auth-User-Id with no signature -> 401.

    Boots BearerAuthMiddleware with trust_gateway=True and
    gateway_require_signature=True, then sends a request with
    X-Auth-User-Id but NO X-Auth-Signature header.
    The middleware MUST reject the forged identity.
    """
    mw = BearerAuthMiddleware(
        _asgi_app(),
        api_key="",
        trust_gateway=True,
        gateway_secret=TEST_GATEWAY_SECRET,
        gateway_require_signature=True,
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=mw), base_url="http://test"
    ) as client:
        resp = await client.get(
            "/mcp",
            headers={
                "X-Auth-User-Id": str(VICTIM_UUID),
            },
        )
    assert resp.status_code == 401
