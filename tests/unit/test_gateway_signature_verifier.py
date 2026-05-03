"""Unit tests for trust-gateway HMAC signature verification in BearerAuthMiddleware.

Covers:
- Legacy path (REQUIRE_SIGNATURE=false, no signature header)
- Signature required path with valid/invalid signatures
- Contract version mismatch
- Timestamp skew (stale, future)
- Tampered fields (user_id, method, path)
- Empty scopes -> legacy default
- Secret not configured -> 503
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

import httpx
from gubbi_common.auth.gateway_signature import build_signature

from journalctl.core.auth_context import current_token_scopes, current_user_id
from journalctl.middleware.auth import BearerAuthMiddleware

TEST_GATEWAY_SECRET = bytes.fromhex("a" * 64)  # 32 bytes / 64 hex chars
TEST_USER_UUID = UUID("11111111-2222-3333-4444-555555555555")


def _now_ts() -> str:
    """Return current UTC time as ISO 8601 Z-suffixed timestamp."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _asgi_app(
    *,
    capture_scopes: list[frozenset[str] | None] | None = None,
    capture_user_id: list[UUID | None] | None = None,
    response_status: int = 200,
) -> Any:
    """Return a minimal ASGI app that optionally captures context vars."""

    async def _app(scope: dict, receive: Any, send: Any) -> None:
        if capture_scopes is not None:
            capture_scopes.append(current_token_scopes.get())
        if capture_user_id is not None:
            capture_user_id.append(current_user_id.get())
        await send(
            {
                "type": "http.response.start",
                "status": response_status,
                "headers": [],
            }
        )
        await send({"type": "http.response.body", "body": b"ok"})

    return _app


def _make_sig(
    *,
    secret: bytes = TEST_GATEWAY_SECRET,
    user_id: str = str(TEST_USER_UUID),
    scopes: str = "journal:read journal:write",
    timestamp: str | None = None,
    method: str = "GET",
    path: str = "/mcp",
) -> str:
    if timestamp is None:
        timestamp = _now_ts()
    return build_signature(secret, user_id, scopes, timestamp, method, path)


def _headers(
    *,
    user_id: str | None = str(TEST_USER_UUID),
    scopes: str = "journal:read journal:write",
    timestamp: str | None = None,
    contract_version: str | None = None,
    signature: str | None = None,
    token_fp: str = "",
) -> dict[str, str]:
    if timestamp is None:
        timestamp = _now_ts()
    hdrs: dict[str, str] = {}
    if user_id is not None:
        hdrs["X-Auth-User-Id"] = user_id
    if scopes is not None:
        hdrs["X-Auth-Scopes"] = scopes
    if timestamp is not None:
        hdrs["X-Auth-Timestamp"] = timestamp
    # Only include contract-version when signature is provided, since the legacy
    # path doesn't need it and sending it would be inaccurate.
    if contract_version is not None:
        hdrs["X-Auth-Contract-Version"] = contract_version
    elif signature is not None:
        hdrs["X-Auth-Contract-Version"] = "1"
    if token_fp:
        hdrs["X-Auth-Token-Fp"] = token_fp
    if signature is not None:
        hdrs["X-Auth-Signature"] = signature
    return hdrs


# ---------------------------------------------------------------------------
# Legacy path: REQUIRE_SIGNATURE=false
# ---------------------------------------------------------------------------


class TestLegacyPath:
    """REQUIRE_SIGNATURE=false with no signature header -> legacy path."""

    async def test_no_sig_no_scopes_returns_200_with_default_scopes(self) -> None:
        """Legacy path: no signature, no scopes header -> default scopes applied."""
        captured_scopes: list[frozenset[str] | None] = []
        mw = BearerAuthMiddleware(
            _asgi_app(capture_scopes=captured_scopes),
            api_key="",
            trust_gateway=True,
            gateway_secret=None,
            gateway_require_signature=False,
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=mw), base_url="http://test"
        ) as client:
            resp = await client.get("/mcp", headers=_headers(scopes="", signature=None))
        assert resp.status_code == 200
        assert captured_scopes == [frozenset({"journal:read", "journal:write"})]

    async def test_no_sig_with_scopes_returns_200_with_parsed_scopes(self) -> None:
        """Legacy path: no signature, scopes present -> scopes parsed from header."""
        captured_scopes: list[frozenset[str] | None] = []
        mw = BearerAuthMiddleware(
            _asgi_app(capture_scopes=captured_scopes),
            api_key="",
            trust_gateway=True,
            gateway_secret=None,
            gateway_require_signature=False,
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=mw), base_url="http://test"
        ) as client:
            resp = await client.get(
                "/mcp",
                headers=_headers(scopes="journal:read", signature=None),
            )
        assert resp.status_code == 200
        assert captured_scopes == [frozenset({"journal:read"})]

    async def test_invalid_sig_triggers_verification_even_with_flag_false(self) -> None:
        """REQUIRE_SIGNATURE=false but signature present -> verification runs and fails."""
        mw = BearerAuthMiddleware(
            _asgi_app(),
            api_key="",
            trust_gateway=True,
            gateway_secret=TEST_GATEWAY_SECRET,
            gateway_require_signature=False,
        )
        bad_sig = "0" * 64
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=mw), base_url="http://test"
        ) as client:
            resp = await client.get(
                "/mcp",
                headers=_headers(signature=bad_sig),
            )
        assert resp.status_code == 401
        assert resp.json() == {"error": "Invalid gateway signature"}

    async def test_valid_sig_with_flag_false_verifies_and_returns_200(self) -> None:
        """REQUIRE_SIGNATURE=false but valid signature present -> verification passes."""
        captured_scopes: list[frozenset[str] | None] = []
        sig = _make_sig(scopes="journal:read")
        mw = BearerAuthMiddleware(
            _asgi_app(capture_scopes=captured_scopes),
            api_key="",
            trust_gateway=True,
            gateway_secret=TEST_GATEWAY_SECRET,
            gateway_require_signature=False,
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=mw), base_url="http://test"
        ) as client:
            resp = await client.get(
                "/mcp",
                headers=_headers(scopes="journal:read", signature=sig),
            )
        assert resp.status_code == 200
        assert captured_scopes == [frozenset({"journal:read"})]


# ---------------------------------------------------------------------------
# Signature required: REQUIRE_SIGNATURE=true
# ---------------------------------------------------------------------------


class TestSignatureRequired:
    """REQUIRE_SIGNATURE=true enforces signature on all requests."""

    async def test_no_signature_returns_401(self) -> None:
        """REQUIRE_SIGNATURE=true, no signature header -> 401."""
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
            resp = await client.get("/mcp", headers=_headers(signature=None))
        assert resp.status_code == 401

    async def test_valid_signature_returns_200_with_scopes(self) -> None:
        """REQUIRE_SIGNATURE=true, valid signature -> 200 with parsed scopes."""
        captured_scopes: list[frozenset[str] | None] = []
        sig = _make_sig(scopes="journal:read journal:write")
        mw = BearerAuthMiddleware(
            _asgi_app(capture_scopes=captured_scopes),
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
                headers=_headers(scopes="journal:read journal:write", signature=sig),
            )
        assert resp.status_code == 200
        assert captured_scopes == [frozenset({"journal:read", "journal:write"})]

    async def test_secret_not_configured_returns_503(self) -> None:
        """REQUIRE_SIGNATURE=true, secret=None on app.state -> 503."""
        mw = BearerAuthMiddleware(
            _asgi_app(),
            api_key="",
            trust_gateway=True,
            gateway_secret=None,  # not configured
            gateway_require_signature=True,
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=mw), base_url="http://test"
        ) as client:
            resp = await client.get("/mcp", headers=_headers(signature="ignored"))
        assert resp.status_code == 503
        assert resp.json() == {"error": "gateway secret not configured"}

    async def test_wrong_contract_version_returns_401(self) -> None:
        """Contract version != 1 -> 401."""
        sig = _make_sig()
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
                headers=_headers(signature=sig, contract_version="2"),
            )
        assert resp.status_code == 401
        assert resp.json() == {"error": "Unsupported X-Auth-Contract-Version"}


# ---------------------------------------------------------------------------
# Timestamp skew
# ---------------------------------------------------------------------------


class TestTimestampSkew:
    """Signature verification enforces ±30s skew window."""

    async def test_stale_timestamp_returns_401(self) -> None:
        """Timestamp > 30s in the past -> 401."""
        now = datetime.now(UTC)
        stale_ts = (now - timedelta(seconds=31)).strftime("%Y-%m-%dT%H:%M:%SZ")
        sig = _make_sig(timestamp=stale_ts)
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
                headers=_headers(signature=sig, timestamp=stale_ts),
            )
        assert resp.status_code == 401
        assert resp.json() == {"error": "Invalid gateway signature"}

    async def test_future_timestamp_returns_401(self) -> None:
        """Timestamp > 30s in the future -> 401."""
        now = datetime.now(UTC)
        future_ts = (now + timedelta(seconds=31)).strftime("%Y-%m-%dT%H:%M:%SZ")
        sig = _make_sig(timestamp=future_ts)
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
                headers=_headers(signature=sig, timestamp=future_ts),
            )
        assert resp.status_code == 401
        assert resp.json() == {"error": "Invalid gateway signature"}


# ---------------------------------------------------------------------------
# Tampered fields
# ---------------------------------------------------------------------------


class TestTamperedFields:
    """Signature verification rejects tampered identity or request metadata."""

    async def test_tampered_user_id_returns_401(self) -> None:
        """Signature was built for user A, request uses user B -> 401."""
        sig = _make_sig(user_id=str(TEST_USER_UUID))
        other_uuid = UUID("22222222-3333-4444-5555-666666666666")
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
                headers=_headers(
                    user_id=str(other_uuid),
                    signature=sig,
                ),
            )
        assert resp.status_code == 401
        assert resp.json() == {"error": "Invalid gateway signature"}

    async def test_tampered_method_returns_401(self) -> None:
        """Signature was built for GET, request uses POST -> 401."""
        sig = _make_sig(method="GET")
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
            resp = await client.post(
                "/mcp",
                headers=_headers(signature=sig),
            )
        assert resp.status_code == 401
        assert resp.json() == {"error": "Invalid gateway signature"}

    async def test_tampered_path_returns_401(self) -> None:
        """Signature was built for /mcp, request uses /other -> 401."""
        sig = _make_sig(path="/mcp")
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
                "/other",
                headers=_headers(signature=sig),
            )
        assert resp.status_code == 401
        assert resp.json() == {"error": "Invalid gateway signature"}


# ---------------------------------------------------------------------------
# Empty scopes
# ---------------------------------------------------------------------------


class TestEmptyScopes:
    """Empty X-Auth-Scopes header -> legacy default with warning."""

    async def test_empty_scopes_legacy_default_200(self) -> None:
        """Empty scopes header with valid sig -> 200, default scopes."""
        captured_scopes: list[frozenset[str] | None] = []
        sig = _make_sig(scopes="")
        mw = BearerAuthMiddleware(
            _asgi_app(capture_scopes=captured_scopes),
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
                headers=_headers(scopes="", signature=sig),
            )
        assert resp.status_code == 200
        assert captured_scopes == [frozenset({"journal:read", "journal:write"})]


# ---------------------------------------------------------------------------
# Context var reset
# ---------------------------------------------------------------------------


class TestContextVarReset:
    """Context vars are reset after gateway-authenticated requests."""

    async def test_contextvars_reset_after_request(self) -> None:
        """User_id and token_scopes reset after signed gateway request."""
        sig = _make_sig()
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
            await client.get("/mcp", headers=_headers(signature=sig))
        assert current_user_id.get() is None
        assert current_token_scopes.get() is None
