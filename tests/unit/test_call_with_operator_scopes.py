"""Unit tests for _call_with_operator scope plumbing in BearerAuthMiddleware.

Verifies that Mode 1 (static API key) and Mode 2 (self-host OAuth) produce
the correct current_token_scopes -- from config / from validator return value,
not the legacy ``frozenset({"journal"})`` literal.

Also verifies that a validator returning None (invalid token) produces 401.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from uuid import UUID

import httpx

from journalctl.core.auth_context import current_token_scopes, current_user_id
from journalctl.middleware.auth import BearerAuthMiddleware

TEST_API_KEY = "a" * 64
TEST_OP_ID = UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
TEST_DEFAULT_SCOPES = frozenset({"journal:read", "journal:write"})


def _asgi_app(
    captured_scopes: list[frozenset[str] | None],
    captured_user_id: list[UUID | None] | None = None,
) -> Callable[..., Any]:
    async def _app(scope: dict, receive: Any, send: Any) -> None:
        captured_scopes.append(current_token_scopes.get())
        if captured_user_id is not None:
            captured_user_id.append(current_user_id.get())
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    return _app


class TestMode1ApiKeyScopes:
    """Mode 1 (static API key) uses configured api_key_scopes."""

    async def test_default_scopes(self) -> None:
        """Default api_key_scopes -> frozenset({"journal:read","journal:write"})."""
        captured: list[frozenset[str] | None] = []
        mw = BearerAuthMiddleware(
            _asgi_app(captured),
            api_key=TEST_API_KEY,
            operator_user_id=TEST_OP_ID,
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=mw), base_url="http://test"
        ) as client:
            resp = await client.get("/", headers={"Authorization": f"Bearer {TEST_API_KEY}"})
        assert resp.status_code == 200
        assert captured == [TEST_DEFAULT_SCOPES]

    async def test_custom_scopes(self) -> None:
        """Custom api_key_scopes config -> custom frozenset."""
        captured: list[frozenset[str] | None] = []
        custom_scopes = frozenset({"journal:read"})
        mw = BearerAuthMiddleware(
            _asgi_app(captured),
            api_key=TEST_API_KEY,
            operator_user_id=TEST_OP_ID,
            api_key_scopes=custom_scopes,
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=mw), base_url="http://test"
        ) as client:
            resp = await client.get("/", headers={"Authorization": f"Bearer {TEST_API_KEY}"})
        assert resp.status_code == 200
        assert captured == [custom_scopes]

    async def test_scopes_reset_after_request(self) -> None:
        """ContextVar is reset to None after the request completes."""
        mw = BearerAuthMiddleware(
            _asgi_app([]),
            api_key=TEST_API_KEY,
            operator_user_id=TEST_OP_ID,
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=mw), base_url="http://test"
        ) as client:
            await client.get("/", headers={"Authorization": f"Bearer {TEST_API_KEY}"})
        assert current_token_scopes.get() is None
        assert current_user_id.get() is None


class TestMode2ValidatorScopes:
    """Mode 2 (self-host OAuth) uses validator's returned frozenset."""

    async def test_validator_returns_scopes(self) -> None:
        """Validator returns frozenset({"journal:read"}) -> middleware sets that."""
        captured: list[frozenset[str] | None] = []

        def validator(token: str) -> frozenset[str] | None:
            return frozenset({"journal:read"})

        mw = BearerAuthMiddleware(
            _asgi_app(captured),
            api_key=TEST_API_KEY,
            selfhost_token_validator=validator,
            operator_user_id=TEST_OP_ID,
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=mw), base_url="http://test"
        ) as client:
            resp = await client.get("/", headers={"Authorization": "Bearer some_token"})
        assert resp.status_code == 200
        assert captured == [frozenset({"journal:read"})]

    async def test_validator_returns_none_returns_401(self) -> None:
        """Validator returns None (invalid token) -> 401."""

        def validator(token: str) -> frozenset[str] | None:
            return None

        mw = BearerAuthMiddleware(
            _asgi_app([]),
            api_key=TEST_API_KEY,
            selfhost_token_validator=validator,
            operator_user_id=TEST_OP_ID,
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=mw), base_url="http://test"
        ) as client:
            resp = await client.get("/", headers={"Authorization": "Bearer bad_token"})
        assert resp.status_code == 401

    async def test_validator_not_configured_falls_through(self) -> None:
        """selfhost_token_validator=None -> falls through to 401."""
        mw = BearerAuthMiddleware(
            _asgi_app([]),
            api_key=TEST_API_KEY,
            selfhost_token_validator=None,
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=mw), base_url="http://test"
        ) as client:
            resp = await client.get("/", headers={"Authorization": "Bearer some_token"})
        assert resp.status_code == 401
