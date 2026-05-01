"""Unit tests for BearerAuthMiddleware with Hydra introspection."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import httpx
import pytest

from journalctl.auth.hydra import (
    HydraIntrospector,
    HydraInvalidToken,
    HydraUnreachable,
    TokenClaims,
)
from journalctl.core.auth_context import current_user_id
from journalctl.middleware.auth import BearerAuthMiddleware

TEST_API_KEY = "a" * 64  # 64-char key
TEST_TOKEN = "ory_at_" + "x" * 80
TEST_SUB = UUID("550e8400-e29b-41d4-a716-446655440000")
TEST_OP_ID = UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")


def _asgi_app(
    *,
    response_status: int = 200,
    response_body: bytes = b"ok",
    capture_app: Callable[..., Awaitable[None]] | None = None,
) -> Callable[..., Awaitable[None]]:
    """Return a minimal ASGI app that responds with the given status/body."""

    async def _app(scope: dict, receive: Any, send: Any) -> None:
        if capture_app is not None:
            await capture_app(scope, receive, send)
            return
        await send(
            {
                "type": "http.response.start",
                "status": response_status,
                "headers": [],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": response_body,
            }
        )

    return _app


def _scope(
    method: str = "GET",
    path: str = "/",
    auth_header: str | None = None,
    scope_type: str = "http",
) -> dict[str, Any]:
    headers: list[tuple[bytes, bytes]] = []
    if auth_header is not None:
        headers.append((b"authorization", auth_header.encode()))
    return {
        "type": scope_type,
        "asgi": {"version": "3.0"},
        "method": method,
        "path": path,
        "headers": headers,
    }


@pytest.fixture
def transport_app() -> tuple[Any, AsyncMock]:
    """Create an ASGI middleware under test with a mocked introspector."""
    claims = TokenClaims(sub=TEST_SUB, scope="openid journal email", exp=9999999999)
    mock_iv = AsyncMock(spec=HydraIntrospector)
    mock_iv.introspect = AsyncMock(return_value=claims)
    app = BearerAuthMiddleware(
        _asgi_app(),
        api_key=TEST_API_KEY,
        introspector=mock_iv,
        required_scope="journal",
    )
    return app, mock_iv


class TestAPIMode:
    async def test_api_key_match_returns_200(self) -> None:
        downstream = _asgi_app()
        mw = BearerAuthMiddleware(downstream, api_key=TEST_API_KEY, operator_user_id=TEST_OP_ID)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=mw), base_url="http://test"
        ) as client:
            resp = await client.get("/", headers={"Authorization": f"Bearer {TEST_API_KEY}"})
        assert resp.status_code == 200

    async def test_wrong_api_key_returns_401(self) -> None:
        downstream = _asgi_app()
        mw = BearerAuthMiddleware(downstream, api_key=TEST_API_KEY)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=mw), base_url="http://test"
        ) as client:
            resp = await client.get("/", headers={"Authorization": "Bearer wrong_key"})
        assert resp.status_code == 401
        assert "error" in resp.json()

    async def test_api_key_no_operator_returns_503(self) -> None:
        """When operator_user_id is None, API key match returns 503 with provisioning hint."""
        downstream = _asgi_app()
        mw = BearerAuthMiddleware(downstream, api_key=TEST_API_KEY, operator_user_id=None)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=mw), base_url="http://test"
        ) as client:
            resp = await client.get("/", headers={"Authorization": f"Bearer {TEST_API_KEY}"})
        assert resp.status_code == 503
        assert "auto-scaffold" in resp.json()["error"]


class TestHydraMode:
    async def test_ory_token_valid_returns_200(self, transport_app: tuple[Any, AsyncMock]) -> None:
        app, mock_iv = transport_app
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/", headers={"Authorization": f"Bearer {TEST_TOKEN}"})
        assert resp.status_code == 200
        mock_iv.introspect.assert_called_once_with(TEST_TOKEN)

    async def test_ory_token_contextvar_set_during_request(self) -> None:
        claims = TokenClaims(sub=TEST_SUB, scope="openid journal email", exp=9999999999)
        mock_iv = AsyncMock(spec=HydraIntrospector)
        mock_iv.introspect = AsyncMock(return_value=claims)

        captured: list[str | None] = []

        async def capture(scope, receive, send):
            captured.append(str(current_user_id.get()))
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [],
                }
            )
            await send({"type": "http.response.body", "body": b"ok"})

        mw = BearerAuthMiddleware(
            capture,
            api_key=TEST_API_KEY,
            introspector=mock_iv,
            required_scope="journal",
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=mw), base_url="http://test"
        ) as client:
            await client.get("/", headers={"Authorization": f"Bearer {TEST_TOKEN}"})
        assert captured == [str(TEST_SUB)]

    async def test_ory_token_contextvar_reset_after(self) -> None:
        claims = TokenClaims(sub=TEST_SUB, scope="journal", exp=9999999999)
        mock_iv = AsyncMock(spec=HydraIntrospector)
        mock_iv.introspect = AsyncMock(return_value=claims)
        mw = BearerAuthMiddleware(
            _asgi_app(),
            api_key=TEST_API_KEY,
            introspector=mock_iv,
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=mw), base_url="http://test"
        ) as client:
            await client.get("/", headers={"Authorization": f"Bearer {TEST_TOKEN}"})
        assert current_user_id.get() is None

    async def test_introspector_none_rejects_ory_token(self) -> None:
        mw = BearerAuthMiddleware(
            _asgi_app(),
            api_key=TEST_API_KEY,
            introspector=None,
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=mw), base_url="http://test"
        ) as client:
            resp = await client.get("/", headers={"Authorization": f"Bearer {TEST_TOKEN}"})
        assert resp.status_code == 401

    async def test_hydra_unreachable_returns_503(
        self, transport_app: tuple[Any, AsyncMock]
    ) -> None:
        app, mock_iv = transport_app
        mock_iv.introspect = AsyncMock(side_effect=HydraUnreachable("timeout"))
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/", headers={"Authorization": f"Bearer {TEST_TOKEN}"})
        assert resp.status_code == 503
        assert "error" in resp.json()
        assert resp.headers.get("retry-after") == "5"

    async def test_hydra_invalid_token_returns_401(
        self, transport_app: tuple[Any, AsyncMock]
    ) -> None:
        app, mock_iv = transport_app
        mock_iv.introspect = AsyncMock(side_effect=HydraInvalidToken("test"))
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/", headers={"Authorization": f"Bearer {TEST_TOKEN}"})
        assert resp.status_code == 401


class TestScopeCheck:
    async def test_missing_required_scope_returns_403(self) -> None:
        """Scope 'openid email' should NOT satisfy 'journal' requirement."""
        claims = TokenClaims(sub=TEST_SUB, scope="openid email", exp=9999999999)
        mock_iv = AsyncMock(spec=HydraIntrospector)
        mock_iv.introspect = AsyncMock(return_value=claims)
        mw = BearerAuthMiddleware(
            _asgi_app(),
            api_key=TEST_API_KEY,
            introspector=mock_iv,
            required_scope="journal",
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=mw), base_url="http://test"
        ) as client:
            resp = await client.get("/", headers={"Authorization": f"Bearer {TEST_TOKEN}"})
        assert resp.status_code == 403

    async def test_scope_contains_required_returns_200(self) -> None:
        """Scope 'openid journal email' satisfies 'journal' requirement."""
        claims = TokenClaims(sub=TEST_SUB, scope="openid journal email", exp=9999999999)
        mock_iv = AsyncMock(spec=HydraIntrospector)
        mock_iv.introspect = AsyncMock(return_value=claims)
        mw = BearerAuthMiddleware(
            _asgi_app(),
            api_key=TEST_API_KEY,
            introspector=mock_iv,
            required_scope="journal",
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=mw), base_url="http://test"
        ) as client:
            resp = await client.get("/", headers={"Authorization": f"Bearer {TEST_TOKEN}"})
        assert resp.status_code == 200

    async def test_scope_substring_rejected(self) -> None:
        """'journaling' must NOT satisfy strict 'journal' scope check."""
        claims = TokenClaims(sub=TEST_SUB, scope="journaling read", exp=9999999999)
        mock_iv = AsyncMock(spec=HydraIntrospector)
        mock_iv.introspect = AsyncMock(return_value=claims)
        mw = BearerAuthMiddleware(
            _asgi_app(),
            api_key=TEST_API_KEY,
            introspector=mock_iv,
            required_scope="journal",
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=mw), base_url="http://test"
        ) as client:
            resp = await client.get("/", headers={"Authorization": f"Bearer {TEST_TOKEN}"})
        assert resp.status_code == 403


class TestMissingAndOversizedTokens:
    async def test_missing_authorization_returns_401(self) -> None:
        mw = BearerAuthMiddleware(_asgi_app(), api_key=TEST_API_KEY)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=mw), base_url="http://test"
        ) as client:
            resp = await client.get("/")
        assert resp.status_code == 401

    async def test_empty_authorization_returns_401(self) -> None:
        mw = BearerAuthMiddleware(_asgi_app(), api_key=TEST_API_KEY)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=mw), base_url="http://test"
        ) as client:
            resp = await client.get("/", headers={"Authorization": ""})
        assert resp.status_code == 401

    async def test_oversized_token_returns_401(self) -> None:
        mw = BearerAuthMiddleware(_asgi_app(), api_key=TEST_API_KEY)
        fake_token = "O" * 300
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=mw), base_url="http://test"
        ) as client:
            resp = await client.get("/", headers={"Authorization": f"Bearer {fake_token}"})
        assert resp.status_code == 401


class TestSelfhostValidator:
    async def test_selfhost_validator_rejects(self) -> None:
        mock_validator = MagicMock(return_value=False)
        mw = BearerAuthMiddleware(
            _asgi_app(),
            api_key=TEST_API_KEY,
            selfhost_token_validator=mock_validator,
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=mw), base_url="http://test"
        ) as client:
            resp = await client.get("/", headers={"Authorization": "Bearer some_token"})
        assert resp.status_code == 401
        mock_validator.assert_called_once()

    async def test_selfhost_validator_accepts(self) -> None:
        mock_validator = MagicMock(return_value=True)
        mw = BearerAuthMiddleware(
            _asgi_app(),
            api_key=TEST_API_KEY,
            selfhost_token_validator=mock_validator,
            operator_user_id=TEST_OP_ID,
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=mw), base_url="http://test"
        ) as client:
            resp = await client.get("/", headers={"Authorization": "Bearer some_token"})
        assert resp.status_code == 200

    async def test_ory_token_no_introspector_falls_to_selfhost_validator(self) -> None:
        """Ory token with no introspector falls through to self-host validator check."""
        mock_validator = MagicMock(return_value=False)
        mw = BearerAuthMiddleware(
            _asgi_app(),
            api_key=TEST_API_KEY,
            introspector=None,
            selfhost_token_validator=mock_validator,
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=mw), base_url="http://test"
        ) as client:
            resp = await client.get("/", headers={"Authorization": f"Bearer {TEST_TOKEN}"})
        assert resp.status_code == 401
        # Self-host validator is called because introspector is None
        mock_validator.assert_called_once()

    async def test_ory_token_with_none_validator_returns_401(self) -> None:
        """Ory token without introspector and selfhost_validator=None -> 401."""
        mw = BearerAuthMiddleware(
            _asgi_app(),
            api_key=TEST_API_KEY,
            introspector=None,
            selfhost_token_validator=None,
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=mw), base_url="http://test"
        ) as client:
            resp = await client.get("/", headers={"Authorization": f"Bearer {TEST_TOKEN}"})
        assert resp.status_code == 401


class TestContextvarReset:
    async def test_contextvar_reset_on_downstream_exception(self) -> None:
        """If downstream app raises, contextvar is reset before the exception propagates."""
        call_order: list[str] = []

        async def crashing_app(scope, receive, send):
            call_order.append("entering")
            raise RuntimeError("downstream crash")

        mw = BearerAuthMiddleware(
            crashing_app,
            api_key=TEST_API_KEY,
            operator_user_id=TEST_OP_ID,
        )
        scope = _scope(auth_header=f"Bearer {TEST_API_KEY}")

        async def receive() -> dict:
            return {"type": "http.request", "body": b""}

        messages: list[dict] = []

        async def send(msg: dict) -> None:
            messages.append(msg)

        with pytest.raises(RuntimeError, match="downstream crash"):
            await mw(scope, receive, send)

        # Contextvar is reset after downstream exception
        assert current_user_id.get() is None

    async def test_contextvar_reset_when_hydra_downstream_crashes(self) -> None:
        """Hydra path: contextvar is set, downstream raises, finally resets it."""
        claims = TokenClaims(sub=TEST_SUB, scope="journal", exp=9999999999)
        mock_iv = AsyncMock(spec=HydraIntrospector)
        mock_iv.introspect = AsyncMock(return_value=claims)

        seen_during_request: list[UUID | None] = []

        async def crashing_app(scope: dict, receive: Any, send: Any) -> None:
            seen_during_request.append(current_user_id.get())
            raise RuntimeError("downstream crash after auth")

        mw = BearerAuthMiddleware(
            crashing_app,
            api_key=TEST_API_KEY,
            introspector=mock_iv,
            required_scope="journal",
        )
        scope = _scope(auth_header=f"Bearer {TEST_TOKEN}")

        async def receive() -> dict:
            return {"type": "http.request", "body": b""}

        async def send(msg: dict) -> None:
            pass

        assert current_user_id.get() is None  # sanity: clean before
        with pytest.raises(RuntimeError, match="downstream crash after auth"):
            await mw(scope, receive, send)
        assert seen_during_request == [TEST_SUB]  # proves set happened
        assert current_user_id.get() is None  # proves finally reset happened


class TestNonHttpScope:
    async def test_non_http_scope_passes_through(self) -> None:
        """Non-http scopes (websocket etc.) bypass all auth logic."""
        got_scope: list[dict] = []

        async def passthrough(scope, receive, send):
            got_scope.append(scope)

        mw = BearerAuthMiddleware(
            passthrough,
            api_key=TEST_API_KEY,
        )
        scope = _scope(scope_type="websocket")

        async def receive() -> dict:
            return {}

        await mw(scope, receive, MagicMock())
        assert len(got_scope) == 1
        assert got_scope[0]["type"] == "websocket"


class TestWWWAuthenticateHeader:
    """RFC 6750 + MCP spec 2025-11-25 compliance for WWW-Authenticate.

    A protected MCP resource must advertise its OAuth protected-resource
    metadata URL on 401/403 so clients can discover the authorization
    server without baked-in knowledge of the deployment.
    """

    PRM_URL = "https://api.journalctl.app/.well-known/oauth-protected-resource"

    async def test_missing_auth_includes_resource_metadata_when_configured(self) -> None:
        mw = BearerAuthMiddleware(
            _asgi_app(),
            api_key=TEST_API_KEY,
            protected_resource_metadata_url=self.PRM_URL,
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=mw), base_url="http://test"
        ) as client:
            resp = await client.get("/")
        assert resp.status_code == 401
        challenge = resp.headers["www-authenticate"]
        assert challenge.startswith("Bearer ")
        assert 'error="invalid_token"' in challenge
        assert f'resource_metadata="{self.PRM_URL}"' in challenge

    async def test_invalid_token_includes_resource_metadata(self) -> None:
        mw = BearerAuthMiddleware(
            _asgi_app(),
            api_key=TEST_API_KEY,
            protected_resource_metadata_url=self.PRM_URL,
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=mw), base_url="http://test"
        ) as client:
            resp = await client.get("/", headers={"Authorization": "Bearer wrong_key"})
        assert resp.status_code == 401
        assert f'resource_metadata="{self.PRM_URL}"' in resp.headers["www-authenticate"]

    async def test_insufficient_scope_includes_resource_metadata(self) -> None:
        claims = TokenClaims(sub=TEST_SUB, scope="openid email", exp=9999999999)
        mock_iv = AsyncMock(spec=HydraIntrospector)
        mock_iv.introspect = AsyncMock(return_value=claims)
        mw = BearerAuthMiddleware(
            _asgi_app(),
            api_key=TEST_API_KEY,
            introspector=mock_iv,
            required_scope="journal",
            protected_resource_metadata_url=self.PRM_URL,
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=mw), base_url="http://test"
        ) as client:
            resp = await client.get("/", headers={"Authorization": f"Bearer {TEST_TOKEN}"})
        assert resp.status_code == 403
        challenge = resp.headers["www-authenticate"]
        assert 'error="insufficient_scope"' in challenge
        assert 'required_scope="journal"' in challenge
        assert f'resource_metadata="{self.PRM_URL}"' in challenge

    async def test_no_url_configured_omits_resource_metadata(self) -> None:
        """Mode 1 (API-key only, no OAuth) deployments get a bare challenge."""
        mw = BearerAuthMiddleware(_asgi_app(), api_key=TEST_API_KEY)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=mw), base_url="http://test"
        ) as client:
            resp = await client.get("/")
        assert resp.status_code == 401
        challenge = resp.headers["www-authenticate"]
        assert 'error="invalid_token"' in challenge
        assert "resource_metadata=" not in challenge

    async def test_challenge_grammar_exact_for_401(self) -> None:
        """Full-string assertion pins RFC 6750 challenge grammar + param order."""
        mw = BearerAuthMiddleware(
            _asgi_app(),
            api_key=TEST_API_KEY,
            protected_resource_metadata_url=self.PRM_URL,
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=mw), base_url="http://test"
        ) as client:
            resp = await client.get("/")
        assert resp.status_code == 401
        expected = f'Bearer error="invalid_token", resource_metadata="{self.PRM_URL}"'
        assert resp.headers["www-authenticate"] == expected

    async def test_challenge_grammar_exact_for_403(self) -> None:
        """Full-string assertion pins order: error, required_scope, resource_metadata."""
        claims = TokenClaims(sub=TEST_SUB, scope="openid email", exp=9999999999)
        mock_iv = AsyncMock(spec=HydraIntrospector)
        mock_iv.introspect = AsyncMock(return_value=claims)
        mw = BearerAuthMiddleware(
            _asgi_app(),
            api_key=TEST_API_KEY,
            introspector=mock_iv,
            required_scope="journal",
            protected_resource_metadata_url=self.PRM_URL,
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=mw), base_url="http://test"
        ) as client:
            resp = await client.get("/", headers={"Authorization": f"Bearer {TEST_TOKEN}"})
        assert resp.status_code == 403
        expected = (
            'Bearer error="insufficient_scope", required_scope="journal", '
            f'resource_metadata="{self.PRM_URL}"'
        )
        assert resp.headers["www-authenticate"] == expected


class TestTrustGateway:
    """JOURNAL_TRUST_GATEWAY mode: skip all auth and trust X-Auth-User-Id header."""

    TEST_USER_UUID = UUID("11111111-2222-3333-4444-555555555555")

    async def test_valid_user_id_passes_through(self) -> None:
        """Valid X-Auth-User-Id UUID header -> request reaches downstream app."""
        captured: list[str | None] = []

        async def capture(scope, receive, send):
            captured.append(str(current_user_id.get()))
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [],
                }
            )
            await send({"type": "http.response.body", "body": b"ok"})

        mw = BearerAuthMiddleware(capture, api_key="", trust_gateway=True)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=mw), base_url="http://test"
        ) as client:
            resp = await client.get("/", headers={"X-Auth-User-Id": str(self.TEST_USER_UUID)})
        assert resp.status_code == 200
        assert captured == [str(self.TEST_USER_UUID)]

    async def test_missing_header_returns_401(self) -> None:
        """Missing X-Auth-User-Id header -> 401."""
        mw = BearerAuthMiddleware(_asgi_app(), api_key="", trust_gateway=True)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=mw), base_url="http://test"
        ) as client:
            resp = await client.get("/")
        assert resp.status_code == 401
        assert resp.json() == {"error": "Missing X-Auth-User-Id header"}

    async def test_empty_header_returns_401(self) -> None:
        """Empty X-Auth-User-Id header -> 401."""
        mw = BearerAuthMiddleware(_asgi_app(), api_key="", trust_gateway=True)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=mw), base_url="http://test"
        ) as client:
            resp = await client.get("/", headers={"X-Auth-User-Id": ""})
        assert resp.status_code == 401
        assert resp.json() == {"error": "Missing X-Auth-User-Id header"}

    async def test_malformed_uuid_returns_401(self) -> None:
        """Malformed X-Auth-User-Id header -> 401."""
        mw = BearerAuthMiddleware(_asgi_app(), api_key="", trust_gateway=True)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=mw), base_url="http://test"
        ) as client:
            resp = await client.get("/", headers={"X-Auth-User-Id": "not-a-uuid"})
        assert resp.status_code == 401
        assert resp.json() == {"error": "Invalid X-Auth-User-Id header"}

    async def test_authorization_header_ignored_when_trust_gateway(self) -> None:
        """Authorization header is IGNORED when trust_gateway=True.
        Even with a forged Bearer token + missing X-Auth-User-Id, response is 401.
        """
        mw = BearerAuthMiddleware(_asgi_app(), api_key="", trust_gateway=True)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=mw), base_url="http://test"
        ) as client:
            resp = await client.get(
                "/",
                headers={
                    "Authorization": "Bearer forged-token",
                    "X-Auth-User-Id": "",
                },
            )
        assert resp.status_code == 401

    async def test_introspector_not_called_when_trust_gateway(self) -> None:
        """When trust_gateway=True, Bearer/API-key/Hydra branches are skipped."""
        mock_iv = AsyncMock(spec=HydraIntrospector)
        mock_iv.introspect = AsyncMock(return_value=None)
        mw = BearerAuthMiddleware(
            _asgi_app(),
            api_key=TEST_API_KEY,
            introspector=mock_iv,
            trust_gateway=True,
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=mw), base_url="http://test"
        ) as client:
            resp = await client.get(
                "/",
                headers={
                    "Authorization": "Bearer forged-token",
                    "X-Auth-User-Id": str(self.TEST_USER_UUID),
                },
            )
        assert resp.status_code == 200
        mock_iv.introspect.assert_not_called()

    async def test_contextvar_reset_after_trust_gateway(self) -> None:
        """ContextVar is reset after the request completes."""
        mw = BearerAuthMiddleware(_asgi_app(), api_key="", trust_gateway=True)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=mw), base_url="http://test"
        ) as client:
            await client.get("/", headers={"X-Auth-User-Id": str(self.TEST_USER_UUID)})
        assert current_user_id.get() is None

    async def test_trust_gateway_false_still_uses_normal_auth(self) -> None:
        """Regression: trust_gateway=False (default) still uses normal auth paths."""
        mw = BearerAuthMiddleware(_asgi_app(), api_key=TEST_API_KEY, trust_gateway=False)
        # With trust_gateway=False, missing Auth header -> 401 (normal path)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=mw), base_url="http://test"
        ) as client:
            resp = await client.get("/")
        assert resp.status_code == 401

    async def test_trust_gateway_false_with_valid_api_key_still_works(self) -> None:
        """Regression: trust_gateway=False + valid API key -> 200 (normal path)."""
        mw = BearerAuthMiddleware(
            _asgi_app(),
            api_key=TEST_API_KEY,
            operator_user_id=TEST_OP_ID,
            trust_gateway=False,
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=mw), base_url="http://test"
        ) as client:
            resp = await client.get("/", headers={"Authorization": f"Bearer {TEST_API_KEY}"})
        assert resp.status_code == 200
