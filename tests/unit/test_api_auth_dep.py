"""Unit tests for the shared auth dependency in gubbi/api/v1/auth.py.

Tests all four auth modes (trust-gateway, static API key, Hydra bearer,
self-host OAuth), token-length cap, scope mismatch, and route integration.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest
from fastapi import Depends, FastAPI, Request
from fastapi.testclient import TestClient

from gubbi.api.v1.auth import require_scope

TEST_USER_ID = UUID("11111111-1111-1111-1111-111111111111")
TEST_OP_ID = UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
TEST_API_KEY = "a" * 64
TEST_ORY_TOKEN = "ory_at_" + "x" * 80

pytestmark = pytest.mark.asyncio(loop_scope="session")


def _make_app(
    trust_gateway: bool = False,
    api_key: str = "",
    api_key_scopes: list[str] | None = None,
    operator_user_id: UUID | None = TEST_OP_ID,
    hydra_introspector: AsyncMock | None = None,
    selfhost_token_validator: MagicMock | None = None,
    require_scope_arg: str | None = None,
    gateway_require_signature: bool = False,
    gateway_secret: bytes | None = None,
) -> FastAPI:
    """Build a minimal FastAPI app with a test route using the shared auth dep.

    The test route returns the resolved (user_id, scopes) as JSON so
    assertions can verify both values.
    """
    app = FastAPI()

    # Mock settings
    mock_auth_settings = MagicMock()
    mock_auth_settings.trust_gateway = trust_gateway
    mock_auth_settings.api_key = api_key
    mock_auth_settings.api_key_scopes = api_key_scopes or ["journal:read", "journal:write"]
    mock_auth_settings.gateway_require_signature = gateway_require_signature

    mock_settings = MagicMock()
    mock_settings.auth = mock_auth_settings

    mock_app_ctx = MagicMock()
    mock_app_ctx.settings = mock_settings
    mock_app_ctx.operator_user_id = operator_user_id

    # Set up app state
    app.state.app_ctx = mock_app_ctx
    app.state.operator_user_id = operator_user_id
    app.state.hydra_introspector = hydra_introspector
    app.state.selfhost_token_validator = selfhost_token_validator
    app.state.gubbi_gateway_secret = gateway_secret

    @app.get("/test-auth")
    async def test_route(
        request: Request,
        auth: tuple[UUID, frozenset[str]] = Depends(require_scope(require_scope_arg)),
    ):
        user_id, scopes = auth
        from fastapi.responses import JSONResponse

        return JSONResponse(
            {
                "user_id": str(user_id),
                "scopes": sorted(scopes),
            }
        )

    return app


class TestTrustGatewayMode:
    """Auth mode (a): trust-gateway envelope via X-Auth-User-Id header."""

    async def test_valid_user_id_and_scopes(self) -> None:
        app = _make_app(trust_gateway=True)
        client = TestClient(app)
        resp = client.get(
            "/test-auth",
            headers={
                "X-Auth-User-Id": str(TEST_USER_ID),
                "X-Auth-Scopes": "journal:read journal:write",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["user_id"] == str(TEST_USER_ID)
        assert sorted(data["scopes"]) == ["journal:read", "journal:write"]

    async def test_missing_header_returns_401(self) -> None:
        app = _make_app(trust_gateway=True)
        client = TestClient(app)
        resp = client.get("/test-auth")
        assert resp.status_code == 401
        assert "Missing X-Auth-User-Id" in resp.json()["detail"]

    async def test_malformed_uuid_returns_401(self) -> None:
        app = _make_app(trust_gateway=True)
        client = TestClient(app)
        resp = client.get("/test-auth", headers={"X-Auth-User-Id": "not-a-uuid"})
        assert resp.status_code == 401
        assert "Invalid X-Auth-User-Id" in resp.json()["detail"]

    async def test_empty_scopes_falls_to_legacy_default(self) -> None:
        app = _make_app(trust_gateway=True)
        client = TestClient(app)
        resp = client.get(
            "/test-auth",
            headers={"X-Auth-User-Id": str(TEST_USER_ID)},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert sorted(data["scopes"]) == ["journal:read", "journal:write"]


class TestTrustGatewayEnvelopeVerification:
    """Auth mode (a) -- H-1 HMAC envelope verification on REST.

    Mirrors the contract that BearerAuthMiddleware enforces on /mcp.
    Without these checks, anything on the gubbi private network
    could forge X-Auth-User-Id and bypass auth (the brief explicitly
    required reusing the middleware's verification path).
    """

    _SECRET = b"\x42" * 32

    @staticmethod
    def _signed_headers(
        user_id: UUID,
        scopes: str = "journal:read journal:write",
        method: str = "GET",
        path: str = "/test-auth",
        secret: bytes | None = None,
        skew_seconds: int = 0,
    ) -> dict[str, str]:
        from datetime import UTC, datetime, timedelta

        from gubbi_common.auth.gateway_signature import (
            GATEWAY_CONTRACT_VERSION,
            build_signature,
        )

        sec = secret if secret is not None else TestTrustGatewayEnvelopeVerification._SECRET
        ts_dt = datetime.now(UTC) + timedelta(seconds=skew_seconds)
        ts = ts_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        sig = build_signature(sec, str(user_id), scopes, ts, method, path)
        return {
            "X-Auth-User-Id": str(user_id),
            "X-Auth-Scopes": scopes,
            "X-Auth-Timestamp": ts,
            "X-Auth-Signature": sig,
            "X-Auth-Contract-Version": str(GATEWAY_CONTRACT_VERSION),
        }

    async def test_valid_signature_passes(self) -> None:
        app = _make_app(
            trust_gateway=True,
            gateway_require_signature=True,
            gateway_secret=self._SECRET,
        )
        client = TestClient(app)
        resp = client.get("/test-auth", headers=self._signed_headers(TEST_USER_ID))
        assert resp.status_code == 200
        assert resp.json()["user_id"] == str(TEST_USER_ID)

    async def test_tampered_signature_returns_401(self) -> None:
        """Forged user_id with a valid signature for a DIFFERENT user is rejected."""
        app = _make_app(
            trust_gateway=True,
            gateway_require_signature=True,
            gateway_secret=self._SECRET,
        )
        client = TestClient(app)
        # Sign for user A but send user B in the header
        other_user = UUID("99999999-9999-9999-9999-999999999999")
        headers = self._signed_headers(other_user)
        headers["X-Auth-User-Id"] = str(TEST_USER_ID)
        resp = client.get("/test-auth", headers=headers)
        assert resp.status_code == 401
        assert "Invalid gateway signature" in resp.json()["detail"]

    async def test_wrong_secret_returns_401(self) -> None:
        app = _make_app(
            trust_gateway=True,
            gateway_require_signature=True,
            gateway_secret=self._SECRET,
        )
        client = TestClient(app)
        headers = self._signed_headers(TEST_USER_ID, secret=b"\x99" * 32)
        resp = client.get("/test-auth", headers=headers)
        assert resp.status_code == 401
        assert "Invalid gateway signature" in resp.json()["detail"]

    async def test_required_signature_missing_returns_401(self) -> None:
        """gateway_require_signature=true + bare X-Auth-User-Id (no envelope)
        must fail closed."""
        app = _make_app(
            trust_gateway=True,
            gateway_require_signature=True,
            gateway_secret=self._SECRET,
        )
        client = TestClient(app)
        resp = client.get("/test-auth", headers={"X-Auth-User-Id": str(TEST_USER_ID)})
        assert resp.status_code == 401

    async def test_required_signature_secret_not_configured_returns_503(self) -> None:
        """gateway_require_signature=true + secret not provisioned is a deployment
        misconfiguration -- 503, not 401."""
        app = _make_app(
            trust_gateway=True,
            gateway_require_signature=True,
            gateway_secret=None,
        )
        client = TestClient(app)
        resp = client.get("/test-auth", headers=self._signed_headers(TEST_USER_ID))
        assert resp.status_code == 503

    async def test_wrong_contract_version_returns_401(self) -> None:
        app = _make_app(
            trust_gateway=True,
            gateway_require_signature=True,
            gateway_secret=self._SECRET,
        )
        client = TestClient(app)
        headers = self._signed_headers(TEST_USER_ID)
        headers["X-Auth-Contract-Version"] = "999"
        resp = client.get("/test-auth", headers=headers)
        assert resp.status_code == 401
        assert "Unsupported X-Auth-Contract-Version" in resp.json()["detail"]

    async def test_legacy_path_accepts_unsigned_when_not_required(self) -> None:
        """gateway_require_signature=false + no signature header -> legacy path
        accepts bare X-Auth-User-Id. Same semantics as the MCP middleware."""
        app = _make_app(
            trust_gateway=True,
            gateway_require_signature=False,
            gateway_secret=self._SECRET,
        )
        client = TestClient(app)
        resp = client.get(
            "/test-auth",
            headers={
                "X-Auth-User-Id": str(TEST_USER_ID),
                "X-Auth-Scopes": "journal:read",
            },
        )
        assert resp.status_code == 200


class TestStaticAPIKeyMode:
    """Auth mode (b): static API key from Authorization header."""

    async def test_valid_api_key_returns_user_and_scopes(self) -> None:
        app = _make_app(
            api_key=TEST_API_KEY,
            api_key_scopes=["journal:read", "journal:write"],
            operator_user_id=TEST_OP_ID,
        )
        client = TestClient(app)
        resp = client.get("/test-auth", headers={"Authorization": f"Bearer {TEST_API_KEY}"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["user_id"] == str(TEST_OP_ID)
        assert sorted(data["scopes"]) == ["journal:read", "journal:write"]

    async def test_wrong_api_key_returns_401(self) -> None:
        app = _make_app(api_key=TEST_API_KEY, operator_user_id=TEST_OP_ID)
        client = TestClient(app)
        resp = client.get("/test-auth", headers={"Authorization": "Bearer wrong_key"})
        assert resp.status_code == 401
        assert "invalid or expired token" in resp.json()["detail"].lower()

    async def test_no_operator_returns_503(self) -> None:
        app = _make_app(api_key=TEST_API_KEY, operator_user_id=None)
        client = TestClient(app)
        resp = client.get("/test-auth", headers={"Authorization": f"Bearer {TEST_API_KEY}"})
        assert resp.status_code == 503
        assert "operator not provisioned" in resp.json()["detail"].lower()

    async def test_missing_auth_header_returns_401(self) -> None:
        app = _make_app(api_key=TEST_API_KEY)
        client = TestClient(app)
        resp = client.get("/test-auth")
        assert resp.status_code == 401

    async def test_empty_api_key_path_fails(self) -> None:
        """When api_key is empty, bearer tokens are rejected."""
        app = _make_app(api_key="", api_key_scopes=[], operator_user_id=TEST_OP_ID)
        client = TestClient(app)
        resp = client.get("/test-auth", headers={"Authorization": f"Bearer {TEST_API_KEY}"})
        assert resp.status_code == 401


class TestHydraBearerMode:
    """Auth mode (c): Hydra introspection for ory_at_ tokens."""

    @pytest.fixture
    def mock_introspector(self) -> AsyncMock:
        from gubbi.auth.hydra import TokenClaims

        mock_iv = AsyncMock()
        mock_iv.introspect = AsyncMock(
            return_value=TokenClaims(
                sub=TEST_USER_ID,
                scope="journal:read journal:write",
                exp=9999999999,
            )
        )
        return mock_iv

    async def test_valid_ory_token_returns_user_and_scopes(
        self, mock_introspector: AsyncMock
    ) -> None:
        app = _make_app(api_key=TEST_API_KEY, hydra_introspector=mock_introspector)
        client = TestClient(app)
        resp = client.get("/test-auth", headers={"Authorization": f"Bearer {TEST_ORY_TOKEN}"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["user_id"] == str(TEST_USER_ID)
        assert sorted(data["scopes"]) == ["journal:read", "journal:write"]

    async def test_hydra_unreachable_returns_503(self, mock_introspector: AsyncMock) -> None:
        from gubbi.auth.hydra import HydraUnreachable

        mock_introspector.introspect = AsyncMock(side_effect=HydraUnreachable("timeout"))
        app = _make_app(api_key=TEST_API_KEY, hydra_introspector=mock_introspector)
        client = TestClient(app)
        resp = client.get("/test-auth", headers={"Authorization": f"Bearer {TEST_ORY_TOKEN}"})
        assert resp.status_code == 503

    async def test_hydra_invalid_token_returns_401(self, mock_introspector: AsyncMock) -> None:
        from gubbi.auth.hydra import HydraInvalidToken

        mock_introspector.introspect = AsyncMock(side_effect=HydraInvalidToken("bad"))
        app = _make_app(api_key=TEST_API_KEY, hydra_introspector=mock_introspector)
        client = TestClient(app)
        resp = client.get("/test-auth", headers={"Authorization": f"Bearer {TEST_ORY_TOKEN}"})
        assert resp.status_code == 401

    async def test_introspector_none_rejects_ory_token(self) -> None:
        app = _make_app(api_key=TEST_API_KEY, hydra_introspector=None)
        client = TestClient(app)
        resp = client.get("/test-auth", headers={"Authorization": f"Bearer {TEST_ORY_TOKEN}"})
        assert resp.status_code == 401


class TestSelfhostOAuthMode:
    """Auth mode (d): self-host OAuth token validation."""

    @pytest.fixture
    def mock_validator(self) -> MagicMock:
        return MagicMock(return_value=frozenset({"journal:read", "journal:write"}))

    async def test_valid_token_returns_user_and_scopes(self, mock_validator: MagicMock) -> None:
        app = _make_app(
            api_key="",
            api_key_scopes=[],
            operator_user_id=TEST_OP_ID,
            selfhost_token_validator=mock_validator,
        )
        client = TestClient(app)
        resp = client.get("/test-auth", headers={"Authorization": "Bearer some_token"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["user_id"] == str(TEST_OP_ID)
        assert sorted(data["scopes"]) == ["journal:read", "journal:write"]

    async def test_rejected_token_returns_401(self, mock_validator: MagicMock) -> None:
        mock_validator.return_value = None
        app = _make_app(
            api_key="",
            api_key_scopes=[],
            operator_user_id=TEST_OP_ID,
            selfhost_token_validator=mock_validator,
        )
        client = TestClient(app)
        resp = client.get("/test-auth", headers={"Authorization": "Bearer bad_token"})
        assert resp.status_code == 401

    async def test_no_validator_returns_401(self) -> None:
        app = _make_app(
            api_key="",
            api_key_scopes=[],
            operator_user_id=TEST_OP_ID,
            selfhost_token_validator=None,
        )
        client = TestClient(app)
        resp = client.get("/test-auth", headers={"Authorization": "Bearer some_token"})
        assert resp.status_code == 401

    async def test_no_operator_returns_503(self, mock_validator: MagicMock) -> None:
        app = _make_app(
            api_key="",
            api_key_scopes=[],
            operator_user_id=None,
            selfhost_token_validator=mock_validator,
        )
        client = TestClient(app)
        resp = client.get("/test-auth", headers={"Authorization": "Bearer some_token"})
        assert resp.status_code == 503


class TestTokenLengthCap:
    """Tokens longer than MAX_BEARER_TOKEN_LEN (256) are rejected."""

    async def test_oversized_token_returns_401(self) -> None:
        app = _make_app(api_key=TEST_API_KEY, operator_user_id=TEST_OP_ID)
        client = TestClient(app)
        oversized = "X" * 300
        resp = client.get("/test-auth", headers={"Authorization": f"Bearer {oversized}"})
        assert resp.status_code == 401
        assert "Invalid token" in resp.json()["detail"]


class TestScopeMismatch:
    """When a required scope is not granted, returns 403."""

    async def test_missing_required_scope_returns_403(self) -> None:
        """Token has journal:read but requires journal:write -> 403."""
        from gubbi.auth.hydra import TokenClaims

        mock_iv = AsyncMock()
        mock_iv.introspect = AsyncMock(
            return_value=TokenClaims(
                sub=TEST_USER_ID,
                scope="journal:read",
                exp=9999999999,
            )
        )
        # require_scope("journal:write")
        app = _make_app(
            api_key=TEST_API_KEY,
            hydra_introspector=mock_iv,
            require_scope_arg="journal:write",
        )
        client = TestClient(app)
        resp = client.get("/test-auth", headers={"Authorization": f"Bearer {TEST_ORY_TOKEN}"})
        assert resp.status_code == 403
        assert "insufficient_scope" in resp.json()["detail"]


class TestRouteIntegration:
    """Both /v1/ingest and /v1/extraction/progress routes accept the shared dep."""

    async def test_ingest_route_accepts_shared_dep(self) -> None:
        """Verify the /api/v1/ingest/conversations route uses require_scope."""
        from gubbi.api.v1.ingest import router as ingest_router

        app = FastAPI()
        app.include_router(ingest_router, prefix="/api/v1")

        from starlette.routing import Route

        routes = [
            r
            for r in app.routes
            if isinstance(r, Route) and r.path == "/api/v1/ingest/conversations"
        ]
        assert len(routes) == 1

    async def test_extraction_progress_route_accepts_shared_dep(self) -> None:
        """Verify the /api/v1/extraction/progress route uses require_scope."""
        from gubbi.api.v1.extraction import router as extraction_router

        app = FastAPI()
        app.include_router(extraction_router, prefix="/api/v1")

        from starlette.routing import Route

        routes = [
            r
            for r in app.routes
            if isinstance(r, Route) and r.path == "/api/v1/extraction/progress"
        ]
        assert len(routes) == 1
