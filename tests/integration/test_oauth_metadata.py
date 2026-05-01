"""Integration test for Mode 3 OAuth protected-resource metadata endpoint."""

from pathlib import Path

from fastapi import FastAPI
from starlette.testclient import TestClient

from journalctl.config import AuthConfig, DbConfig, ServerConfig, Settings
from journalctl.oauth.router import register_oauth_routes
from journalctl.oauth.storage import OAuthStorage


def _make_mode3_settings() -> Settings:
    """Build a Mode 3 Settings instance (Hydra-backed) bypassing validation."""
    return Settings.model_construct(
        db=DbConfig.model_construct(app_url="sqlite:///memory:", admin_url=""),
        auth=AuthConfig.model_construct(
            password_hash="",
            hydra_admin_url="http://hydra:4445",
            hydra_public_issuer_url="https://auth-dev.meepolabs.com",
            hydra_public_url=None,
            api_key="",
            operator_email="",
            trust_gateway=False,
        ),
        server=ServerConfig.model_construct(
            url="http://localhost:8100",
            host="0.0.0.0",  # noqa: S104
            port=8100,
            transport="streamable-http",
        ),
    )  # type: ignore[call-arg]


def _make_app(storage: OAuthStorage) -> FastAPI:
    """Create a minimal FastAPI app with only Mode 3 OAuth routes."""
    app = FastAPI()
    register_oauth_routes(app, storage, _make_mode3_settings())
    return app


class TestMode3ProtectedResourceMetadata:
    """Verify the /.well-known/oauth-protected-resource/mcp endpoint in Mode 3.

    RFC 9728 requires the response to contain "resource" and
    "authorization_servers" fields.
    """

    def test_get_protected_resource_returns_200(self, tmp_path: Path) -> None:
        storage = OAuthStorage(tmp_path / "oauth.db")
        _ = storage.conn  # Force schema init
        app = _make_app(storage)
        storage.close()
        client = TestClient(app)
        response = client.get("/.well-known/oauth-protected-resource/mcp")
        assert response.status_code == 200

    def test_response_is_valid_json(self, tmp_path: Path) -> None:
        storage = OAuthStorage(tmp_path / "oauth.db")
        _ = storage.conn
        app = _make_app(storage)
        storage.close()
        client = TestClient(app)
        response = client.get("/.well-known/oauth-protected-resource/mcp")
        data = response.json()
        assert isinstance(data, dict)

    def test_response_contains_resource_field(self, tmp_path: Path) -> None:
        storage = OAuthStorage(tmp_path / "oauth.db")
        _ = storage.conn
        app = _make_app(storage)
        storage.close()
        client = TestClient(app)
        response = client.get("/.well-known/oauth-protected-resource/mcp")
        data = response.json()
        assert data["resource"] == "http://localhost:8100/mcp"

    def test_authorization_servers_points_at_public_issuer(self, tmp_path: Path) -> None:
        """The authorization_servers list must reference the PUBLIC issuer URL,
        not the internal hydra_admin_url. This is the critical correctness
        check.
        """
        storage = OAuthStorage(tmp_path / "oauth.db")
        _ = storage.conn
        app = _make_app(storage)
        storage.close()
        client = TestClient(app)
        response = client.get("/.well-known/oauth-protected-resource/mcp")
        data = response.json()
        # AnyHttpUrl adds a trailing slash in its string representation.
        assert data["authorization_servers"] == ["https://auth-dev.meepolabs.com/"]

    def test_rfc9728_compliant_fields(self, tmp_path: Path) -> None:
        """RFC 9728 requires 'resource' and 'authorization_servers' keys."""
        storage = OAuthStorage(tmp_path / "oauth.db")
        _ = storage.conn
        app = _make_app(storage)
        storage.close()
        client = TestClient(app)
        response = client.get("/.well-known/oauth-protected-resource/mcp")
        data = response.json()
        assert "resource" in data
        assert "authorization_servers" in data
        assert isinstance(data["authorization_servers"], list)
        assert len(data["authorization_servers"]) == 1
