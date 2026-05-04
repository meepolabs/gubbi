"""Unit tests for oauth.selfhost module."""

from pathlib import Path

from fastapi import FastAPI
from starlette.routing import Route

from gubbi.config import AuthConfig, DbConfig, ServerConfig, Settings
from gubbi.oauth.selfhost import _make_token_validator, register
from gubbi.oauth.storage import OAuthStorage


def _make_settings(
    api_key: str = "testkey123",
    password_hash: str = "hashedpw123",
    hydra_admin_url: str = "",
    hydra_public_issuer_url: str = "",
    server_url: str = "http://localhost:8100",
    db_app_url: str = "sqlite:///memory:",
    operator_email: str = "admin@example.com",
) -> Settings:
    return Settings.model_construct(
        db=DbConfig.model_construct(app_url=db_app_url, admin_url=""),
        auth=AuthConfig.model_construct(
            api_key=api_key,
            password_hash=password_hash,
            hydra_admin_url=hydra_admin_url,
            hydra_public_issuer_url=hydra_public_issuer_url,
            hydra_public_url=None,
            operator_email=operator_email,
            trust_gateway=False,
        ),
        server=ServerConfig.model_construct(
            url=server_url,
            host="0.0.0.0",  # noqa: S104
            port=8100,
            transport="streamable-http",
        ),
    )  # type: ignore[call-arg]


def _make_storage(tmp_path: Path) -> OAuthStorage:
    storage = OAuthStorage(tmp_path / "oauth.db")
    _ = storage.conn  # Force schema init
    return storage


class TestSelfhostRegister:
    def test_returns_token_validator(self, tmp_path: Path) -> None:
        settings = _make_settings()
        app = FastAPI()
        storage = _make_storage(tmp_path)
        result = register(app, storage, settings)
        storage.close()
        assert callable(result)

    def test_all_oauth_routes_present(self, tmp_path: Path) -> None:
        settings = _make_settings()
        app = FastAPI()
        storage = _make_storage(tmp_path)
        register(app, storage, settings)
        expected_paths = [
            "/authorize",
            "/token",
            "/register",
            "/.well-known/oauth-protected-resource/mcp",
            "/login",
        ]
        registered: set[str] = {
            r.path  # type: ignore[union-attr]
            for r in app.routes
            if isinstance(r, Route)
        }
        storage.close()
        for p in expected_paths:
            assert p in registered, f"Missing route {p}; got: {registered}"

    def test_token_validator_rejects_unknown(self, tmp_path: Path) -> None:
        storage = _make_storage(tmp_path)
        validator = _make_token_validator(storage)
        storage.close()
        assert validator("nonexistent") is None
