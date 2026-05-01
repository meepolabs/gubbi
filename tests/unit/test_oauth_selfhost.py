"""Unit tests for oauth.selfhost module."""

from pathlib import Path

from fastapi import FastAPI
from starlette.routing import Route

from journalctl.config import Settings
from journalctl.oauth.selfhost import _make_token_validator, register
from journalctl.oauth.storage import OAuthStorage


def _make_settings(**overrides: str) -> Settings:
    defaults: dict[str, str] = {
        "api_key": "testkey123",
        "password_hash": "hashedpw123",
        "hydra_admin_url": "",
        "hydra_public_issuer_url": "",
        "server_url": "http://localhost:8100",
        "db_app_url": "sqlite:///memory:",
        "operator_email": "admin@example.com",
    }
    defaults.update(overrides)
    return Settings.model_construct(**defaults)  # type: ignore[call-arg, arg-type]


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
        assert validator("nonexistent") is False
