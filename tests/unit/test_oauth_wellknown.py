"""Unit tests for oauth.wellknown module."""

from fastapi import FastAPI
from pydantic import AnyHttpUrl

from journalctl.config import Settings
from journalctl.oauth.wellknown import register


def _make_settings(**overrides: str) -> Settings:
    defaults: dict[str, str] = {
        "api_key": "testkey123",
        "password_hash": "",
        "hydra_admin_url": "",
        "hydra_public_issuer_url": "",
        "server_url": "http://localhost:8100",
        "db_app_url": "sqlite:///memory:",
        "operator_email": "admin@example.com",
    }
    defaults.update(overrides)
    return Settings.model_construct(**defaults)  # type: ignore[call-arg, arg-type]


class TestWellknownRegister:
    def test_registers_protected_resource_routes(self) -> None:
        settings = _make_settings()
        app = FastAPI()
        authorization_servers = [AnyHttpUrl("https://auth.example.com")]
        register(app, settings, authorization_servers)
        route_paths = [r.path for r in app.routes if hasattr(r, "path")]
        assert any(
            ".well-known/oauth-protected-resource/mcp" in p for p in route_paths
        ), f"Expected well-known route; got: {route_paths}"

    def test_no_auth_routes_registered(self) -> None:
        settings = _make_settings()
        app = FastAPI()
        authorization_servers = [AnyHttpUrl("https://auth.example.com")]
        register(app, settings, authorization_servers)
        route_paths = [r.path for r in app.routes if hasattr(r, "path")]
        for forbidden in ["/authorize", "/token", "/register", "/login"]:
            assert (
                forbidden not in route_paths
            ), f"{forbidden} should not be registered; got: {route_paths}"
