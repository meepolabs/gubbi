"""Unit tests for oauth.wellknown module."""

from fastapi import FastAPI
from pydantic import AnyHttpUrl

from journalctl.config import AuthConfig, DbConfig, ServerConfig, Settings
from journalctl.oauth.wellknown import register


def _make_settings(
    api_key: str = "testkey123",
    password_hash: str = "",
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
