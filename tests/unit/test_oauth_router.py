"""Unit tests for register_oauth_routes across the three deploy shapes."""

from pathlib import Path

from fastapi import FastAPI
from starlette.routing import Route

from gubbi.config import AuthConfig, DbConfig, ServerConfig, Settings
from gubbi.oauth.router import register_oauth_routes
from gubbi.oauth.storage import OAuthStorage


def _make_settings(
    api_key: str = "",
    password_hash: str = "",
    hydra_admin_url: str = "",
    hydra_public_issuer_url: str = "",
    server_url: str = "http://localhost:8100",
    db_app_url: str = "sqlite:///memory:",
    operator_email: str = "",
) -> Settings:
    """Build a Settings instance bypassing validation (model_construct).

    This sidesteps the deploy-shape validator so tests can exercise
    combinations the validator normally forbids (e.g. Mode 3 without an
    API key).
    """
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


def _routes_exist(app: FastAPI, desired_paths: list[str]) -> bool:
    """Return True if every path in *desired_paths* has a matching Route."""
    registered: set[str] = {
        r.path  # type: ignore[union-attr]
        for r in app.routes
        if isinstance(r, Route)
    }
    return all(p in registered for p in desired_paths)


async def _make_storage(tmp_path: Path) -> OAuthStorage:
    """Create an OAuthStorage backed by a temp file."""
    storage = OAuthStorage(tmp_path / "oauth.db")
    await storage.initialize()
    return storage


class TestMode1NoOAuth:
    """Mode 1: neither password_hash nor hydra_admin_url.

    No OAuth routes should be registered. The function returns None.
    """

    async def test_returns_none(self, tmp_path: Path) -> None:
        settings = _make_settings()
        app = FastAPI()
        storage = await _make_storage(tmp_path)
        result = await register_oauth_routes(app, storage, settings)
        await storage.close()
        assert result is None

    async def test_no_oauth_routes_registered(self, tmp_path: Path) -> None:
        settings = _make_settings()
        app = FastAPI()
        storage = await _make_storage(tmp_path)
        await register_oauth_routes(app, storage, settings)
        route_paths = [r.path for r in app.routes if isinstance(r, Route)]
        await storage.close()
        oauth_paths = [
            "/authorize",
            "/token",
            "/register",
            "/login",
            "/.well-known/oauth-protected-resource/mcp",
        ]
        assert not any(p in route_paths for p in oauth_paths), (
            f"Expected no OAuth routes; got: {route_paths}"
        )


class TestMode2SelfHostOAuth:
    """Mode 2: password_hash set, no hydra.

    All Phase 3.5 routes are registered: authenticate, authorize, token,
    register, revoke, login, plus the protected-resource metadata route.
    A token_validator callable is returned.
    """

    async def test_returns_token_validator(self, tmp_path: Path) -> None:
        settings = _make_settings(password_hash="hashedpw123")
        app = FastAPI()
        storage = await _make_storage(tmp_path)
        result = await register_oauth_routes(app, storage, settings)
        await storage.close()
        assert callable(result)

    async def test_all_oauth_routes_present(self, tmp_path: Path) -> None:
        settings = _make_settings(password_hash="hashedpw123")
        app = FastAPI()
        storage = await _make_storage(tmp_path)
        await register_oauth_routes(app, storage, settings)
        expected_paths = [
            "/authorize",
            "/token",
            "/register",
            "/.well-known/oauth-protected-resource/mcp",
            "/login",
        ]
        await storage.close()
        assert _routes_exist(app, expected_paths), (
            "Missing OAuth routes. "
            f"Registered: {[r.path for r in app.routes if isinstance(r, Route)]}"
        )


class TestMode3HydraBacked:
    """Mode 3: hydra_admin_url set, password_hash empty.

    Only protected-resource routes are registered, pointing at the
    Hydra public issuer as the authorization server. No auth-flow
    routes (/authorize, /token, /register, /login) are present.
    Returns None because Hydra introspection middleware owns auth.
    """

    async def test_returns_none(self, tmp_path: Path) -> None:
        settings = _make_settings(
            hydra_admin_url="http://hydra:4445",
            hydra_public_issuer_url="https://auth.example.com",
        )
        app = FastAPI()
        storage = await _make_storage(tmp_path)
        result = await register_oauth_routes(app, storage, settings)
        await storage.close()
        assert result is None

    async def test_only_protected_resource_route(self, tmp_path: Path) -> None:
        settings = _make_settings(
            hydra_admin_url="http://hydra:4445",
            hydra_public_issuer_url="https://auth.example.com",
        )
        app = FastAPI()
        storage = await _make_storage(tmp_path)
        await register_oauth_routes(app, storage, settings)
        route_paths = [r.path for r in app.routes if isinstance(r, Route)]
        await storage.close()
        assert any(".well-known/oauth-protected-resource/mcp" in p for p in route_paths), (
            f"Missing protected-resource route; got: {route_paths}"
        )
        for forbidden in ["/authorize", "/token", "/register", "/login"]:
            assert forbidden not in route_paths, (
                f"{forbidden} should not be registered in Mode 3; got: {route_paths}"
            )
