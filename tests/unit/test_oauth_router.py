"""Unit tests for register_oauth_routes across the three deploy shapes."""

from pathlib import Path

import pytest
from fastapi import FastAPI
from starlette.routing import Route

from journalctl.config import Settings
from journalctl.oauth.router import register_oauth_routes
from journalctl.oauth.storage import OAuthStorage


def _make_settings(**overrides: str) -> Settings:
    """Build a Settings instance bypassing validation (model_construct).

    This sidesteps the deploy-shape validator so tests can exercise
    combinations the validator normally forbids (e.g. Mode 3 without an
    API key).
    """
    defaults: dict[str, str] = {
        "api_key": "",
        "password_hash": "",
        "hydra_admin_url": "",
        "hydra_public_issuer_url": "",
        "server_url": "http://localhost:8100",
        "db_app_url": "sqlite:///memory:",
        "operator_email": "",
    }
    defaults.update(overrides)
    return Settings.model_construct(**defaults)  # type: ignore[call-arg, arg-type]


def _routes_exist(app: FastAPI, desired_paths: list[str]) -> bool:
    """Return True if every path in *desired_paths* has a matching Route."""
    registered: set[str] = {
        r.path  # type: ignore[union-attr]
        for r in app.routes
        if isinstance(r, Route)
    }
    return all(p in registered for p in desired_paths)


def _make_storage(tmp_path: Path) -> OAuthStorage:
    """Create an OAuthStorage backed by a temp file."""
    storage = OAuthStorage(tmp_path / "oauth.db")
    _ = storage.conn  # Force schema init
    return storage


class TestMode1NoOAuth:
    """Mode 1: neither password_hash nor hydra_admin_url.

    No OAuth routes should be registered. The function returns None.
    """

    def test_returns_none(self, tmp_path: Path) -> None:
        settings = _make_settings()
        app = FastAPI()
        storage = _make_storage(tmp_path)
        # Storage is auto-cleaned by tmp_path fixture on test exit
        result = register_oauth_routes(app, storage, settings)
        storage.close()
        assert result is None

    def test_no_oauth_routes_registered(self, tmp_path: Path) -> None:
        settings = _make_settings()
        app = FastAPI()
        storage = _make_storage(tmp_path)
        register_oauth_routes(app, storage, settings)
        route_paths = [r.path for r in app.routes if isinstance(r, Route)]
        storage.close()
        oauth_paths = [
            "/authorize",
            "/token",
            "/register",
            "/login",
            "/.well-known/oauth-protected-resource/mcp",
        ]
        assert not any(
            p in route_paths for p in oauth_paths
        ), f"Expected no OAuth routes; got: {route_paths}"


class TestMode2SelfHostOAuth:
    """Mode 2: password_hash set, no hydra.

    All Phase 3.5 routes are registered: authenticate, authorize, token,
    register, revoke, login, plus the protected-resource metadata route.
    A token_validator callable is returned.
    """

    def test_returns_token_validator(self, tmp_path: Path) -> None:
        settings = _make_settings(password_hash="hashedpw123")
        app = FastAPI()
        storage = _make_storage(tmp_path)
        result = register_oauth_routes(app, storage, settings)
        storage.close()
        assert callable(result)

    def test_all_oauth_routes_present(self, tmp_path: Path) -> None:
        settings = _make_settings(password_hash="hashedpw123")
        app = FastAPI()
        storage = _make_storage(tmp_path)
        register_oauth_routes(app, storage, settings)
        expected_paths = [
            "/authorize",
            "/token",
            "/register",
            "/.well-known/oauth-protected-resource/mcp",
            "/login",
        ]
        storage.close()
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

    def test_returns_none(self, tmp_path: Path) -> None:
        settings = _make_settings(
            hydra_admin_url="http://hydra:4445",
            hydra_public_issuer_url="https://auth-dev.meepolabs.com",
        )
        app = FastAPI()
        storage = _make_storage(tmp_path)
        result = register_oauth_routes(app, storage, settings)
        storage.close()
        assert result is None

    def test_only_protected_resource_route(self, tmp_path: Path) -> None:
        settings = _make_settings(
            hydra_admin_url="http://hydra:4445",
            hydra_public_issuer_url="https://auth-dev.meepolabs.com",
        )
        app = FastAPI()
        storage = _make_storage(tmp_path)
        register_oauth_routes(app, storage, settings)
        route_paths = [r.path for r in app.routes if isinstance(r, Route)]
        storage.close()
        assert any(
            ".well-known/oauth-protected-resource/mcp" in p for p in route_paths
        ), f"Missing protected-resource route; got: {route_paths}"
        for forbidden in ["/authorize", "/token", "/register", "/login"]:
            assert (
                forbidden not in route_paths
            ), f"{forbidden} should not be registered in Mode 3; got: {route_paths}"


class TestDeployShapeValidator:
    """Verify the deploy-shape validator rejects partial Mode 3 config."""

    def _patch_mode3(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Clear env vars that would interfere, then set Mode 3 triples."""
        for key in ("JOURNAL_PASSWORD_HASH",):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("JOURNAL_HYDRA_ADMIN_URL", "http://hydra:4445")
        monkeypatch.setenv("JOURNAL_HYDRA_PUBLIC_ISSUER_URL", "https://auth.meepo.com")
        monkeypatch.setenv("JOURNAL_HYDRA_PUBLIC_URL", "https://hydra.example.com")
        monkeypatch.setenv("JOURNAL_DB_APP_URL", "sqlite:///memory:")
        from journalctl.config import get_settings

        get_settings.cache_clear()

    def _get_settings(self) -> Settings:
        """Rebuild Settings from current env (cache cleared)."""
        from journalctl.config import get_settings

        return get_settings()

    def test_valid_mode3_both_hydra_fields_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._patch_mode3(monkeypatch)
        # Should succeed when both hydra fields are present
        settings = self._get_settings()
        assert settings.hydra_admin_url
        assert settings.hydra_public_issuer_url
        assert settings.hydra_public_url

    def test_hydra_admin_without_issuer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("JOURNAL_PASSWORD_HASH", raising=False)
        monkeypatch.setenv("JOURNAL_HYDRA_ADMIN_URL", "http://hydra:4445")
        monkeypatch.delenv("JOURNAL_HYDRA_PUBLIC_ISSUER_URL", raising=False)
        monkeypatch.setenv("JOURNAL_DB_APP_URL", "sqlite:///memory:")
        from journalctl.config import get_settings

        get_settings.cache_clear()
        with pytest.raises(ValueError, match="(?i)hydra_public_issuer_url.*required"):
            self._get_settings()

    def test_hydra_issuer_without_admin(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("JOURNAL_PASSWORD_HASH", raising=False)
        monkeypatch.delenv("JOURNAL_HYDRA_ADMIN_URL", raising=False)
        monkeypatch.setenv("JOURNAL_HYDRA_PUBLIC_ISSUER_URL", "https://auth.meepo.com")
        monkeypatch.setenv("JOURNAL_DB_APP_URL", "sqlite:///memory:")
        from journalctl.config import get_settings

        get_settings.cache_clear()
        with pytest.raises(ValueError, match="(?i)hydra_admin_url.*required"):
            self._get_settings()

    def test_hydra_admin_without_public_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Hydra ADMIN_URL without PUBLIC_URL is a rejectable error."""
        for key in ("JOURNAL_PASSWORD_HASH", "JOURNAL_HYDRA_PUBLIC_URL"):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("JOURNAL_HYDRA_ADMIN_URL", "http://hydra:4445")
        monkeypatch.setenv("JOURNAL_HYDRA_PUBLIC_ISSUER_URL", "https://auth.meepo.com")
        monkeypatch.setenv("JOURNAL_DB_APP_URL", "sqlite:///memory:")
        from journalctl.config import get_settings

        get_settings.cache_clear()
        with pytest.raises(ValueError, match="(?i)hydra_public_url.*required"):
            self._get_settings()

    def test_hydra_public_url_without_admin(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """PUBLIC_URL without ADMIN_URL is a rejectable error."""
        for key in ("JOURNAL_PASSWORD_HASH",):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.delenv("JOURNAL_HYDRA_ADMIN_URL", raising=False)
        monkeypatch.setenv("JOURNAL_HYDRA_PUBLIC_ISSUER_URL", "https://auth.meepo.com")
        monkeypatch.setenv("JOURNAL_HYDRA_PUBLIC_URL", "https://hydra.example.com")
        monkeypatch.setenv("JOURNAL_DB_APP_URL", "sqlite:///memory:")
        from journalctl.config import get_settings

        get_settings.cache_clear()
        with pytest.raises(ValueError, match="(?i)hydra_admin_url.*required"):
            self._get_settings()
