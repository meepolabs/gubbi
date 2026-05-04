"""Tests verifying nested Settings construction from flat env vars."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from gubbi.config import Settings


def _base_env(**extra: str) -> dict[str, str]:
    base: dict[str, str] = {
        "JOURNAL_DB_APP_URL": "postgresql://user:pass@localhost/db",
        "JOURNAL_API_KEY": "a-valid-api-key-that-is-at-least-32-chars-long",
        "JOURNAL_OPERATOR_EMAIL": "op@example.com",
    }
    base.update(extra)
    return base


def _make_settings(**extra: str) -> Settings:
    env = _base_env(**extra)
    with patch.dict(os.environ, env, clear=False):
        return Settings()


def test_flat_db_url_populates_nested_db() -> None:
    s = _make_settings()
    assert s.db.app_url == "postgresql://user:pass@localhost/db"
    assert s.db.admin_url == ""


def test_flat_api_key_populates_nested_auth() -> None:
    s = _make_settings()
    assert s.auth.api_key == "a-valid-api-key-that-is-at-least-32-chars-long"
    assert s.auth.operator_email == "op@example.com"


def test_flat_transport_populates_nested_server() -> None:
    s = _make_settings(JOURNAL_TRANSPORT="stdio", JOURNAL_PORT="9000")
    assert s.server.transport == "stdio"
    assert s.server.port == 9000


def test_flat_server_url_populates_nested_server_url() -> None:
    s = _make_settings(JOURNAL_SERVER_URL="http://myhost:9999")
    assert s.server.url == "http://myhost:9999"


def test_flat_db_admin_url_populates_nested() -> None:
    s = _make_settings(JOURNAL_DB_ADMIN_URL="postgresql://admin:pw@localhost/db")
    assert s.db.admin_url == "postgresql://admin:pw@localhost/db"


def test_new_style_double_underscore_also_works() -> None:
    env = {
        "JOURNAL_DB__APP_URL": "postgresql://user:pass@localhost/db2",
        "JOURNAL_API_KEY": "a-valid-api-key-that-is-at-least-32-chars-long",
        "JOURNAL_OPERATOR_EMAIL": "op@example.com",
    }
    with patch.dict(os.environ, env, clear=False):
        s = Settings()
    assert s.db.app_url == "postgresql://user:pass@localhost/db2"


def test_trust_gateway_flat_env() -> None:
    s = _make_settings(JOURNAL_TRUST_GATEWAY="true")
    assert s.auth.trust_gateway is True


def test_short_api_key_raises() -> None:
    with pytest.raises(Exception, match="at least 32 characters"):
        _make_settings(JOURNAL_API_KEY="short")
