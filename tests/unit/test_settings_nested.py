"""Tests verifying nested Settings construction from flat env vars."""

from __future__ import annotations

import os
from unittest.mock import patch

import pydantic
import pytest

from gubbi.config import Settings


def _base_env(**extra: str) -> dict[str, str]:
    base: dict[str, str] = {
        "JOURNAL_DB_APP_URL": "postgresql://user:pass@localhost/db",
        "JOURNAL_API_KEY": "placeholder-not-a-real-key-padding-to-32plus",
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
    assert s.auth.api_key == "placeholder-not-a-real-key-padding-to-32plus"
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
        "JOURNAL_API_KEY": "placeholder-not-a-real-key-padding-to-32plus",
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


# --- api_key_scopes CSV parsing ------------------------------------------
#
# api_key_scopes is the only env-driven complex-type field on AuthConfig
# (sibling of cors_allowed_origins on the cloud-api Settings, which uses
# the same NoDecode + CSV pattern). Operators set JOURNAL_API_KEY_SCOPES
# (legacy flat) or JOURNAL_AUTH__API_KEY_SCOPES (nested) as a comma-
# separated string. JSON-array shape is rejected loudly so an operator
# pasting a JSON array out of habit gets a clear pointer at boot.


def test_api_key_scopes_csv_string_is_split() -> None:
    """Comma-separated env value parses into a list of scopes."""
    s = _make_settings(JOURNAL_API_KEY_SCOPES="journal:read,journal:write,admin:all")
    assert s.auth.api_key_scopes == ["journal:read", "journal:write", "admin:all"]


def test_api_key_scopes_csv_strips_whitespace() -> None:
    """Whitespace around comma separators is stripped; empty entries dropped."""
    s = _make_settings(JOURNAL_API_KEY_SCOPES=" journal:read , journal:write ,, ")
    assert s.auth.api_key_scopes == ["journal:read", "journal:write"]


def test_api_key_scopes_unset_defaults_to_read_write() -> None:
    """Unset env yields the in-code default ``[journal:read, journal:write]``."""
    s = _make_settings()
    assert s.auth.api_key_scopes == ["journal:read", "journal:write"]


def test_api_key_scopes_rejects_json_array() -> None:
    """JSON-array shape is rejected with a clear operator hint."""
    with pytest.raises(
        pydantic.ValidationError,
        match="comma-separated string.*not a JSON array",
    ):
        _make_settings(JOURNAL_API_KEY_SCOPES='["journal:read","journal:write"]')


def test_api_key_scopes_accepts_newline_separators() -> None:
    """CRLF / LF separators (e.g. from a Doppler multi-line paste) split cleanly."""
    s = _make_settings(JOURNAL_API_KEY_SCOPES="journal:read\r\njournal:write\nadmin:all")
    assert s.auth.api_key_scopes == ["journal:read", "journal:write", "admin:all"]


@pytest.mark.parametrize("raw", ["", "   ", ",,,,"])
def test_api_key_scopes_empty_inputs_yield_empty_list(raw: str) -> None:
    """Empty / whitespace / comma-only inputs collapse to an empty list (default-deny)."""
    s = _make_settings(JOURNAL_API_KEY_SCOPES=raw)
    assert s.auth.api_key_scopes == []


def test_api_key_scopes_via_nested_double_underscore() -> None:
    """Nested env var spelling parses through the same validator."""
    s = _make_settings(**{"JOURNAL_AUTH__API_KEY_SCOPES": "scope:a,scope:b"})
    assert s.auth.api_key_scopes == ["scope:a", "scope:b"]
