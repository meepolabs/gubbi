"""Tests for the MCP Origin allowlist assembled by ``ServerConfig``.

The allowlist guards the /mcp mount against DNS rebinding, so the tests
below pin both directions of the boundary: which operator values are
accepted, and which are rejected loudly instead of landing an inert or
over-wide entry.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pydantic
import pytest

from gubbi.config import DEFAULT_ALLOWED_ORIGINS, ServerConfig


def _server_config(raw: str | None) -> ServerConfig:
    """Build a ServerConfig with JOURNAL_EXTRA_ALLOWED_ORIGINS set (or unset)."""
    env = dict(os.environ)
    env.pop("JOURNAL_EXTRA_ALLOWED_ORIGINS", None)
    if raw is not None:
        env["JOURNAL_EXTRA_ALLOWED_ORIGINS"] = raw
    with patch.dict(os.environ, env, clear=True):
        return ServerConfig()


def test_default_allowlist_is_the_generic_mcp_client_origins() -> None:
    """Unset env yields only the hosted MCP client origins."""
    config = _server_config(None)

    assert config.extra_allowed_origins == []
    assert config.allowed_origins == frozenset({"https://claude.ai", "https://chatgpt.com"})


def test_default_allowlist_carries_no_deployment_specific_host() -> None:
    """Only vendor MCP client origins are defaults; no deployment hostname.

    Exact set equality is the regression guard: any deployment-specific
    host re-added to the default set fails here rather than shipping in a
    self-hosters' allowlist.
    """
    assert frozenset({"https://claude.ai", "https://chatgpt.com"}) == DEFAULT_ALLOWED_ORIGINS


def test_configured_origins_are_added_to_the_allowlist() -> None:
    """A CSV env value parses and joins the effective allowlist."""
    config = _server_config("https://journal.example.com,https://mcp.example.com:8443")

    assert config.allowed_origins == frozenset(
        {
            "https://claude.ai",
            "https://chatgpt.com",
            "https://journal.example.com",
            "https://mcp.example.com:8443",
        }
    )


def test_configured_origins_do_not_replace_the_generic_defaults() -> None:
    """Semantics are ADDITIVE: the hosted MCP clients stay allowed."""
    config = _server_config("https://journal.example.com")

    assert config.allowed_origins >= DEFAULT_ALLOWED_ORIGINS


def test_newline_separated_origins_are_split() -> None:
    """A multi-line secret-store paste splits into separate origins."""
    config = _server_config("https://journal.example.com\r\nhttps://mcp.example.com")

    assert config.extra_allowed_origins == [
        "https://journal.example.com",
        "https://mcp.example.com",
    ]


@pytest.mark.parametrize("raw", ["", "   ", ",,,,"])
def test_empty_value_leaves_the_defaults_untouched(raw: str) -> None:
    """An empty or separator-only value must not widen the allowlist."""
    config = _server_config(raw)

    assert config.extra_allowed_origins == []
    assert config.allowed_origins == DEFAULT_ALLOWED_ORIGINS


def test_json_array_shape_is_rejected() -> None:
    """JSON-array shape fails loudly with an operator hint, not silently."""
    with pytest.raises(pydantic.ValidationError, match=r"comma-separated string.*not a JSON array"):
        _server_config('["https://journal.example.com"]')


@pytest.mark.parametrize(
    "raw",
    [
        "journal.example.com",
        "//journal.example.com",
        "ftp://journal.example.com",
        "https://",
        "https://journal.example.com/mcp",
        "https://journal.example.com?a=b",
        "https://journal.example.com#frag",
        "https://*.example.com",
        "https://user:pw@journal.example.com",
        "https://journal.example.com:notaport",
        "*",
    ],
)
def test_malformed_origin_fails_validation(raw: str) -> None:
    """Anything that is not a bare absolute http(s) origin is rejected."""
    with pytest.raises(pydantic.ValidationError, match="is invalid"):
        _server_config(raw)


def test_one_malformed_entry_rejects_the_whole_value() -> None:
    """A partially-bad list never lands its good half -- fail closed at boot."""
    with pytest.raises(pydantic.ValidationError, match="is invalid"):
        _server_config("https://journal.example.com,not-an-origin")


def test_plain_http_origin_is_accepted() -> None:
    """http is valid for a LAN or reverse-proxied self-host deployment."""
    config = _server_config("http://journal.example.com:8100")

    assert "http://journal.example.com:8100" in config.allowed_origins
