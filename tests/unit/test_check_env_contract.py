"""Tests for tools/check_env_contract.py pure functions."""

from __future__ import annotations

import importlib.util
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent.parent.parent / "tools"
spec = importlib.util.spec_from_file_location(
    "check_env_contract", TOOLS_DIR / "check_env_contract.py"
)
lint = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
spec.loader.exec_module(lint)  # type: ignore[union-attr]


def make_settings(
    fields: dict[str, dict],
    prefix: str = "JOURNAL_",
) -> dict:
    """Build a synthetic Settings datastructure."""
    return {"fields": fields, "prefix": prefix}


def make_compose(
    services: dict[str, dict],
) -> dict:
    """Build a synthetic compose datastructure."""
    return dict(services)


# -- helpers ---------------------------------------------------------------


_CLEAN_SETTINGS = make_settings(
    {
        "api_key": {
            "env_var": "JOURNAL_API_KEY",
            "required": True,
            "alias": None,
        },
        "db_app_url": {
            "env_var": "JOURNAL_DB_APP_URL",
            "required": True,
            "alias": None,
        },
        "log_level": {
            "env_var": "JOURNAL_LOG_LEVEL",
            "required": False,
            "alias": None,
        },
    },
)

_CLEAN_COMPOSE = make_compose(
    {
        "my-service": {
            "declared_keys": {
                "JOURNAL_API_KEY",
                "JOURNAL_DB_APP_URL",
                "JOURNAL_LOG_LEVEL",
            },
            "referenced_vars": set(),
        },
    },
)

_BROKEN_COMPOSE = make_compose(
    {
        "my-service": {
            "declared_keys": {
                "JOURNAL_API_KEY",
                # JOURNAL_DB_APP_URL is MISSING
                "JOURNAL_LOG_LEVEL",
            },
            "referenced_vars": set(),
        },
    },
)

_VARSUB_COMPOSE = make_compose(
    {
        "my-service": {
            "declared_keys": {
                "JOURNAL_API_KEY",
                "JOURNAL_DB_APP_URL",
                "JOURNAL_LOG_LEVEL",
            },
            "referenced_vars": {
                "JOURNAL_DB_APP_PASSWORD",
            },
        },
    },
)

_STALE_COMPOSE = make_compose(
    {
        "my-service": {
            "declared_keys": {
                "JOURNAL_API_KEY",
                "JOURNAL_DB_APP_URL",
                "JOURNAL_LOG_LEVEL",
                "JOURNAL_EXTRA_STALE",  # not in Settings
            },
            "referenced_vars": {
                "JOURNAL_DB_APP_PASSWORD",
            },
        },
    },
)

# -- tests -----------------------------------------------------------------


def test_clean_pass() -> None:
    """Script reports no drift when all required fields are declared."""
    drifts, stale = lint.check_env_contract(
        _CLEAN_SETTINGS, _CLEAN_COMPOSE, "my-service", ["docker-compose.yml"]
    )
    assert drifts == []
    assert stale == []


def test_required_field_missing_fails() -> None:
    """Exit code 1 + DRIFT message when a required field has no compose entry."""
    drifts, stale = lint.check_env_contract(
        _CLEAN_SETTINGS, _BROKEN_COMPOSE, "my-service", ["docker-compose.yml"]
    )
    assert len(drifts) == 2  # DRIFT line + Remediation line
    assert "DRIFT: field=db_app_url env=JOURNAL_DB_APP_URL" in drifts[0]
    assert "Remediation:" in drifts[1]
    assert stale == []


def test_var_substitution_counts_as_declared() -> None:
    """${SOME_VAR} in compose values counts as declaring SOME_VAR (not drift)."""
    drifts, stale = lint.check_env_contract(
        _CLEAN_SETTINGS, _VARSUB_COMPOSE, "my-service", ["docker-compose.yml"]
    )
    assert drifts == []
    # JOURNAL_DB_APP_PASSWORD is a ref_var from a DSN value, not a declared_key
    # It should NOT appear as stale.
    assert stale == []


def test_stale_passthrough_warns_not_fails() -> None:
    """A bare passthrough for an unknown env var emits a warning but does NOT
    cause exit 1 (drift is empty)."""
    drifts, stale = lint.check_env_contract(
        _CLEAN_SETTINGS, _STALE_COMPOSE, "my-service", ["docker-compose.yml"]
    )
    assert drifts == []
    assert len(stale) == 1
    assert "JOURNAL_EXTRA_STALE" in stale[0]
    assert "stale passthrough" in stale[0]
