"""Tests for tools/check_env_contract.py pure functions."""

from __future__ import annotations

import importlib.util
import tempfile
import textwrap
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


# -- parse_settings targeted tests --------------------------------------------


def test_parse_settings_raises_on_invalid_syntax() -> None:
    """parse_settings raises SyntaxError when the source file has invalid Python."""
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write("def broken(\n")
        tmp = f.name
    import pytest

    with pytest.raises(SyntaxError):
        lint.parse_settings(tmp)


def test_parse_settings_field_with_default_factory() -> None:
    """Fields with default_factory=... are treated as optional (has_py_default=True)."""
    src = textwrap.dedent("""\
        from pydantic_settings import BaseSettings, SettingsConfigDict
        from pydantic import Field

        class Settings(BaseSettings):
            model_config = SettingsConfigDict(env_prefix="TEST_")
            tags: list[str] = Field(default_factory=list)
    """)
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(src)
        tmp = f.name
    result = lint.parse_settings(tmp)
    assert result["fields"]["tags"]["required"] is False


def test_parse_settings_field_with_validation_alias() -> None:
    """validation_alias inside Field() is used as the env var name."""
    src = textwrap.dedent("""\
        from pydantic_settings import BaseSettings, SettingsConfigDict
        from pydantic import Field

        class Settings(BaseSettings):
            model_config = SettingsConfigDict(env_prefix="TEST_")
            db_url: str = Field(validation_alias="DATABASE_URL")
    """)
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(src)
        tmp = f.name
    result = lint.parse_settings(tmp)
    assert result["fields"]["db_url"]["env_var"] == "DATABASE_URL"
    assert result["fields"]["db_url"]["alias"] == "DATABASE_URL"


# -- _collect_ref_vars_deep targeted tests ------------------------------------


def _make_svc() -> dict:
    return {"declared_keys": set(), "referenced_vars": set()}


def test_collect_ref_vars_deep_nested_dict() -> None:
    """Deeply nested dict structure yields all ${VAR} references."""
    svc = _make_svc()
    obj = {"outer": {"inner": {"deep": "${DEEP_VAR}"}}}
    lint._collect_ref_vars_deep(obj, svc)
    assert "DEEP_VAR" in svc["referenced_vars"]


def test_collect_ref_vars_deep_list_of_dicts() -> None:
    """List containing dicts with variable refs are all collected."""
    svc = _make_svc()
    obj = [{"key": "${VAR_A}"}, {"key": "${VAR_B}"}]
    lint._collect_ref_vars_deep(obj, svc)
    assert "VAR_A" in svc["referenced_vars"]
    assert "VAR_B" in svc["referenced_vars"]


def test_collect_ref_vars_deep_plain_string() -> None:
    """A plain string at the root level is collected correctly."""
    svc = _make_svc()
    lint._collect_ref_vars_deep("${TOP_LEVEL}", svc)
    assert "TOP_LEVEL" in svc["referenced_vars"]


def test_collect_ref_vars_deep_no_refs_leaves_sets_empty() -> None:
    """A structure with no ${VAR} patterns leaves sets unchanged."""
    svc = _make_svc()
    lint._collect_ref_vars_deep({"a": {"b": "no_vars_here"}}, svc)
    assert svc["referenced_vars"] == set()


# -- _find_compose_file_for_service empty-list guard --------------------------


def test_find_compose_file_empty_list_returns_default() -> None:
    """Empty compose_files list returns 'docker-compose.yml' without IndexError."""
    result = lint._find_compose_file_for_service("any-service", [])
    assert result == "docker-compose.yml"


# ---------------------------------------------------------------------------
# B1-T7: Environment Literal cross-repo parity (DEC-094).
# ---------------------------------------------------------------------------


def test_environment_literal_byte_identical(tmp_path: Path) -> None:
    """Two repos carrying the same canonical line produce zero drift."""
    canonical = 'Environment = Literal["dev", "ci", "staging", "production"]'
    a = tmp_path / "gubbi_config.py"
    b = tmp_path / "cloud_config.py"
    a.write_text(f"from typing import Literal\n\n{canonical}\n\nclass S: ...\n")
    b.write_text(f"from typing import Literal\n\n{canonical}\n\nclass S: ...\n")

    drifts = lint.check_environment_literal_parity(str(a), str(b))

    assert drifts == []


def test_environment_literal_drift_detected(tmp_path: Path) -> None:
    """A literal that differs between the two repos surfaces a DRIFT message."""
    a = tmp_path / "gubbi_config.py"
    b = tmp_path / "cloud_config.py"
    a.write_text(
        "from typing import Literal\n\n"
        'Environment = Literal["dev", "ci", "staging", "production"]\n'
    )
    b.write_text("from typing import Literal\n\n" 'Environment = Literal["dev", "prod"]\n')

    drifts = lint.check_environment_literal_parity(str(a), str(b))

    assert len(drifts) == 1
    assert "DRIFT" in drifts[0]
    assert "differs" in drifts[0]
    assert "DEC-094" in drifts[0]


def test_environment_literal_parity_missing_path_hard_fails(tmp_path: Path) -> None:
    """A non-existent cross-repo config path is itself a drift signal.

    DEC-094 rule 2 ("drift fails CI") -- the parity check must NOT silently
    skip when the cross-repo path does not resolve; it must surface a DRIFT
    message so the CI invocation exits non-zero.
    """
    # Arrange: real path on the gubbi side, missing path on the cloud side.
    a = tmp_path / "gubbi_config.py"
    a.write_text(
        "from typing import Literal\n\n"
        'Environment = Literal["dev", "ci", "staging", "production"]\n'
    )
    missing = tmp_path / "does_not_exist.py"

    # Act
    drifts = lint.check_environment_literal_parity(str(a), str(missing))

    # Assert
    assert len(drifts) == 1
    assert "DRIFT" in drifts[0]
    assert "cross-repo config path not found" in drifts[0]
    assert str(missing) in drifts[0]
