"""Unit tests for the extraction limit settings and the ``limits_for`` accessor."""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import timedelta
from uuid import uuid4

import pytest
from pydantic import ValidationError

from gubbi.config import LLMConfig, get_settings
from gubbi.extraction.limits import ExtractionLimits, limits_for


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[pytest.MonkeyPatch]:
    for name in list(os.environ):
        if name.startswith("JOURNAL_"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("JOURNAL_DB_APP_URL", "postgresql://app@host/db")
    monkeypatch.setenv("JOURNAL_API_KEY", "x" * 32)
    monkeypatch.setenv("JOURNAL_OPERATOR_EMAIL", "op@example.com")
    get_settings.cache_clear()
    yield monkeypatch
    get_settings.cache_clear()


@pytest.mark.unit
def test_limits_for_returns_defaults_when_unconfigured(clean_env: pytest.MonkeyPatch) -> None:
    # Act
    limits = limits_for(uuid4())

    # Assert
    assert limits == ExtractionLimits(
        monthly_conversations=200,
        monthly_tokens=4_000_000,
        first_sync_conversations=500,
        first_sync_tokens=10_000_000,
        global_monthly_token_ceiling=None,
        backfill_stop_ratio=0.8,
        nightly_window_hour=3,
        onboarding_window=timedelta(days=7),
    )


@pytest.mark.unit
def test_limits_for_returns_configured_values(clean_env: pytest.MonkeyPatch) -> None:
    # Arrange
    for name, value in {
        "JOURNAL_LLM_BUDGET_ENABLED": "true",
        "JOURNAL_EXTRACTION_MONTHLY_CONVERSATIONS": "11",
        "JOURNAL_EXTRACTION_MONTHLY_TOKENS": "12000",
        "JOURNAL_EXTRACTION_FIRST_SYNC_CONVERSATIONS": "13",
        "JOURNAL_EXTRACTION_FIRST_SYNC_TOKENS": "14000",
        "JOURNAL_EXTRACTION_GLOBAL_MONTHLY_TOKEN_CEILING": "15000",
        "JOURNAL_EXTRACTION_BACKFILL_STOP_RATIO": "0.5",
        "JOURNAL_EXTRACTION_NIGHTLY_HOUR": "22",
        "JOURNAL_EXTRACTION_ONBOARDING_WINDOW_DAYS": "2",
    }.items():
        clean_env.setenv(name, value)

    # Act
    limits = limits_for(uuid4())

    # Assert
    assert limits == ExtractionLimits(
        monthly_conversations=11,
        monthly_tokens=12000,
        first_sync_conversations=13,
        first_sync_tokens=14000,
        global_monthly_token_ceiling=15000,
        backfill_stop_ratio=0.5,
        nightly_window_hour=22,
        onboarding_window=timedelta(days=2),
    )


@pytest.mark.unit
def test_extraction_limits_is_frozen() -> None:
    # Arrange
    limits = ExtractionLimits(
        monthly_conversations=1,
        monthly_tokens=1,
        first_sync_conversations=1,
        first_sync_tokens=1,
        global_monthly_token_ceiling=None,
        backfill_stop_ratio=0.8,
        nightly_window_hour=3,
        onboarding_window=timedelta(days=7),
    )

    # Act + Assert
    with pytest.raises(AttributeError):
        limits.monthly_tokens = 2  # type: ignore[misc]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("env_var", "raw"),
    [
        ("JOURNAL_EXTRACTION_BACKFILL_STOP_RATIO", "-0.01"),
        ("JOURNAL_EXTRACTION_BACKFILL_STOP_RATIO", "1.01"),
        ("JOURNAL_EXTRACTION_NIGHTLY_HOUR", "-1"),
        ("JOURNAL_EXTRACTION_NIGHTLY_HOUR", "24"),
        ("JOURNAL_EXTRACTION_MONTHLY_CONVERSATIONS", "-1"),
        ("JOURNAL_EXTRACTION_MONTHLY_TOKENS", "-1"),
        ("JOURNAL_EXTRACTION_FIRST_SYNC_CONVERSATIONS", "-1"),
        ("JOURNAL_EXTRACTION_FIRST_SYNC_TOKENS", "-1"),
        ("JOURNAL_EXTRACTION_ONBOARDING_WINDOW_DAYS", "-1"),
        ("JOURNAL_EXTRACTION_GLOBAL_MONTHLY_TOKEN_CEILING", "0"),
    ],
)
def test_out_of_range_limit_is_rejected(
    clean_env: pytest.MonkeyPatch, env_var: str, raw: str
) -> None:
    # Arrange
    clean_env.setenv(env_var, raw)

    # Act + Assert
    with pytest.raises(ValidationError, match=env_var):
        LLMConfig()


@pytest.mark.unit
@pytest.mark.parametrize(
    ("env_var", "raw", "field", "expected"),
    [
        ("JOURNAL_EXTRACTION_BACKFILL_STOP_RATIO", "0", "backfill_stop_ratio", 0.0),
        ("JOURNAL_EXTRACTION_BACKFILL_STOP_RATIO", "1", "backfill_stop_ratio", 1.0),
        ("JOURNAL_EXTRACTION_NIGHTLY_HOUR", "0", "nightly_window_hour", 0),
        ("JOURNAL_EXTRACTION_NIGHTLY_HOUR", "23", "nightly_window_hour", 23),
    ],
)
def test_boundary_limit_is_accepted(
    clean_env: pytest.MonkeyPatch, env_var: str, raw: str, field: str, expected: object
) -> None:
    # Arrange
    clean_env.setenv(env_var, raw)

    # Act
    config = LLMConfig()

    # Assert
    assert getattr(config, field) == expected


@pytest.mark.unit
def test_budgets_on_without_global_ceiling_is_rejected(clean_env: pytest.MonkeyPatch) -> None:
    # Arrange
    clean_env.setenv("JOURNAL_LLM_BUDGET_ENABLED", "true")

    # Act + Assert
    with pytest.raises(
        ValidationError, match="JOURNAL_EXTRACTION_GLOBAL_MONTHLY_TOKEN_CEILING is required"
    ):
        LLMConfig()
