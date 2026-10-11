"""Unit tests for the spend-rule table, the policy accessors and the wallet settings."""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from gubbi.config import LLMConfig, get_settings
from gubbi.extraction.intake import Lane
from gubbi.extraction.spend_rules import (
    SPEND_RULES,
    GlobalCeiling,
    WalletLimits,
    global_ceiling,
    limits_for,
    run_after_for,
)

pytestmark = pytest.mark.unit

DAY0 = datetime(2026, 1, 1, 12, tzinfo=UTC)


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


def test_every_lane_has_a_spend_rule() -> None:
    assert set(SPEND_RULES) == set(Lane)


def test_limits_for_returns_defaults_when_unconfigured(clean_env: pytest.MonkeyPatch) -> None:
    assert limits_for(uuid4()) == WalletLimits(
        monthly_amount=5_000_000, background_reserve=1_000_000, running_cap=3
    )


def test_limits_for_returns_configured_values(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("JOURNAL_LLM_WALLET_MONTHLY_AMOUNT", "11")
    clean_env.setenv("JOURNAL_LLM_WALLET_BACKGROUND_RESERVE", "12")
    clean_env.setenv("JOURNAL_EXTRACTION_RUNNING_JOB_CAP", "13")

    assert limits_for(uuid4()) == WalletLimits(
        monthly_amount=11, background_reserve=12, running_cap=13
    )


def test_global_ceiling_is_none_when_unset(clean_env: pytest.MonkeyPatch) -> None:
    assert global_ceiling() is None


def test_global_ceiling_reads_ceiling_and_background_ratio(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("JOURNAL_EXTRACTION_GLOBAL_MONTHLY_TOKEN_CEILING", "15000")
    clean_env.setenv("JOURNAL_EXTRACTION_BACKFILL_STOP_RATIO", "0.5")

    assert global_ceiling() == GlobalCeiling(tokens=15000, background_ratio=0.5)


@pytest.mark.parametrize(
    ("lane", "expected"),
    [
        pytest.param(Lane.PASTE, DAY0, id="paste-on-save"),
        pytest.param(Lane.BACKFILL, DAY0, id="backfill-soon-after-upload"),
        pytest.param(Lane.LIVE, datetime(2026, 1, 2, 3, tzinfo=UTC), id="live-next-night"),
    ],
)
def test_run_after_follows_the_lane_schedule(lane: Lane, expected: datetime) -> None:
    assert run_after_for(lane, DAY0, 3) == expected


@pytest.mark.parametrize(
    ("env_var", "raw"),
    [
        ("JOURNAL_EXTRACTION_BACKFILL_STOP_RATIO", "-0.01"),
        ("JOURNAL_EXTRACTION_BACKFILL_STOP_RATIO", "1.01"),
        ("JOURNAL_EXTRACTION_NIGHTLY_HOUR", "-1"),
        ("JOURNAL_EXTRACTION_NIGHTLY_HOUR", "24"),
        ("JOURNAL_EXTRACTION_ONBOARDING_WINDOW_DAYS", "-1"),
        ("JOURNAL_EXTRACTION_GLOBAL_MONTHLY_TOKEN_CEILING", "0"),
        ("JOURNAL_LLM_WALLET_MONTHLY_AMOUNT", "-1"),
        ("JOURNAL_LLM_WALLET_BACKGROUND_RESERVE", "-1"),
        ("JOURNAL_EXTRACTION_RUNNING_JOB_CAP", "0"),
        ("JOURNAL_LLM_WEIGHT_INPUT", "-0.1"),
        ("JOURNAL_LLM_WEIGHT_CACHED_INPUT", "-0.1"),
        ("JOURNAL_LLM_WEIGHT_OUTPUT", "-0.1"),
        *(
            (f"JOURNAL_LLM_WEIGHT_{weight}", raw)
            for weight in ("INPUT", "CACHED_INPUT", "OUTPUT")
            for raw in ("inf", "-inf", "nan")
        ),
    ],
)
def test_out_of_range_setting_is_rejected(
    clean_env: pytest.MonkeyPatch, env_var: str, raw: str
) -> None:
    clean_env.setenv(env_var, raw)

    with pytest.raises(ValidationError, match=env_var):
        LLMConfig()


@pytest.mark.parametrize(
    ("env_var", "raw", "field", "expected"),
    [
        ("JOURNAL_EXTRACTION_BACKFILL_STOP_RATIO", "0", "backfill_stop_ratio", 0.0),
        ("JOURNAL_EXTRACTION_BACKFILL_STOP_RATIO", "1", "backfill_stop_ratio", 1.0),
        ("JOURNAL_EXTRACTION_NIGHTLY_HOUR", "0", "nightly_window_hour", 0),
        ("JOURNAL_EXTRACTION_NIGHTLY_HOUR", "23", "nightly_window_hour", 23),
        ("JOURNAL_EXTRACTION_RUNNING_JOB_CAP", "1", "running_job_cap", 1),
        ("JOURNAL_LLM_WALLET_MONTHLY_AMOUNT", "0", "wallet_monthly_amount", 0),
    ],
)
def test_boundary_setting_is_accepted(
    clean_env: pytest.MonkeyPatch, env_var: str, raw: str, field: str, expected: object
) -> None:
    clean_env.setenv(env_var, raw)

    assert getattr(LLMConfig(), field) == expected


def test_default_weights(clean_env: pytest.MonkeyPatch) -> None:
    config = LLMConfig()

    assert (config.weight_input, config.weight_cached_input, config.weight_output) == (
        1.0,
        0.1,
        4.0,
    )


def test_budgets_on_without_global_ceiling_is_rejected(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("JOURNAL_LLM_BUDGET_ENABLED", "true")

    with pytest.raises(
        ValidationError, match="JOURNAL_EXTRACTION_GLOBAL_MONTHLY_TOKEN_CEILING is required"
    ):
        LLMConfig()
