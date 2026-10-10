"""Unit tests for the extraction LLM settings (``LLMConfig``)."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from gubbi.config import LLMConfig

_LLM_ENV_VARS = (
    "JOURNAL_LLM_PROVIDER",
    "JOURNAL_LLM_BASE_URL",
    "JOURNAL_LLM_MODEL",
    "JOURNAL_LLM_API_KEY",
    "JOURNAL_LLM_INPUT_PRICE_PER_MTOK",
    "JOURNAL_LLM_OUTPUT_PRICE_PER_MTOK",
    "JOURNAL_EXTRACTION_MODE",
    "JOURNAL_EXTRACTION_TRIAGE",
)


@pytest.fixture
def clean_llm_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for name in _LLM_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


@pytest.mark.unit
def test_defaults_select_openai_compatible_single_call_without_triage(
    clean_llm_env: pytest.MonkeyPatch,
) -> None:
    config = LLMConfig()

    assert config.provider == "openai_compatible"
    assert config.base_url == ""
    assert config.input_price_per_mtok == 1.00
    assert config.output_price_per_mtok == 5.00
    assert config.extraction_mode == "single"
    assert config.extraction_triage is False


@pytest.mark.unit
@pytest.mark.parametrize(
    ("env_var", "raw", "field", "expected"),
    [
        ("JOURNAL_LLM_PROVIDER", "fake", "provider", "fake"),
        (
            "JOURNAL_LLM_BASE_URL",
            "https://llm.example.test/v1",
            "base_url",
            "https://llm.example.test/v1",
        ),
        ("JOURNAL_LLM_INPUT_PRICE_PER_MTOK", "0.4", "input_price_per_mtok", 0.4),
        ("JOURNAL_LLM_OUTPUT_PRICE_PER_MTOK", "1.6", "output_price_per_mtok", 1.6),
        ("JOURNAL_EXTRACTION_MODE", "two_call", "extraction_mode", "two_call"),
        ("JOURNAL_EXTRACTION_TRIAGE", "true", "extraction_triage", True),
    ],
)
def test_env_var_overrides_field(
    clean_llm_env: pytest.MonkeyPatch,
    env_var: str,
    raw: str,
    field: str,
    expected: Any,
) -> None:
    clean_llm_env.setenv(env_var, raw)

    config = LLMConfig()

    assert getattr(config, field) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    ("env_var", "raw"),
    [
        ("JOURNAL_EXTRACTION_MODE", "three_call"),
        ("JOURNAL_LLM_INPUT_PRICE_PER_MTOK", "-1"),
        ("JOURNAL_LLM_OUTPUT_PRICE_PER_MTOK", "-0.01"),
    ],
)
def test_invalid_value_is_rejected(
    clean_llm_env: pytest.MonkeyPatch,
    env_var: str,
    raw: str,
) -> None:
    clean_llm_env.setenv(env_var, raw)

    with pytest.raises(ValidationError, match=env_var):
        LLMConfig()
