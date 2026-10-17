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
    "JOURNAL_LLM_EFFORT",
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
    assert config.effort == ""
    assert config.input_price_per_mtok == 1.00
    assert config.output_price_per_mtok == 5.00
    assert config.extraction_mode == "single"
    assert config.extraction_triage is False


@pytest.mark.unit
@pytest.mark.parametrize(
    ("env_var", "raw", "field", "expected"),
    [
        ("JOURNAL_LLM_PROVIDER", "fake", "provider", "fake"),
        ("JOURNAL_LLM_PROVIDER", "anthropic", "provider", "anthropic"),
        ("JOURNAL_LLM_EFFORT", "low", "effort", "low"),
        ("JOURNAL_LLM_EFFORT", "max", "effort", "max"),
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
        ("JOURNAL_LLM_EFFORT", "extreme"),
        ("JOURNAL_LLM_EFFORT", "LOW"),
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


@pytest.mark.unit
@pytest.mark.parametrize(
    "url",
    [
        pytest.param("https://llm.example.test/v1", id="https-public"),
        pytest.param("https://203.0.113.7/v1", id="https-public-ip"),
        pytest.param("http://localhost:11434/v1", id="http-localhost"),
        pytest.param("http://127.0.0.1:8080/v1", id="http-loopback-v4"),
        pytest.param("http://127.8.9.10/v1", id="http-loopback-v4-range"),
        pytest.param("http://[::1]:8080/v1", id="http-loopback-v6"),
        pytest.param("http://10.1.2.3/v1", id="http-rfc1918-10"),
        pytest.param("http://172.16.0.5/v1", id="http-rfc1918-172-low"),
        pytest.param("http://172.31.255.254/v1", id="http-rfc1918-172-high"),
        pytest.param("http://192.168.1.20:8000/v1", id="http-rfc1918-192"),
        pytest.param("http://[fd12:3456::1]/v1", id="http-ula"),
        pytest.param("", id="unset"),
    ],
)
def test_base_url_accepted(clean_llm_env: pytest.MonkeyPatch, url: str) -> None:
    clean_llm_env.setenv("JOURNAL_LLM_BASE_URL", url)

    assert LLMConfig().base_url == url


_CREDENTIAL = "s3cret-pass"


@pytest.mark.unit
@pytest.mark.parametrize(
    "url",
    [
        pytest.param("http://llm.example.test/v1", id="http-public-name"),
        pytest.param("http://203.0.113.7/v1", id="http-public-ip"),
        pytest.param("http://172.32.0.1/v1", id="http-just-outside-rfc1918-172"),
        pytest.param("http://169.254.169.254/v1", id="http-link-local"),
        pytest.param("http://[2001:db8::1]/v1", id="http-public-v6"),
        pytest.param("http://localhost.example.test/v1", id="http-localhost-lookalike"),
        pytest.param("ftp://llm.example.test/v1", id="other-scheme"),
        pytest.param("llm.example.test/v1", id="no-scheme"),
        pytest.param("https:///v1", id="no-host"),
        pytest.param(f"http://user:{_CREDENTIAL}@llm.example.test/v1", id="http-with-userinfo"),
    ],
)
def test_base_url_rejected_without_echoing_it(clean_llm_env: pytest.MonkeyPatch, url: str) -> None:
    clean_llm_env.setenv("JOURNAL_LLM_BASE_URL", url)

    with pytest.raises(ValidationError, match="JOURNAL_LLM_BASE_URL") as caught:
        LLMConfig()

    assert url not in str(caught.value)
    assert _CREDENTIAL not in str(caught.value)


@pytest.mark.unit
@pytest.mark.parametrize("provider", ["fake", "FAKE"], ids=["lower", "upper"])
def test_base_url_unchecked_for_fake_provider(
    clean_llm_env: pytest.MonkeyPatch, provider: str
) -> None:
    clean_llm_env.setenv("JOURNAL_LLM_PROVIDER", provider)
    clean_llm_env.setenv("JOURNAL_LLM_BASE_URL", "http://llm.example.test/v1")

    assert LLMConfig().base_url == "http://llm.example.test/v1"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("url", "accepted"),
    [
        pytest.param("https://llm.example.test", True, id="https-accepted"),
        pytest.param("http://llm.example.test", False, id="http-public-rejected"),
    ],
)
def test_base_url_checked_for_anthropic_provider(
    clean_llm_env: pytest.MonkeyPatch, url: str, accepted: bool
) -> None:
    clean_llm_env.setenv("JOURNAL_LLM_PROVIDER", "anthropic")
    clean_llm_env.setenv("JOURNAL_LLM_BASE_URL", url)

    if accepted:
        assert LLMConfig().base_url == url
    else:
        with pytest.raises(ValidationError, match="JOURNAL_LLM_BASE_URL"):
            LLMConfig()
