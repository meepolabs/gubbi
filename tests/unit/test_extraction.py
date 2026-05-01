"""Tests for the extraction package.

Covers:
  - LLMProvider ABC cannot be instantiated; mock concrete provider works.
  - ExtractionService builds correct prompts and parses structured output.
  - AnthropicProvider constructs correct API call (mock anthropic client).
  - Health endpoint returns 200.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.testclient import TestClient

from journalctl.config import LLMConfig
from journalctl.extraction.health import app as health_app
from journalctl.extraction.llm import AnthropicProvider
from journalctl.extraction.llm.provider import LLMProvider, LLMResponse
from journalctl.extraction.service import (
    CategorizationResult,
    ExtractedEntry,
    ExtractionService,
)

# ---------------------------------------------------------------------------
# LLMProvider ABC
# ---------------------------------------------------------------------------


def test_llm_provider_abc_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        LLMProvider()  # type: ignore[abstract]


def test_mock_concrete_provider_works() -> None:
    class MockProvider(LLMProvider):
        async def complete(
            self,
            messages: list[dict],
            system_prompt: str,
            output_schema: dict | None = None,
        ) -> LLMResponse:
            return LLMResponse(content="ok", input_tokens=0, output_tokens=0, model="mock")

        def estimate_cost_cents(self, input_tokens: int, output_tokens: int) -> float:
            return 0.0

    provider = MockProvider()
    assert isinstance(provider, LLMProvider)
    assert provider.estimate_cost_cents(100, 50) == 0.0


# ---------------------------------------------------------------------------
# ExtractionService
# ---------------------------------------------------------------------------


class _MockCategorizeProvider(LLMProvider):
    """Returns a structured categorization response."""

    async def complete(
        self,
        messages: list[dict],
        system_prompt: str,
        output_schema: dict | None = None,
    ) -> LLMResponse:
        return LLMResponse(
            content={
                "topic_path": "health/fitness",
                "topic_title": "Fitness Routine",
                "summary": "User discussed morning workout plan",
                "confidence": 0.92,
            },
            input_tokens=10,
            output_tokens=20,
            model="mock",
        )

    def estimate_cost_cents(self, input_tokens: int, output_tokens: int) -> float:
        return 0.0


class _MockExtractProvider(LLMProvider):
    """Returns a structured extraction response."""

    async def complete(
        self,
        messages: list[dict],
        system_prompt: str,
        output_schema: dict | None = None,
    ) -> LLMResponse:
        return LLMResponse(
            content={
                "entries": [
                    {
                        "content": "Started daily jogging",
                        "reasoning": "New exercise habit worth recording",
                        "tags": ["exercise", "health"],
                        "entry_date": "2026-04-15",
                    },
                    {
                        "content": "Planned gym session",
                        "reasoning": "Weekly commitment",
                        "tags": ["gym", "fitness"],
                    },
                ],
            },
            input_tokens=10,
            output_tokens=30,
            model="mock",
        )

    def estimate_cost_cents(self, input_tokens: int, output_tokens: int) -> float:
        return 0.0


@pytest.mark.asyncio
async def test_extraction_service_categorize() -> None:
    service = ExtractionService(_MockCategorizeProvider())
    result = await service.categorize_conversation(
        [{"role": "user", "content": "I went jogging today"}],
        ["existing/topic"],
    )
    assert isinstance(result, CategorizationResult)
    assert result.topic_path == "health/fitness"
    assert result.topic_title == "Fitness Routine"
    assert result.summary == "User discussed morning workout plan"
    assert result.confidence == 0.92


@pytest.mark.asyncio
async def test_extraction_service_extract_entries() -> None:
    service = ExtractionService(_MockExtractProvider())
    results = await service.extract_entries(
        [{"role": "user", "content": "I started jogging daily"}],
        "health/fitness",
    )
    assert len(results) == 2
    assert isinstance(results[0], ExtractedEntry)
    assert results[0].content == "Started daily jogging"
    assert results[0].reasoning == "New exercise habit worth recording"
    assert results[0].tags == ["exercise", "health"]
    assert results[0].entry_date == "2026-04-15"
    assert results[1].entry_date is None


# ---------------------------------------------------------------------------
# AnthropicProvider — verifies correct API call construction via mock
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_anthropic_provider_calls_api_with_expected_kwargs() -> None:
    """Verify the provider constructs correct API kwargs for the anthropic SDK."""
    with patch("journalctl.extraction.llm.anthropic_provider.AsyncAnthropic") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        # Build a fake response that mimics anthropic SDK's Message
        fake_content_block = MagicMock()
        fake_content_block.text = '{"result": "ok"}'

        fake_response = MagicMock()
        fake_response.content = [fake_content_block]
        fake_response.usage.input_tokens = 5
        fake_response.usage.output_tokens = 10
        fake_response.model = "claude-haiku-4-5-20251001"

        mock_client.messages.create = AsyncMock(return_value=fake_response)

        provider = _make_anthropic_provider()
        result = await provider.complete(
            [{"role": "user", "content": "hello"}],
            "Be helpful",
        )

        # Without output_schema, content is the raw text from the API
        assert result.content == '{"result": "ok"}'
        assert result.input_tokens == 5
        assert result.output_tokens == 10
        assert result.model == "claude-haiku-4-5-20251001"

        mock_client.messages.create.assert_called_once()
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["model"] == "test-model"
        assert len(call_kwargs["system"]) == 1
        assert call_kwargs["system"][0]["type"] == "text"
        assert call_kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}
        # No output_schema -> no tools
        assert "tools" not in call_kwargs


@pytest.mark.asyncio
async def test_anthropic_provider_uses_tool_use_for_structured_output() -> None:
    """When output_schema is provided, the API call should include tools + tool_choice."""
    with patch("journalctl.extraction.llm.anthropic_provider.AsyncAnthropic") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        fake_tool_block = MagicMock()
        fake_tool_block.name = "respond"
        fake_tool_block.input = {"answer": 42}

        fake_response = MagicMock()
        fake_response.content = [fake_tool_block]
        fake_response.usage.input_tokens = 5
        fake_response.usage.output_tokens = 10
        fake_response.model = "claude-haiku-4-5-20251001"

        mock_client.messages.create = AsyncMock(return_value=fake_response)

        provider = _make_anthropic_provider()
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "integer"}},
            "required": ["answer"],
        }
        result = await provider.complete(
            [{"role": "user", "content": "what is 6*7?"}],
            "Answer math questions",
            output_schema=schema,
        )

        assert result.content == {"answer": 42}
        call_kwargs = mock_client.messages.create.call_args[1]
        assert "tools" in call_kwargs
        assert len(call_kwargs["tools"]) == 1
        assert call_kwargs["tools"][0]["name"] == "respond"
        assert call_kwargs["tools"][0]["input_schema"] == schema
        assert call_kwargs["tool_choice"] == {"type": "tool", "name": "respond"}


@pytest.mark.asyncio
async def test_anthropic_provider_retries_on_429() -> None:
    """Verify exponential backoff retry on 429 status."""
    with patch("journalctl.extraction.llm.anthropic_provider.AsyncAnthropic") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        # Must be actual Exception instances so AsyncMock raises them.
        class _RateLimit(Exception):
            status_code = 429

        fake_text_block = MagicMock()
        fake_text_block.text = "ok"
        fake_response = MagicMock()
        fake_response.content = [fake_text_block]
        fake_response.usage.input_tokens = 1
        fake_response.usage.output_tokens = 1
        fake_response.model = "test-model"

        mock_client.messages.create = AsyncMock(
            side_effect=[_RateLimit("too fast"), _RateLimit("too fast"), fake_response]
        )

        provider = _make_anthropic_provider()
        result = await provider.complete(
            [{"role": "user", "content": "hi"}],
            "System prompt",
        )

        assert result.content == "ok"
        assert mock_client.messages.create.call_count == 3


@pytest.mark.asyncio
async def test_anthropic_provider_estimate_cost() -> None:
    with patch("journalctl.extraction.llm.anthropic_provider.AsyncAnthropic"):
        provider = _make_anthropic_provider()
        cost = provider.estimate_cost_cents(1_000_000, 500_000)
        # Haiku: $1.00/M input, $5.00/M output
        # Input: 1M * 1.00 / 1M * 100 = 100 cents
        # Output: 500K * 5.00 / 1M * 100 = 250 cents
        assert cost == 350.0


def _make_anthropic_provider() -> AnthropicProvider:
    """Helper: create an AnthropicProvider and return it.

    The client class is already patched; we just need to ensure
    the init args are fixed for deterministic tests.
    """
    return AnthropicProvider(LLMConfig(api_key="test-api-key", model="test-model"))


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


def test_health_endpoint_returns_200() -> None:
    client = TestClient(health_app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
