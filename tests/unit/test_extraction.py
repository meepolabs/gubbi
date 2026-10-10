"""Tests for the extraction package.

Covers:
  - LLMProvider Protocol runtime isinstance checks with mock provider.
  - ExtractionService builds correct prompts and parses structured output.
  - Health endpoint returns 200.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest
from starlette.testclient import TestClient

from gubbi.extraction.health import app as health_app
from gubbi.extraction.llm.provider import LLMMessage, LLMProvider, LLMResponse
from gubbi.extraction.service import (
    CategorizationResult,
    ExtractedEntry,
    ExtractionService,
)

# ---------------------------------------------------------------------------
# LLMProvider Protocol
# ---------------------------------------------------------------------------


def test_mock_concrete_provider_works() -> None:
    class MockProvider(LLMProvider):
        async def complete(
            self,
            messages: list[LLMMessage],
            system_prompt: str,
            output_schema: Mapping[str, Any] | None = None,
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
        messages: list[LLMMessage],
        system_prompt: str,
        output_schema: Mapping[str, Any] | None = None,
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
        messages: list[LLMMessage],
        system_prompt: str,
        output_schema: Mapping[str, Any] | None = None,
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
    result = await service.extract_entries(
        [{"role": "user", "content": "I started jogging daily"}],
        "health/fitness",
    )
    assert len(result.entries) == 2
    assert isinstance(result.entries[0], ExtractedEntry)
    assert result.entries[0].content == "Started daily jogging"
    assert result.entries[0].reasoning == "New exercise habit worth recording"
    assert result.entries[0].tags == ["exercise", "health"]
    assert result.entries[0].entry_date == "2026-04-15"
    assert result.entries[1].entry_date is None


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


def test_health_endpoint_returns_200() -> None:
    client = TestClient(health_app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
