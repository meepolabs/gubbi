"""Tests for the extraction package.

Covers:
  - LLMProvider Protocol runtime isinstance checks with mock provider.
  - Health endpoint returns 200.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from starlette.testclient import TestClient

from gubbi.extraction.health import app as health_app
from gubbi.extraction.llm.provider import LLMMessage, LLMProvider, LLMResponse

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
# Health endpoint
# ---------------------------------------------------------------------------


def test_health_endpoint_returns_200() -> None:
    client = TestClient(health_app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
