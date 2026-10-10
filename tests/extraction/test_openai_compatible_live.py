"""Contract test for OpenAICompatibleProvider against a live OpenAI-compatible endpoint.

Skipped unless ``JOURNAL_LLM_BASE_URL`` and ``JOURNAL_LLM_MODEL`` are set
(``JOURNAL_LLM_API_KEY`` defaults to a placeholder for keyless gateways).
Each test records the HTTP exchange so it can compare the adapter's
numbers against the raw wire body the endpoint returned.
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx2
import pytest

from gubbi.extraction.llm.openai_compatible import OpenAICompatibleProvider
from gubbi.extraction.llm.provider import LLMMessage, LLMTruncatedError

_BASE_URL = os.environ.get("JOURNAL_LLM_BASE_URL", "")
_MODEL = os.environ.get("JOURNAL_LLM_MODEL", "")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (_BASE_URL and _MODEL),
        reason="set JOURNAL_LLM_BASE_URL and JOURNAL_LLM_MODEL to run the live LLM contract test",
    ),
]

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"topic": {"type": "string"}, "summary": {"type": "string"}},
    "required": ["topic", "summary"],
    "additionalProperties": False,
}
_MESSAGES: list[LLMMessage] = [
    {"role": "user", "content": "I decided to buy a Toyota Corolla today."},
]
_SYSTEM_PROMPT = "Name the topic of the conversation and summarize it in one sentence."


class _WireLog:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.responses: list[dict[str, Any]] = []

    async def on_request(self, request: httpx2.Request) -> None:
        self.requests.append(json.loads(request.content))

    async def on_response(self, response: httpx2.Response) -> None:
        await response.aread()
        self.responses.append(response.json())


def _provider(wire: _WireLog, max_output_tokens: int) -> OpenAICompatibleProvider:
    client = httpx2.AsyncClient(
        event_hooks={"request": [wire.on_request], "response": [wire.on_response]},
    )
    return OpenAICompatibleProvider(
        base_url=_BASE_URL,
        api_key=os.environ.get("JOURNAL_LLM_API_KEY") or "unused",
        model=_MODEL,
        input_price_per_mtok=1.0,
        output_price_per_mtok=5.0,
        max_output_tokens=max_output_tokens,
        timeout=120.0,
        http_client=client,
    )


def _wire_counts(usage: dict[str, Any]) -> tuple[int, int, int, int]:
    cached = usage.get("cache_read_input_tokens")
    if cached is None:
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    reasoning = usage.get("reasoning_tokens")
    if reasoning is None:
        reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
    return usage["prompt_tokens"], usage["completion_tokens"], cached, reasoning


async def test_stop_preserves_wire_usage_in_one_request() -> None:
    wire = _WireLog()

    response = await _provider(wire, max_output_tokens=512).complete(
        _MESSAGES, _SYSTEM_PROMPT, _SCHEMA
    )

    assert len(wire.requests) == 1
    assert response.finish_reason == "stop"
    assert isinstance(response.content, dict)
    assert set(response.content) == {"topic", "summary"}
    assert (
        response.input_tokens,
        response.output_tokens,
        response.cached_input_tokens,
        response.reasoning_tokens,
    ) == _wire_counts(wire.responses[0]["usage"])


async def test_strict_schema_reaches_the_wire() -> None:
    wire = _WireLog()

    await _provider(wire, max_output_tokens=512).complete(_MESSAGES, _SYSTEM_PROMPT, _SCHEMA)

    response_format = wire.requests[0]["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["schema"] == _SCHEMA


async def test_length_raises_truncated_with_wire_usage() -> None:
    wire = _WireLog()

    with pytest.raises(LLMTruncatedError) as excinfo:
        await _provider(wire, max_output_tokens=5).complete(_MESSAGES, _SYSTEM_PROMPT, _SCHEMA)

    assert len(wire.requests) == 1
    assert excinfo.value.usage is not None
    usage = excinfo.value.usage
    assert (usage.input, usage.output, usage.cached_input, usage.reasoning) == _wire_counts(
        wire.responses[0]["usage"]
    )


async def test_repeated_long_prompt_preserves_nonzero_cached_count() -> None:
    long_prompt = (
        _SYSTEM_PROMPT
        + " "
        + " ".join(
            f"Guideline {i}: keep the summary short and factual about item {i}." for i in range(900)
        )
    )
    wire = _WireLog()
    provider = _provider(wire, max_output_tokens=1024)

    await provider.complete(_MESSAGES, long_prompt, _SCHEMA)
    response = await provider.complete(_MESSAGES, long_prompt, _SCHEMA)

    assert len(wire.requests) == 2
    assert response.cached_input_tokens > 0
    assert (
        response.input_tokens,
        response.output_tokens,
        response.cached_input_tokens,
        response.reasoning_tokens,
    ) == _wire_counts(wire.responses[1]["usage"])
