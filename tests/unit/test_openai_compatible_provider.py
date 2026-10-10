"""OpenAICompatibleProvider against a real AsyncOpenAI on an httpx2 mock transport.

Response bodies come from ``tests/fixtures/openai_chat_completions.json``,
recorded from an OpenAI-compatible gateway fronting an Anthropic model (ids,
timestamps and model name replaced). That gateway reports cached input and
reasoning tokens as top-level usage fields, which is the shape under test.
"""

from __future__ import annotations

import copy
import json
import pathlib
from collections.abc import Callable
from typing import Any

import httpx2
import pytest

from gubbi.extraction.llm.openai_compatible import OpenAICompatibleProvider
from gubbi.extraction.llm.provider import (
    LLMMessage,
    LLMPermanentError,
    LLMProviderError,
    LLMRateLimitError,
    LLMRefusalError,
    LLMSchemaError,
    LLMTimeoutError,
    LLMTransientError,
    LLMTruncatedError,
    LLMUsage,
)
from tests.fixtures.log_capture import LogCapture

_FIXTURES = json.loads(
    (pathlib.Path(__file__).parents[1] / "fixtures" / "openai_chat_completions.json").read_text()
)

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"topic": {"type": "string"}, "summary": {"type": "string"}},
    "required": ["topic", "summary"],
    "additionalProperties": False,
}
_MESSAGES: list[LLMMessage] = [{"role": "user", "content": "I bought a car today."}]
_REFUSAL_TEXT = "REFUSAL-SENTINEL I cannot help with that"
_OUTPUT_SENTINEL = "OUTPUT-SENTINEL"


class _Recorder:
    def __init__(self, respond: Callable[[httpx2.Request], httpx2.Response]) -> None:
        self.requests: list[dict[str, Any]] = []
        self._respond = respond

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(json.loads(request.content))
        return self._respond(request)


def _json_response(body: dict[str, Any], status: int = 200) -> Callable[..., httpx2.Response]:
    def respond(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(status, json=body)

    return respond


def _provider(recorder: _Recorder) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        base_url="https://llm.example.test/v1",
        api_key="test-key",
        model="fixture-model",
        input_price_per_mtok=1.0,
        output_price_per_mtok=5.0,
        max_output_tokens=512,
        timeout=5.0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(recorder)),
    )


def _stop_body() -> dict[str, Any]:
    return copy.deepcopy(_FIXTURES["stop_cached"])


def _refusal_body() -> dict[str, Any]:
    body = _stop_body()
    body["choices"][0]["message"] = {
        "role": "assistant",
        "content": None,
        "refusal": _REFUSAL_TEXT,
    }
    body["choices"][0]["finish_reason"] = "stop"
    return body


def _content_filter_body() -> dict[str, Any]:
    body = _stop_body()
    body["choices"][0]["message"] = {"role": "assistant", "content": _OUTPUT_SENTINEL}
    body["choices"][0]["finish_reason"] = "content_filter"
    return body


def _length_body() -> dict[str, Any]:
    body = copy.deepcopy(_FIXTURES["length"])
    body["choices"][0]["message"]["content"] = '{"summary": "' + _OUTPUT_SENTINEL
    return body


def _wire_usage(body: dict[str, Any]) -> LLMUsage:
    usage = body["usage"]
    return LLMUsage(
        input=usage["prompt_tokens"],
        output=usage["completion_tokens"],
        cached_input=usage["cache_read_input_tokens"],
        reasoning=usage["reasoning_tokens"],
    )


async def test_usage_counts_equal_wire_values() -> None:
    body = _stop_body()
    recorder = _Recorder(_json_response(body))

    response = await _provider(recorder).complete(_MESSAGES, "system prompt", _SCHEMA)

    wire = _wire_usage(body)
    assert wire.cached_input > 0
    assert wire.reasoning > 0
    assert (
        response.input_tokens,
        response.output_tokens,
        response.cached_input_tokens,
        response.reasoning_tokens,
    ) == (wire.input, wire.output, wire.cached_input, wire.reasoning)


def _cached_top_level_only(usage: dict[str, Any]) -> None:
    usage.pop("prompt_tokens_details", None)
    usage["cache_read_input_tokens"] = 4321


def _cached_standard_field_only(usage: dict[str, Any]) -> None:
    usage.pop("cache_read_input_tokens", None)
    usage["prompt_tokens_details"] = {"cached_tokens": 4321}


@pytest.mark.parametrize(
    "shape_usage",
    [
        pytest.param(_cached_top_level_only, id="top-level-cache-read-field"),
        pytest.param(_cached_standard_field_only, id="prompt-tokens-details"),
    ],
)
async def test_cached_tokens_read_from_either_wire_field(
    shape_usage: Callable[[dict[str, Any]], None],
) -> None:
    body = _stop_body()
    shape_usage(body["usage"])
    recorder = _Recorder(_json_response(body))

    response = await _provider(recorder).complete(_MESSAGES, "system prompt", _SCHEMA)

    assert response.cached_input_tokens == 4321


async def test_structured_response_returns_parsed_object_and_stop() -> None:
    body = _stop_body()
    recorder = _Recorder(_json_response(body))

    response = await _provider(recorder).complete(_MESSAGES, "system prompt", _SCHEMA)

    assert response.content == json.loads(body["choices"][0]["message"]["content"])
    assert response.finish_reason == "stop"
    assert response.attempts == 1


async def test_request_carries_strict_json_schema_in_one_http_call() -> None:
    recorder = _Recorder(_json_response(_stop_body()))

    await _provider(recorder).complete(_MESSAGES, "system prompt", _SCHEMA)

    assert len(recorder.requests) == 1
    response_format = recorder.requests[0]["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["schema"] == _SCHEMA


async def test_request_sends_system_prompt_messages_and_token_cap() -> None:
    recorder = _Recorder(_json_response(_stop_body()))

    await _provider(recorder).complete(_MESSAGES, "system prompt", _SCHEMA)

    sent = recorder.requests[0]
    assert sent["model"] == "fixture-model"
    assert sent["max_completion_tokens"] == 512
    assert sent["messages"] == [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "I bought a car today."},
    ]


@pytest.mark.parametrize(
    ("body_factory", "error_class"),
    [
        pytest.param(_length_body, LLMTruncatedError, id="length"),
        pytest.param(_refusal_body, LLMRefusalError, id="refusal-field"),
        pytest.param(_content_filter_body, LLMRefusalError, id="content-filter"),
    ],
)
async def test_unusable_finish_raises_with_usage_after_one_request(
    body_factory: Callable[[], dict[str, Any]],
    error_class: type[LLMProviderError],
) -> None:
    body = body_factory()
    recorder = _Recorder(_json_response(body))

    with pytest.raises(error_class) as excinfo:
        await _provider(recorder).complete(_MESSAGES, "system prompt", _SCHEMA)

    assert excinfo.value.usage == _wire_usage(body)
    assert len(recorder.requests) == 1


@pytest.mark.parametrize(
    "body_factory",
    [
        pytest.param(_length_body, id="length"),
        pytest.param(_refusal_body, id="refusal-field"),
        pytest.param(_content_filter_body, id="content-filter"),
    ],
)
async def test_unusable_finish_keeps_model_text_out_of_errors_and_logs(
    body_factory: Callable[[], dict[str, Any]],
    log_capture: LogCapture,
) -> None:
    recorder = _Recorder(_json_response(body_factory()))

    with pytest.raises(LLMProviderError) as excinfo:
        await _provider(recorder).complete(_MESSAGES, "system prompt", _SCHEMA)

    assert "llm_response_unusable" in log_capture.text
    for leaked in (_REFUSAL_TEXT, _OUTPUT_SENTINEL):
        assert leaked not in str(excinfo.value)
        assert leaked not in repr(excinfo.value.args)
        assert leaked not in log_capture.text


async def test_non_json_output_raises_schema_error_with_usage() -> None:
    body = _stop_body()
    body["choices"][0]["message"]["content"] = _OUTPUT_SENTINEL
    recorder = _Recorder(_json_response(body))

    with pytest.raises(LLMSchemaError) as excinfo:
        await _provider(recorder).complete(_MESSAGES, "system prompt", _SCHEMA)

    assert excinfo.value.usage == _wire_usage(body)
    assert _OUTPUT_SENTINEL not in str(excinfo.value)


async def test_text_mode_returns_text_without_response_format() -> None:
    body = _stop_body()
    body["choices"][0]["message"]["content"] = "plain answer"
    recorder = _Recorder(_json_response(body))

    response = await _provider(recorder).complete(_MESSAGES, "system prompt")

    assert response.content == "plain answer"
    assert "response_format" not in recorder.requests[0]


@pytest.mark.parametrize(
    ("status", "error_class"),
    [
        pytest.param(400, LLMPermanentError, id="400"),
        pytest.param(401, LLMPermanentError, id="401"),
        pytest.param(408, LLMTransientError, id="408"),
        pytest.param(429, LLMRateLimitError, id="429"),
        pytest.param(503, LLMTransientError, id="503"),
    ],
)
async def test_http_status_maps_to_error_class_after_one_request(
    status: int, error_class: type[LLMProviderError]
) -> None:
    error_body = {"error": {"message": _OUTPUT_SENTINEL, "type": "error"}}
    recorder = _Recorder(_json_response(error_body, status=status))

    with pytest.raises(LLMProviderError) as excinfo:
        await _provider(recorder).complete(_MESSAGES, "system prompt", _SCHEMA)

    assert type(excinfo.value) is error_class
    assert _OUTPUT_SENTINEL not in str(excinfo.value)
    assert len(recorder.requests) == 1


async def test_transport_timeout_maps_to_timeout_error() -> None:
    def respond(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ReadTimeout("timed out", request=request)

    recorder = _Recorder(respond)

    with pytest.raises(LLMTimeoutError):
        await _provider(recorder).complete(_MESSAGES, "system prompt", _SCHEMA)

    assert len(recorder.requests) == 1


async def test_connection_error_maps_to_transient_not_timeout() -> None:
    def respond(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("refused", request=request)

    recorder = _Recorder(respond)

    with pytest.raises(LLMTransientError) as excinfo:
        await _provider(recorder).complete(_MESSAGES, "system prompt", _SCHEMA)

    assert type(excinfo.value) is LLMTransientError


def test_estimate_cost_uses_configured_prices() -> None:
    provider = _provider(_Recorder(_json_response(_stop_body())))

    assert provider.estimate_cost_cents(1_000_000, 1_000_000) == pytest.approx(600.0)
