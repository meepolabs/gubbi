"""OpenAICompatibleProvider against a real AsyncOpenAI on an httpx2 mock transport.

Response bodies come from ``tests/fixtures/openai_chat_completions.json``,
recorded from an OpenAI-compatible gateway fronting an Anthropic model (ids,
timestamps and model name replaced). That gateway reports cached input and
reasoning tokens as top-level usage fields, which is the shape under test.
"""

from __future__ import annotations

import copy
import email.utils
import json
import pathlib
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx2
import pytest

from gubbi.constants import LLM_BACKOFF_CAP_SECS, LLM_MAX_ATTEMPTS
from gubbi.extraction.llm import openai_compatible
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
        self.authorizations: list[str | None] = []
        self._respond = respond

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(json.loads(request.content))
        self.authorizations.append(request.headers.get("authorization"))
        return self._respond(request)


def _json_response(body: dict[str, Any], status: int = 200) -> Callable[..., httpx2.Response]:
    def respond(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(status, json=body)

    return respond


class _SleepLog:
    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


def _sequence(*responders: Callable[[httpx2.Request], httpx2.Response]) -> _Recorder:
    """Answer the n-th request with the n-th responder; the last one repeats."""
    queue: Iterator[Callable[[httpx2.Request], httpx2.Response]] = iter(responders)
    current = responders[-1]

    def respond(request: httpx2.Request) -> httpx2.Response:
        nonlocal current
        current = next(queue, current)
        return current(request)

    return _Recorder(respond)


def _status(status: int, headers: dict[str, str] | None = None) -> Callable[..., httpx2.Response]:
    def respond(_request: httpx2.Request) -> httpx2.Response:
        body = {"error": {"message": _OUTPUT_SENTINEL, "type": "error"}}
        return httpx2.Response(status, json=body, headers=headers)

    return respond


def _provider(
    recorder: _Recorder, sleep: _SleepLog | None = None, api_key: str = "test-key"
) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        base_url="https://llm.example.test/v1",
        api_key=api_key,
        model="fixture-model",
        input_price_per_mtok=1.0,
        output_price_per_mtok=5.0,
        max_output_tokens=512,
        timeout=5.0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(recorder)),
        sleep=sleep or _SleepLog(),
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


@pytest.mark.parametrize("status", [400, 401, 404, 422])
async def test_permanent_http_status_raises_after_one_request(status: int) -> None:
    recorder = _Recorder(_status(status))
    sleep = _SleepLog()

    with pytest.raises(LLMProviderError) as excinfo:
        await _provider(recorder, sleep).complete(_MESSAGES, "system prompt", _SCHEMA)

    assert type(excinfo.value) is LLMPermanentError
    assert _OUTPUT_SENTINEL not in str(excinfo.value)
    assert len(recorder.requests) == 1
    assert sleep.delays == []


@pytest.mark.parametrize(
    ("status", "error_class"),
    [
        pytest.param(408, LLMTransientError, id="408"),
        pytest.param(429, LLMRateLimitError, id="429"),
        pytest.param(500, LLMTransientError, id="500"),
        pytest.param(503, LLMTransientError, id="503"),
    ],
)
async def test_retryable_http_status_exhausts_attempts_then_maps_to_error_class(
    status: int, error_class: type[LLMProviderError]
) -> None:
    recorder = _Recorder(_status(status))
    sleep = _SleepLog()

    with pytest.raises(LLMProviderError) as excinfo:
        await _provider(recorder, sleep).complete(_MESSAGES, "system prompt", _SCHEMA)

    assert type(excinfo.value) is error_class
    assert _OUTPUT_SENTINEL not in str(excinfo.value)
    assert len(recorder.requests) == LLM_MAX_ATTEMPTS
    assert len(sleep.delays) == LLM_MAX_ATTEMPTS - 1


async def test_transient_failures_then_success_reports_attempt_count() -> None:
    recorder = _sequence(_status(503), _status(503), _json_response(_stop_body()))
    sleep = _SleepLog()

    response = await _provider(recorder, sleep).complete(_MESSAGES, "system prompt", _SCHEMA)

    assert response.attempts == 3
    assert response.content == json.loads(_stop_body()["choices"][0]["message"]["content"])
    assert len(recorder.requests) == 3
    assert len(sleep.delays) == 2


async def test_backoff_grows_between_attempts() -> None:
    recorder = _Recorder(_status(503))
    sleep = _SleepLog()

    with pytest.raises(LLMTransientError):
        await _provider(recorder, sleep).complete(_MESSAGES, "system prompt", _SCHEMA)

    first, second = sleep.delays
    assert 0 < first <= second <= LLM_BACKOFF_CAP_SECS


def _http_date_in_30s() -> str:
    return email.utils.format_datetime(datetime.now(UTC) + timedelta(seconds=30), usegmt=True)


@pytest.mark.parametrize(
    ("retry_after", "minimum"),
    [
        pytest.param(lambda: "7", 7.0, id="delta-seconds"),
        pytest.param(lambda: "2.5", 2.5, id="fractional-seconds"),
        pytest.param(_http_date_in_30s, 28.0, id="http-date"),
    ],
)
async def test_rate_limit_waits_at_least_retry_after(
    retry_after: Callable[[], str], minimum: float
) -> None:
    recorder = _sequence(_status(429, {"Retry-After": retry_after()}), _json_response(_stop_body()))
    sleep = _SleepLog()

    response = await _provider(recorder, sleep).complete(_MESSAGES, "system prompt", _SCHEMA)

    assert response.attempts == 2
    assert len(sleep.delays) == 1
    assert minimum <= sleep.delays[0] <= LLM_BACKOFF_CAP_SECS


async def test_retry_after_beyond_cap_stops_retrying() -> None:
    too_long = str(int(LLM_BACKOFF_CAP_SECS) + 1)
    recorder = _Recorder(_status(429, {"Retry-After": too_long}))
    sleep = _SleepLog()

    with pytest.raises(LLMRateLimitError):
        await _provider(recorder, sleep).complete(_MESSAGES, "system prompt", _SCHEMA)

    assert len(recorder.requests) == 1
    assert sleep.delays == []


@pytest.mark.parametrize("retry_after", ["soon", ""], ids=["unparseable", "empty"])
async def test_unusable_retry_after_falls_back_to_backoff(retry_after: str) -> None:
    recorder = _sequence(_status(429, {"Retry-After": retry_after}), _json_response(_stop_body()))
    sleep = _SleepLog()

    response = await _provider(recorder, sleep).complete(_MESSAGES, "system prompt", _SCHEMA)

    assert response.attempts == 2
    assert 0 < sleep.delays[0] <= LLM_BACKOFF_CAP_SECS


class _CounterLog:
    def __init__(self) -> None:
        self.points: list[dict[str, str]] = []

    def add(self, amount: int, attributes: dict[str, str]) -> None:
        self.points.extend([attributes] * amount)


async def test_retry_counter_records_each_retry_and_the_exhaustion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counter = _CounterLog()
    monkeypatch.setattr(openai_compatible, "_get_llm_retry_counter", lambda: counter)
    recorder = _Recorder(_status(429))

    with pytest.raises(LLMRateLimitError):
        await _provider(recorder).complete(_MESSAGES, "system prompt", _SCHEMA)

    retried = {"result": "retried", "error_class": "LLMRateLimitError"}
    exhausted = {"result": "exhausted", "error_class": "LLMRateLimitError"}
    assert counter.points == [retried] * (LLM_MAX_ATTEMPTS - 1) + [exhausted]


async def test_transport_timeout_retries_then_maps_to_timeout_error() -> None:
    def respond(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ReadTimeout("timed out", request=request)

    recorder = _Recorder(respond)

    with pytest.raises(LLMTimeoutError):
        await _provider(recorder).complete(_MESSAGES, "system prompt", _SCHEMA)

    assert len(recorder.requests) == LLM_MAX_ATTEMPTS


async def test_connection_error_maps_to_transient_not_timeout() -> None:
    def respond(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("refused", request=request)

    recorder = _Recorder(respond)

    with pytest.raises(LLMTransientError) as excinfo:
        await _provider(recorder).complete(_MESSAGES, "system prompt", _SCHEMA)

    assert type(excinfo.value) is LLMTransientError
    assert len(recorder.requests) == LLM_MAX_ATTEMPTS


async def test_non_json_output_is_not_retried() -> None:
    body = _stop_body()
    body["choices"][0]["message"]["content"] = _OUTPUT_SENTINEL
    recorder = _Recorder(_json_response(body))
    sleep = _SleepLog()

    with pytest.raises(LLMSchemaError):
        await _provider(recorder, sleep).complete(_MESSAGES, "system prompt", _SCHEMA)

    assert len(recorder.requests) == 1
    assert sleep.delays == []


def test_estimate_cost_uses_configured_prices() -> None:
    provider = _provider(_Recorder(_json_response(_stop_body())))

    assert provider.estimate_cost_cents(1_000_000, 1_000_000) == pytest.approx(600.0)


@pytest.mark.parametrize(
    ("api_key", "expected_authorization"),
    [
        pytest.param("", "Bearer unused", id="empty-key-sends-placeholder"),
        pytest.param("real-key", "Bearer real-key", id="set-key-passed-through"),
    ],
)
async def test_api_key_is_optional_and_passed_through_when_set(
    monkeypatch: pytest.MonkeyPatch, api_key: str, expected_authorization: str
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    recorder = _Recorder(_json_response(_stop_body()))

    await _provider(recorder, api_key=api_key).complete(_MESSAGES, "system prompt", _SCHEMA)

    assert recorder.authorizations == [expected_authorization]


@pytest.mark.parametrize(
    ("responders", "expected_attempts"),
    [
        pytest.param((_status(503),), LLM_MAX_ATTEMPTS, id="exhausted-retries"),
        pytest.param((_status(400),), 1, id="permanent-first-attempt"),
        pytest.param((_status(503), _status(400)), 2, id="permanent-after-a-retry"),
        pytest.param(
            (_status(503), _json_response(_length_body())), 2, id="truncated-after-a-retry"
        ),
    ],
)
async def test_raised_error_carries_the_number_of_requests_made(
    responders: tuple[Callable[[httpx2.Request], httpx2.Response], ...],
    expected_attempts: int,
) -> None:
    recorder = _sequence(*responders)

    with pytest.raises(LLMProviderError) as excinfo:
        await _provider(recorder).complete(_MESSAGES, "system prompt", _SCHEMA)

    assert excinfo.value.attempts == expected_attempts
    assert len(recorder.requests) == expected_attempts
