"""AnthropicProvider against a real AsyncAnthropic on an httpx2 mock transport.

Response bodies come from ``tests/fixtures/anthropic_messages.json``, shaped
after Messages API responses; ``test_fixture_bodies_parse_as_sdk_messages``
holds them to the SDK's own response type. Retry, budget and backoff rules
are shared with the OpenAI-compatible provider and pinned in its tests; the
cases here pin what differs: the wire request, finish-reason mapping, status
classification and usage.
"""

from __future__ import annotations

import copy
import json
import pathlib
import traceback
from collections.abc import Callable, Iterator
from typing import Any

import httpx2
import pytest
from anthropic.types.beta import BetaMessage

from gubbi.constants import LLM_BACKOFF_CAP_SECS, LLM_MAX_ATTEMPTS
from gubbi.extraction.llm.anthropic_messages import AnthropicEffort, AnthropicProvider
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
    (pathlib.Path(__file__).parents[1] / "fixtures" / "anthropic_messages.json").read_text()
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
        self.urls: list[str] = []
        self.api_keys: list[str | None] = []
        self._respond = respond

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(json.loads(request.content))
        self.urls.append(str(request.url))
        self.api_keys.append(request.headers.get("x-api-key"))
        return self._respond(request)


def _json_response(body: dict[str, Any], status: int = 200) -> Callable[..., httpx2.Response]:
    def respond(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(status, json=body)

    return respond


def _status(status: int, headers: dict[str, str] | None = None) -> Callable[..., httpx2.Response]:
    def respond(_request: httpx2.Request) -> httpx2.Response:
        body = {"type": "error", "error": {"type": "api_error", "message": _OUTPUT_SENTINEL}}
        return httpx2.Response(status, json=body, headers=headers)

    return respond


def _sequence(*responders: Callable[[httpx2.Request], httpx2.Response]) -> _Recorder:
    """Answer the n-th request with the n-th responder; the last one repeats."""
    queue: Iterator[Callable[[httpx2.Request], httpx2.Response]] = iter(responders)
    current = responders[-1]

    def respond(request: httpx2.Request) -> httpx2.Response:
        nonlocal current
        current = next(queue, current)
        return current(request)

    return _Recorder(respond)


class _SleepLog:
    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)

    def clock(self) -> float:
        return sum(self.delays)


def _provider(
    recorder: _Recorder,
    sleep: _SleepLog | None = None,
    *,
    base_url: str = "",
    effort: AnthropicEffort | None = None,
    model: str = "fixture-model",
) -> AnthropicProvider:
    sleep = sleep or _SleepLog()
    return AnthropicProvider(
        api_key="test-key",
        model=model,
        input_price_per_mtok=0.1,
        output_price_per_mtok=0.5,
        base_url=base_url,
        effort=effort,
        max_output_tokens=512,
        timeout=5.0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(recorder)),
        sleep=sleep,
        clock=sleep.clock,
    )


def _end_turn_body() -> dict[str, Any]:
    return copy.deepcopy(_FIXTURES["end_turn_cached"])


def _refusal_body() -> dict[str, Any]:
    body = copy.deepcopy(_FIXTURES["refusal"])
    body["content"][0]["text"] = _OUTPUT_SENTINEL
    body["stop_details"]["explanation"] = _REFUSAL_TEXT
    return body


def _max_tokens_body() -> dict[str, Any]:
    body = copy.deepcopy(_FIXTURES["max_tokens"])
    body["content"][0]["text"] = '{"summary": "' + _OUTPUT_SENTINEL
    return body


def _structured_text(body: dict[str, Any]) -> str:
    return next(str(block["text"]) for block in body["content"] if block["type"] == "text")


def _expected_usage(body: dict[str, Any]) -> LLMUsage:
    """Cached reads and cache writes are input the call was billed for, so both count."""
    usage = body["usage"]
    cache_read = usage.get("cache_read_input_tokens") or 0
    cache_write = usage.get("cache_creation_input_tokens") or 0
    thinking = (usage.get("output_tokens_details") or {}).get("thinking_tokens", 0)
    return LLMUsage(
        input=usage["input_tokens"] + cache_read + cache_write,
        output=usage["output_tokens"],
        cached_input=cache_read,
        reasoning=thinking,
    )


@pytest.mark.parametrize("name", sorted(_FIXTURES))
def test_fixture_bodies_parse_as_sdk_messages(name: str) -> None:
    message = BetaMessage.model_validate(_FIXTURES[name])

    assert message.stop_reason == name.removesuffix("_cached")


# -- Wire request --------------------------------------------------------------------


@pytest.mark.parametrize("model", ["claude-haiku-5-5", "fixture-model"])
async def test_structured_request_sends_json_schema_output_config_in_one_http_call(
    model: str,
) -> None:
    recorder = _Recorder(_json_response(_end_turn_body()))

    await _provider(recorder, model=model).complete(_MESSAGES, "system prompt", _SCHEMA)

    assert len(recorder.requests) == 1
    output_format = recorder.requests[0]["output_config"]["format"]
    assert output_format["type"] == "json_schema"
    assert output_format["schema"]["properties"] == _SCHEMA["properties"]
    assert output_format["schema"]["required"] == _SCHEMA["required"]
    assert output_format["schema"]["additionalProperties"] is False
    assert "tools" not in recorder.requests[0]
    assert "tool_choice" not in recorder.requests[0]


async def test_request_sends_model_system_prompt_messages_token_cap_and_key() -> None:
    recorder = _Recorder(_json_response(_end_turn_body()))

    await _provider(recorder).complete(_MESSAGES, "system prompt", _SCHEMA)

    sent = recorder.requests[0]
    assert sent["model"] == "fixture-model"
    assert sent["max_tokens"] == 512
    assert sent["system"] == "system prompt"
    assert sent["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "I bought a car today."}]}
    ]
    assert recorder.api_keys == ["test-key"]


async def test_mid_conversation_system_message_joins_the_top_level_system_prompt() -> None:
    recorder = _Recorder(_json_response(_end_turn_body()))
    messages: list[LLMMessage] = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
        {"role": "system", "content": "later instruction"},
        {"role": "user", "content": "second"},
    ]

    await _provider(recorder).complete(messages, "system prompt", _SCHEMA)

    sent = recorder.requests[0]
    assert "later instruction" in sent["system"]
    assert [m["role"] for m in sent["messages"]] == ["user", "assistant", "user"]


@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
async def test_configured_effort_is_sent_in_output_config(effort: AnthropicEffort) -> None:
    recorder = _Recorder(_json_response(_end_turn_body()))

    await _provider(recorder, effort=effort).complete(_MESSAGES, "system prompt", _SCHEMA)

    assert recorder.requests[0]["output_config"]["effort"] == effort


async def test_unset_effort_and_thinking_are_left_to_the_model_default() -> None:
    recorder = _Recorder(_json_response(_end_turn_body()))

    await _provider(recorder).complete(_MESSAGES, "system prompt", _SCHEMA)

    sent = recorder.requests[0]
    assert "effort" not in sent["output_config"]
    assert "thinking" not in sent


async def test_text_mode_sends_no_output_format() -> None:
    body = _end_turn_body()
    body["content"] = [{"type": "text", "text": "plain answer"}]
    recorder = _Recorder(_json_response(body))

    response = await _provider(recorder).complete(_MESSAGES, "system prompt")

    assert response.content == "plain answer"
    assert "output_config" not in recorder.requests[0]


@pytest.mark.parametrize(
    ("base_url", "expected_prefix"),
    [
        pytest.param("", "https://api.anthropic.com/v1/messages", id="default"),
        pytest.param(
            "https://llm.example.test", "https://llm.example.test/v1/messages", id="configured"
        ),
    ],
)
async def test_requests_go_to_the_configured_or_default_base_url(
    monkeypatch: pytest.MonkeyPatch, base_url: str, expected_prefix: str
) -> None:
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://elsewhere.example.test")
    recorder = _Recorder(_json_response(_end_turn_body()))

    await _provider(recorder, base_url=base_url).complete(_MESSAGES, "system prompt", _SCHEMA)

    assert recorder.urls[0].startswith(expected_prefix)


# -- Response mapping ----------------------------------------------------------------


async def test_structured_response_returns_parsed_object_skipping_thinking() -> None:
    body = _end_turn_body()
    recorder = _Recorder(_json_response(body))

    response = await _provider(recorder).complete(_MESSAGES, "system prompt", _SCHEMA)

    assert response.content == json.loads(_structured_text(body))
    assert response.finish_reason == "stop"
    assert response.attempts == 1


async def test_usage_input_includes_cache_reads_and_writes_and_cached_is_reads() -> None:
    body = _end_turn_body()
    recorder = _Recorder(_json_response(body))

    response = await _provider(recorder).complete(_MESSAGES, "system prompt", _SCHEMA)

    assert (
        response.input_tokens,
        response.cached_input_tokens,
        response.output_tokens,
        response.reasoning_tokens,
    ) == (412 + 5120 + 768, 5120, 233, 180)


@pytest.mark.parametrize(
    ("body_factory", "error_class"),
    [
        pytest.param(_max_tokens_body, LLMTruncatedError, id="max-tokens"),
        pytest.param(_refusal_body, LLMRefusalError, id="refusal"),
    ],
)
async def test_unusable_stop_reason_raises_with_usage_after_one_request(
    body_factory: Callable[[], dict[str, Any]],
    error_class: type[LLMProviderError],
) -> None:
    body = body_factory()
    recorder = _Recorder(_json_response(body))
    sleep = _SleepLog()

    with pytest.raises(error_class) as excinfo:
        await _provider(recorder, sleep).complete(_MESSAGES, "system prompt", _SCHEMA)

    assert excinfo.value.usage == _expected_usage(body)
    assert excinfo.value.attempts == 1
    assert len(recorder.requests) == 1
    assert sleep.delays == []


@pytest.mark.parametrize(
    "body_factory",
    [
        pytest.param(_max_tokens_body, id="max-tokens"),
        pytest.param(_refusal_body, id="refusal"),
    ],
)
async def test_unusable_stop_reason_keeps_model_text_out_of_errors_and_logs(
    body_factory: Callable[[], dict[str, Any]],
    log_capture: LogCapture,
) -> None:
    recorder = _Recorder(_json_response(body_factory()))

    with pytest.raises(LLMProviderError) as excinfo:
        await _provider(recorder).complete(_MESSAGES, "system prompt", _SCHEMA)

    assert "llm_response_unusable" in log_capture.text
    for leaked in (_REFUSAL_TEXT, _OUTPUT_SENTINEL):
        assert leaked not in str(excinfo.value)
        assert leaked not in "".join(traceback.format_exception(excinfo.value))
        assert leaked not in log_capture.text


@pytest.mark.parametrize(
    ("text", "message"),
    [
        pytest.param(_OUTPUT_SENTINEL, "not valid JSON", id="non-json"),
        pytest.param(f'["{_OUTPUT_SENTINEL}"]', "not a JSON object", id="json-array"),
    ],
)
async def test_unparseable_structured_output_raises_schema_error_once(
    text: str, message: str
) -> None:
    body = _end_turn_body()
    body["content"] = [{"type": "text", "text": text}]
    recorder = _Recorder(_json_response(body))

    with pytest.raises(LLMSchemaError, match=message) as excinfo:
        await _provider(recorder).complete(_MESSAGES, "system prompt", _SCHEMA)

    assert excinfo.value.usage == _expected_usage(body)
    assert len(recorder.requests) == 1
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__context__ is None
    assert _OUTPUT_SENTINEL not in "".join(traceback.format_exception(excinfo.value))


# -- Status classification and retries -----------------------------------------------


@pytest.mark.parametrize("status", [400, 401, 403, 404, 413])
async def test_permanent_http_status_raises_after_one_request(status: int) -> None:
    recorder = _Recorder(_status(status))
    sleep = _SleepLog()

    with pytest.raises(LLMProviderError) as excinfo:
        await _provider(recorder, sleep).complete(_MESSAGES, "system prompt", _SCHEMA)

    assert type(excinfo.value) is LLMPermanentError
    assert len(recorder.requests) == 1
    assert sleep.delays == []


@pytest.mark.parametrize(
    ("status", "error_class"),
    [
        pytest.param(408, LLMTransientError, id="408"),
        pytest.param(429, LLMRateLimitError, id="429"),
        pytest.param(500, LLMTransientError, id="500"),
        pytest.param(529, LLMTransientError, id="529-overloaded"),
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
    assert len(recorder.requests) == LLM_MAX_ATTEMPTS
    assert len(sleep.delays) == LLM_MAX_ATTEMPTS - 1


async def test_overloaded_then_success_reports_attempt_count() -> None:
    recorder = _sequence(_status(529), _status(503), _json_response(_end_turn_body()))
    sleep = _SleepLog()

    response = await _provider(recorder, sleep).complete(_MESSAGES, "system prompt", _SCHEMA)

    assert response.attempts == 3
    assert len(recorder.requests) == 3


async def test_rate_limit_waits_at_least_retry_after_then_succeeds() -> None:
    recorder = _sequence(_status(429, {"Retry-After": "7"}), _json_response(_end_turn_body()))
    sleep = _SleepLog()

    response = await _provider(recorder, sleep).complete(_MESSAGES, "system prompt", _SCHEMA)

    assert response.attempts == 2
    assert len(sleep.delays) == 1
    assert 7.0 <= sleep.delays[0] <= LLM_BACKOFF_CAP_SECS


def _read_timeout(request: httpx2.Request) -> httpx2.Response:
    raise httpx2.ReadTimeout("timed out", request=request)


def _connect_error(request: httpx2.Request) -> httpx2.Response:
    raise httpx2.ConnectError("refused", request=request)


async def test_transport_timeout_retries_then_maps_to_timeout_error() -> None:
    recorder = _Recorder(_read_timeout)

    with pytest.raises(LLMTimeoutError) as excinfo:
        await _provider(recorder).complete(_MESSAGES, "system prompt", _SCHEMA)

    assert len(recorder.requests) == LLM_MAX_ATTEMPTS
    assert excinfo.value.unmetered_attempts == LLM_MAX_ATTEMPTS


async def test_connection_error_maps_to_transient_not_timeout() -> None:
    recorder = _Recorder(_connect_error)

    with pytest.raises(LLMTransientError) as excinfo:
        await _provider(recorder).complete(_MESSAGES, "system prompt", _SCHEMA)

    assert type(excinfo.value) is LLMTransientError
    assert excinfo.value.unmetered_attempts == 0


@pytest.mark.parametrize(
    "responder",
    [
        pytest.param(_status(400), id="permanent-status-body"),
        pytest.param(_status(529), id="overloaded-status-body"),
        pytest.param(_status(429, {"Retry-After": "1"}), id="rate-limit-body"),
    ],
)
async def test_upstream_body_never_reaches_the_raised_error_or_its_traceback(
    responder: Callable[[httpx2.Request], httpx2.Response],
) -> None:
    recorder = _Recorder(responder)

    with pytest.raises(LLMProviderError) as excinfo:
        await _provider(recorder).complete(_MESSAGES, "system prompt", _SCHEMA)

    assert _OUTPUT_SENTINEL not in str(excinfo.value)
    assert _OUTPUT_SENTINEL not in "".join(traceback.format_exception(excinfo.value))
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__context__ is None


def test_estimate_cost_uses_configured_prices() -> None:
    provider = _provider(_Recorder(_json_response(_end_turn_body())))

    assert provider.estimate_cost_cents(1_000_000, 1_000_000) == pytest.approx(60.0)
