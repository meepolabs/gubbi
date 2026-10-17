"""OpenAICompatibleProvider against a real AsyncOpenAI on an httpx2 mock transport.

Response bodies come from ``tests/fixtures/openai_chat_completions.json``,
recorded from an OpenAI-compatible gateway fronting an Anthropic model (ids,
timestamps and model name replaced). That gateway reports cached input and
reasoning tokens as top-level usage fields, which is the shape under test.
"""

from __future__ import annotations

import asyncio
import copy
import email.utils
import json
import logging
import pathlib
import ssl
import time
import traceback
import warnings
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx2
import pytest
from openai._utils import _logs as openai_logs

from gubbi.constants import LLM_BACKOFF_CAP_SECS, LLM_MAX_ATTEMPTS
from gubbi.extraction.llm import model_provider
from gubbi.extraction.llm.openai_compatible import OpenAICompatibleProvider
from gubbi.extraction.llm.provider import (
    LLMDeadlineError,
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
from tests.fixtures.llm_http import ENV_CA_NAME, ca_subjects, deeply_nested_json, write_env_ca
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
        self.urls: list[str] = []
        self.headers: list[httpx2.Headers] = []
        self._respond = respond

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(json.loads(request.content))
        self.authorizations.append(request.headers.get("authorization"))
        self.urls.append(str(request.url))
        self.headers.append(request.headers)
        return self._respond(request)


def _json_response(body: dict[str, Any], status: int = 200) -> Callable[..., httpx2.Response]:
    def respond(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(status, json=body)

    return respond


class _SleepLog:
    """Records each wait and advances a fake clock by it."""

    def __init__(self) -> None:
        self.delays: list[float] = []
        self.now = 0.0

    async def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)
        self.now += seconds

    def clock(self) -> float:
        return self.now


class _OverrunningSleep(_SleepLog):
    """Advances the fake clock past each requested wait, as a busy scheduler can."""

    def __init__(self, overrun: float) -> None:
        super().__init__()
        self._overrun = overrun

    async def __call__(self, seconds: float) -> None:
        await super().__call__(seconds)
        self.now += self._overrun


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
    recorder: _Recorder,
    sleep: _SleepLog | None = None,
    api_key: str = "test-key",
    clock: Callable[[], float] | None = None,
) -> OpenAICompatibleProvider:
    sleep = sleep or _SleepLog()
    return OpenAICompatibleProvider(
        base_url="https://llm.example.test/v1",
        api_key=api_key,
        model="fixture-model",
        input_price_per_mtok=1.0,
        output_price_per_mtok=5.0,
        max_output_tokens=512,
        timeout=5.0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(recorder)),
        sleep=sleep,
        clock=clock or sleep.clock,
    )


def _default_client_provider(
    monkeypatch: pytest.MonkeyPatch, recorder: _Recorder, base_url: str
) -> OpenAICompatibleProvider:
    """A provider on the HTTP client it builds itself, its transport answered by ``recorder``."""
    mock = httpx2.MockTransport(recorder)

    async def handle(_self: httpx2.AsyncHTTPTransport, request: httpx2.Request) -> httpx2.Response:
        return await mock.handle_async_request(request)

    monkeypatch.setattr(httpx2.AsyncHTTPTransport, "handle_async_request", handle)
    sleep = _SleepLog()
    return OpenAICompatibleProvider(
        base_url=base_url,
        api_key="test-key",
        model="fixture-model",
        input_price_per_mtok=1.0,
        output_price_per_mtok=5.0,
        sleep=sleep,
        clock=sleep.clock,
    )


def _record_handling_transport(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Record which transport instance handles each request; a proxy mount is a different one."""
    handled_by: list[object] = []
    handle = httpx2.AsyncHTTPTransport.handle_async_request

    async def recording(
        self: httpx2.AsyncHTTPTransport, request: httpx2.Request
    ) -> httpx2.Response:
        handled_by.append(self)
        return await handle(self, request)

    monkeypatch.setattr(httpx2.AsyncHTTPTransport, "handle_async_request", recording)
    return handled_by


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
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__context__ is None
    assert _OUTPUT_SENTINEL not in "".join(traceback.format_exception(excinfo.value))


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
    monkeypatch.setattr(model_provider, "_get_llm_retry_counter", lambda: counter)
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


def _read_timeout(request: httpx2.Request) -> httpx2.Response:
    raise httpx2.ReadTimeout("timed out", request=request)


def _connect_error(request: httpx2.Request) -> httpx2.Response:
    raise httpx2.ConnectError("refused", request=request)


@pytest.mark.parametrize(
    ("responders", "expected_unmetered"),
    [
        pytest.param((_read_timeout, _json_response(_stop_body())), 1, id="timeout-then-success"),
        pytest.param((_connect_error, _json_response(_stop_body())), 0, id="connect-then-success"),
        pytest.param((_json_response(_stop_body()),), 0, id="first-attempt-success"),
    ],
)
async def test_response_counts_the_attempts_that_timed_out_without_usage(
    responders: tuple[Callable[[httpx2.Request], httpx2.Response], ...],
    expected_unmetered: int,
) -> None:
    response = await _provider(_sequence(*responders)).complete(_MESSAGES, "system prompt", _SCHEMA)

    assert response.unmetered_attempts == expected_unmetered


async def test_raised_error_counts_the_attempts_that_timed_out_without_usage() -> None:
    recorder = _sequence(_read_timeout, _connect_error, _read_timeout)

    with pytest.raises(LLMTimeoutError) as excinfo:
        await _provider(recorder).complete(_MESSAGES, "system prompt", _SCHEMA)

    assert excinfo.value.attempts == LLM_MAX_ATTEMPTS == 3
    assert excinfo.value.unmetered_attempts == 2


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
    ("api_key", "expected_authorization", "custom_headers"),
    [
        pytest.param("", "Bearer unused", "Authorization: Bearer FROM_ENV", id="empty-key"),
        pytest.param("real-key", "Bearer real-key", "Authorization: Bearer FROM_ENV", id="set-key"),
        pytest.param(
            "real-key",
            "Bearer real-key",
            "AUTHORIZATION: Bearer FROM_ENV\nopenai-ORGANIZATION: FROM_ENV",
            id="mixed-casing",
        ),
    ],
)
async def test_headers_from_the_environment_never_reach_the_request(
    monkeypatch: pytest.MonkeyPatch, api_key: str, expected_authorization: str, custom_headers: str
) -> None:
    monkeypatch.setenv("OPENAI_CUSTOM_HEADERS", custom_headers)
    monkeypatch.setenv("OPENAI_ORG_ID", "FROM_ENV")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "FROM_ENV")
    recorder = _Recorder(_json_response(_stop_body()))

    await _provider(recorder, api_key=api_key).complete(_MESSAGES, "system prompt", _SCHEMA)

    assert recorder.authorizations == [expected_authorization]
    assert "openai-organization" not in recorder.headers[0]
    assert "openai-project" not in recorder.headers[0]


@pytest.mark.parametrize(
    ("status", "location"),
    [
        pytest.param(status, "https://elsewhere.example.test/v1", id=f"{status}-other-host")
        for status in (301, 302, 303, 307, 308)
    ]
    + [pytest.param(307, "https://llm.example.test/elsewhere", id="307-same-host-other-path")],
)
async def test_redirect_is_not_followed_and_fails_permanently(
    monkeypatch: pytest.MonkeyPatch, status: int, location: str
) -> None:
    configured_host = "llm.example.test"
    configured_url = f"https://{configured_host}/v1/chat/completions"

    def respond(request: httpx2.Request) -> httpx2.Response:
        if str(request.url) == configured_url:
            return httpx2.Response(status, headers={"Location": location})
        return httpx2.Response(200, json=_stop_body())

    recorder = _Recorder(respond)
    provider = _default_client_provider(monkeypatch, recorder, f"https://{configured_host}/v1")

    with pytest.raises(LLMProviderError) as excinfo:
        await provider.complete(_MESSAGES, "system prompt", _SCHEMA)

    assert type(excinfo.value) is LLMPermanentError
    assert recorder.urls == [configured_url]


async def test_default_client_ignores_proxy_and_ca_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        monkeypatch.setenv(name, "http://proxy.example.test:3128")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.setenv("SSL_CERT_FILE", str(write_env_ca(tmp_path)))
    recorder = _Recorder(_json_response(_stop_body()))
    provider = _default_client_provider(monkeypatch, recorder, "https://llm.example.test/v1")
    handled_by = _record_handling_transport(monkeypatch)
    http_client = provider._chat_model.client._client

    await provider.complete(_MESSAGES, "system prompt", _SCHEMA)

    assert ca_subjects(ssl.create_default_context()) == {ENV_CA_NAME}
    assert http_client._mounts == {}
    assert ENV_CA_NAME not in ca_subjects(http_client._transport._pool._ssl_context)
    assert handled_by == [http_client._transport]
    assert recorder.urls == ["https://llm.example.test/v1/chat/completions"]


@pytest.fixture
def sdk_debug_logging(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[logging.LogRecord]]:
    """Apply ``OPENAI_LOG=debug`` as SDK import does; collect the SDK's records."""
    sdk_logger = logging.getLogger("openai")
    saved_level = sdk_logger.level
    root_handlers = list(logging.getLogger().handlers)
    records: list[logging.LogRecord] = []
    handler = logging.Handler(level=logging.DEBUG)
    handler.emit = records.append  # type: ignore[method-assign]
    monkeypatch.setenv("OPENAI_LOG", "debug")
    openai_logs.setup_logging()
    sdk_logger.addHandler(handler)
    try:
        yield records
    finally:
        sdk_logger.removeHandler(handler)
        sdk_logger.setLevel(saved_level)
        logging.getLogger().handlers[:] = root_handlers


async def test_sdk_debug_logging_emits_nothing_below_warning(
    sdk_debug_logging: list[logging.LogRecord],
) -> None:
    assert logging.getLogger("openai").isEnabledFor(logging.DEBUG)
    recorder = _Recorder(_json_response(_stop_body()))
    messages: list[LLMMessage] = [{"role": "user", "content": _OUTPUT_SENTINEL}]

    await _provider(recorder).complete(messages, "system prompt", _SCHEMA)

    assert _OUTPUT_SENTINEL in json.dumps(recorder.requests[0])
    assert not [r for r in sdk_debug_logging if r.levelno < logging.WARNING]
    assert not [r for r in sdk_debug_logging if _OUTPUT_SENTINEL in r.getMessage()]


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


# -- Time budget -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("budget", "retry_after"),
    [
        pytest.param(5.0, "7", id="retry-after-past-budget"),
        pytest.param(7.0, "7", id="retry-after-equal-to-budget"),
    ],
)
async def test_retry_wait_past_the_time_budget_raises_deadline_without_sleeping(
    budget: float, retry_after: str
) -> None:
    recorder = _Recorder(_status(429, {"Retry-After": retry_after}))
    sleep = _SleepLog()

    with pytest.raises(LLMDeadlineError) as excinfo:
        await _provider(recorder, sleep).complete(
            _MESSAGES, "system prompt", _SCHEMA, time_budget=budget
        )

    assert sleep.delays == []
    assert len(recorder.requests) == 1
    assert excinfo.value.attempts == 1
    assert _OUTPUT_SENTINEL not in "".join(traceback.format_exception(excinfo.value))


async def test_retry_wait_inside_the_time_budget_still_retries() -> None:
    recorder = _sequence(_status(429, {"Retry-After": "7"}), _json_response(_stop_body()))
    sleep = _SleepLog()

    response = await _provider(recorder, sleep).complete(
        _MESSAGES, "system prompt", _SCHEMA, time_budget=30.0
    )

    assert response.attempts == 2
    assert len(sleep.delays) == 1
    assert 7.0 <= sleep.delays[0] < 30.0


async def test_retry_wait_that_overruns_the_time_budget_starts_no_further_request() -> None:
    recorder = _Recorder(_status(503))
    sleep = _OverrunningSleep(overrun=30.0)

    with pytest.raises(LLMDeadlineError) as excinfo:
        await _provider(recorder, sleep).complete(
            _MESSAGES, "system prompt", _SCHEMA, time_budget=20.0
        )

    assert len(sleep.delays) == 1
    assert len(recorder.requests) == 1
    assert excinfo.value.attempts == 1


async def test_spent_time_budget_makes_no_request() -> None:
    recorder = _Recorder(_json_response(_stop_body()))

    with pytest.raises(LLMDeadlineError) as excinfo:
        await _provider(recorder).complete(_MESSAGES, "system prompt", _SCHEMA, time_budget=0.0)

    assert recorder.requests == []
    assert excinfo.value.attempts == 0


async def test_attempt_is_cut_off_at_the_time_budget() -> None:
    async def respond_slowly(request: httpx2.Request) -> httpx2.Response:
        await asyncio.sleep(5.0)
        return httpx2.Response(200, json=_stop_body())

    recorder = _Recorder(respond_slowly)  # type: ignore[arg-type]
    started = time.monotonic()

    with pytest.raises(LLMDeadlineError) as excinfo:
        await _provider(recorder, clock=time.monotonic).complete(
            _MESSAGES, "system prompt", _SCHEMA, time_budget=0.2
        )

    assert time.monotonic() - started < 2.0
    assert excinfo.value.attempts == 1
    assert excinfo.value.unmetered_attempts == 1


async def test_unusable_finish_inside_the_time_budget_keeps_its_usage() -> None:
    body = _length_body()
    recorder = _Recorder(_json_response(body))

    with pytest.raises(LLMTruncatedError) as excinfo:
        await _provider(recorder).complete(_MESSAGES, "system prompt", _SCHEMA, time_budget=60.0)

    assert excinfo.value.usage == _wire_usage(body)


# -- Upstream text stays out of raised errors ----------------------------------------


def _malformed_completion(_request: httpx2.Request) -> httpx2.Response:
    body = _stop_body()
    body["choices"][0]["finish_reason"] = _OUTPUT_SENTINEL
    return httpx2.Response(200, json=body)


async def test_malformed_json_success_body_is_transient_and_kept_out_of_the_error() -> None:
    def respond(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200,
            content=f"not json {_OUTPUT_SENTINEL}".encode(),
            headers={"content-type": "application/json"},
        )

    recorder = _Recorder(respond)

    with pytest.raises(LLMProviderError) as excinfo:
        await _provider(recorder).complete(_MESSAGES, "system prompt", _SCHEMA)

    assert type(excinfo.value) is LLMTransientError
    assert len(recorder.requests) == LLM_MAX_ATTEMPTS
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__context__ is None
    assert _OUTPUT_SENTINEL not in str(excinfo.value)
    assert _OUTPUT_SENTINEL not in repr(excinfo.value.args)


@pytest.mark.parametrize(
    "responder",
    [
        pytest.param(_status(400), id="permanent-status-body"),
        pytest.param(_status(503), id="retryable-status-body"),
        pytest.param(_status(429, {"Retry-After": "1"}), id="rate-limit-body"),
        pytest.param(_malformed_completion, id="schema-invalid-completion"),
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


def _empty_choices_body() -> dict[str, Any]:
    body = _stop_body()
    body["choices"] = []
    body["id"] = _OUTPUT_SENTINEL
    return body


def _list_content_body() -> dict[str, Any]:
    body = _stop_body()
    body["choices"][0]["message"]["content"] = [_OUTPUT_SENTINEL]
    return body


@pytest.mark.parametrize(
    "body",
    [
        pytest.param(json.dumps(_empty_choices_body()), id="empty-choices"),
        pytest.param(json.dumps(_list_content_body()), id="non-string-content"),
        pytest.param(deeply_nested_json(_OUTPUT_SENTINEL), id="deeply-nested-json"),
    ],
)
async def test_unmappable_success_body_is_transient_and_kept_out_of_the_error(body: str) -> None:
    def respond(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200, content=body.encode(), headers={"content-type": "application/json"}
        )

    recorder = _Recorder(respond)

    with pytest.raises(LLMProviderError) as excinfo:
        await _provider(recorder).complete(_MESSAGES, "system prompt", _SCHEMA)

    assert type(excinfo.value) is LLMTransientError
    assert len(recorder.requests) == LLM_MAX_ATTEMPTS
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__context__ is None
    assert _OUTPUT_SENTINEL not in "".join(traceback.format_exception(excinfo.value))


async def test_wrongly_typed_content_emits_no_warning_quoting_the_body() -> None:
    recorder = _Recorder(_json_response(_list_content_body()))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        provider = _provider(recorder)
        with pytest.raises(LLMTransientError):
            await provider.complete(_MESSAGES, "system prompt", _SCHEMA)

    assert recorder.requests
    assert not [w for w in caught if _OUTPUT_SENTINEL in str(w.message)]
