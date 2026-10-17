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
import logging
import pathlib
import ssl
import traceback
from collections.abc import Callable, Iterator
from typing import Any

import httpx2
import pytest
from anthropic._utils import _logs as anthropic_logs
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
from tests.fixtures.llm_http import ENV_CA_NAME, ca_subjects, deeply_nested_json, write_env_ca
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
        self.authorizations: list[str | None] = []
        self._respond = respond

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(json.loads(request.content))
        self.urls.append(str(request.url))
        self.api_keys.append(request.headers.get("x-api-key"))
        self.authorizations.append(request.headers.get("authorization"))
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


def _default_client_provider(
    monkeypatch: pytest.MonkeyPatch, recorder: _Recorder, base_url: str
) -> AnthropicProvider:
    """A provider on the HTTP client it builds itself, its transport answered by ``recorder``."""
    mock = httpx2.MockTransport(recorder)

    async def handle(_self: httpx2.AsyncHTTPTransport, request: httpx2.Request) -> httpx2.Response:
        return await mock.handle_async_request(request)

    monkeypatch.setattr(httpx2.AsyncHTTPTransport, "handle_async_request", handle)
    sleep = _SleepLog()
    return AnthropicProvider(
        api_key="test-key",
        model="fixture-model",
        input_price_per_mtok=0.1,
        output_price_per_mtok=0.5,
        base_url=base_url,
        sleep=sleep,
        clock=sleep.clock,
    )


def _host_and_path(url: str) -> tuple[str, str]:
    parsed = httpx2.URL(url)
    return parsed.host, parsed.path


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


@pytest.mark.parametrize(
    "custom_headers",
    [
        pytest.param("X-Api-Key: FROM_ENV\nAuthorization: Bearer X", id="canonical-casing"),
        pytest.param("x-API-key: FROM_ENV\nAUTHORIZATION: Bearer X", id="mixed-casing"),
    ],
)
async def test_custom_headers_from_the_environment_cannot_replace_the_configured_key(
    monkeypatch: pytest.MonkeyPatch, custom_headers: str
) -> None:
    monkeypatch.setenv("ANTHROPIC_CUSTOM_HEADERS", custom_headers)
    recorder = _Recorder(_json_response(_end_turn_body()))

    await _provider(recorder).complete(_MESSAGES, "system prompt", _SCHEMA)

    assert recorder.api_keys == ["test-key"]
    assert recorder.authorizations == [None]


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
    configured_path = "/v1/messages"

    def respond(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == configured_path:
            return httpx2.Response(status, headers={"Location": location})
        return httpx2.Response(200, json=_end_turn_body())

    recorder = _Recorder(respond)
    provider = _default_client_provider(monkeypatch, recorder, f"https://{configured_host}")

    with pytest.raises(LLMProviderError) as excinfo:
        await provider.complete(_MESSAGES, "system prompt", _SCHEMA)

    assert type(excinfo.value) is LLMPermanentError
    assert [_host_and_path(url) for url in recorder.urls] == [(configured_host, configured_path)]


async def test_default_client_ignores_proxy_and_ca_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        monkeypatch.setenv(name, "http://proxy.example.test:3128")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.setenv("SSL_CERT_FILE", str(write_env_ca(tmp_path)))
    recorder = _Recorder(_json_response(_end_turn_body()))
    provider = _default_client_provider(monkeypatch, recorder, "https://llm.example.test")
    handled_by = _record_handling_transport(monkeypatch)
    http_client = provider._chat_model.client._client

    await provider.complete(_MESSAGES, "system prompt", _SCHEMA)

    assert ca_subjects(ssl.create_default_context()) == {ENV_CA_NAME}
    assert http_client._mounts == {}
    assert ENV_CA_NAME not in ca_subjects(http_client._transport._pool._ssl_context)
    assert handled_by == [http_client._transport]
    assert [_host_and_path(url) for url in recorder.urls] == [("llm.example.test", "/v1/messages")]


@pytest.fixture
def sdk_debug_logging(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[logging.LogRecord]]:
    """Apply ``ANTHROPIC_LOG=debug`` as SDK import does; collect the SDK's records."""
    sdk_logger = logging.getLogger("anthropic")
    saved = [(logger, logger.level) for logger in (sdk_logger, logging.getLogger("httpx2"))]
    root_handlers = list(logging.getLogger().handlers)
    records: list[logging.LogRecord] = []
    handler = logging.Handler(level=logging.DEBUG)
    handler.emit = records.append  # type: ignore[method-assign]
    monkeypatch.setenv("ANTHROPIC_LOG", "debug")
    anthropic_logs.setup_logging()
    sdk_logger.addHandler(handler)
    try:
        yield records
    finally:
        sdk_logger.removeHandler(handler)
        for logger, level in saved:
            logger.setLevel(level)
        logging.getLogger().handlers[:] = root_handlers


async def test_sdk_debug_logging_emits_nothing_below_warning(
    sdk_debug_logging: list[logging.LogRecord],
) -> None:
    assert logging.getLogger("anthropic").isEnabledFor(logging.DEBUG)
    recorder = _Recorder(_json_response(_end_turn_body()))
    messages: list[LLMMessage] = [{"role": "user", "content": _OUTPUT_SENTINEL}]

    await _provider(recorder).complete(messages, "system prompt", _SCHEMA)

    assert _OUTPUT_SENTINEL in json.dumps(recorder.requests[0])
    assert not [r for r in sdk_debug_logging if r.levelno < logging.WARNING]
    assert not [r for r in sdk_debug_logging if _OUTPUT_SENTINEL in r.getMessage()]


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


@pytest.mark.parametrize(
    "content_type",
    [
        pytest.param(None, id="untyped-body"),
        pytest.param("application/json", id="json-typed-body"),
    ],
)
async def test_malformed_success_body_is_transient_and_kept_out_of_the_error(
    content_type: str | None,
) -> None:
    headers = {"content-type": content_type} if content_type else None

    def respond(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200, content=f"not json {_OUTPUT_SENTINEL}".encode(), headers=headers
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


def _string_content_body() -> dict[str, Any]:
    body = _end_turn_body()
    body["content"] = _OUTPUT_SENTINEL
    return body


def _null_text_body() -> dict[str, Any]:
    body = _end_turn_body()
    for block in body["content"]:
        if block["type"] == "text":
            block["text"] = None
    return body


@pytest.mark.parametrize(
    "body",
    [
        pytest.param(json.dumps(_string_content_body()), id="string-content"),
        pytest.param(json.dumps(_null_text_body()), id="null-text"),
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


def test_estimate_cost_uses_configured_prices() -> None:
    provider = _provider(_Recorder(_json_response(_end_turn_body())))

    assert provider.estimate_cost_cents(1_000_000, 1_000_000) == pytest.approx(60.0)
