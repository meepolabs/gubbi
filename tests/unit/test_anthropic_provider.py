"""Unit tests for the AnthropicProvider exception translation + retry observability.

Covers (B2 / S7 M2):
  * Each anthropic SDK exception type translates to the right LLM* class.
  * _call_with_retry retries LLMTransientError, raises LLMPermanentError
    immediately, and emits the retry observability counter.
  * _classify_error consumers see only LLM* (no anthropic types leak).
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from anthropic import (
    APIConnectionError,
    APIResponseValidationError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    ConflictError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
)
from anthropic.types import Message, Usage

from gubbi.extraction.llm.anthropic_provider import (
    _get_anthropic_retry_counter,
    _translate_anthropic_error,
)
from gubbi.extraction.llm.provider import (
    LLMPermanentError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTransientError,
)

# ---------------------------------------------------------------------------
# Translation matrix
# ---------------------------------------------------------------------------


def _make_status_error(status_code: int) -> APIStatusError:
    return APIStatusError(
        message=f"status {status_code}",
        response=httpx.Response(
            status_code=status_code,
            request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
        ),
        body=None,
    )


def _make_status_error_subclass(cls: type[APIStatusError], status_code: int) -> APIStatusError:
    """Construct a 4xx APIStatusError subclass with a synthetic httpx response."""
    return cls(
        message=f"{cls.__name__} {status_code}",
        response=httpx.Response(
            status_code=status_code,
            request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
        ),
        body=None,
    )


def _make_response_validation_error() -> APIResponseValidationError:
    """APIResponseValidationError takes (response, body, *, message)."""
    return APIResponseValidationError(
        response=httpx.Response(
            status_code=200,
            request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
        ),
        body=None,
        message="response shape invalid",
    )


@pytest.mark.parametrize(
    ("exc_factory", "expected_cls"),
    [
        (
            lambda: RateLimitError(
                message="rate limit",
                response=MagicMock(status_code=429),
                body=None,
            ),
            LLMRateLimitError,
        ),
        (
            lambda: APIConnectionError(request=MagicMock()),
            LLMTransientError,
        ),
        (
            lambda: APITimeoutError(request=MagicMock()),
            LLMTransientError,
        ),
        (
            lambda: InternalServerError(
                message="boom",
                response=MagicMock(status_code=500),
                body=None,
            ),
            LLMTransientError,
        ),
        (
            lambda: _make_status_error(503),
            LLMTransientError,
        ),
        (
            _make_response_validation_error,
            LLMTransientError,
        ),
        (
            lambda: _make_status_error(401),
            LLMPermanentError,
        ),
        (
            lambda: _make_status_error(400),
            LLMPermanentError,
        ),
        (
            lambda: _make_status_error_subclass(BadRequestError, 400),
            LLMPermanentError,
        ),
        (
            lambda: _make_status_error_subclass(AuthenticationError, 401),
            LLMPermanentError,
        ),
        (
            lambda: _make_status_error_subclass(PermissionDeniedError, 403),
            LLMPermanentError,
        ),
        (
            lambda: _make_status_error_subclass(NotFoundError, 404),
            LLMPermanentError,
        ),
        (
            lambda: _make_status_error_subclass(ConflictError, 409),
            LLMPermanentError,
        ),
        (
            lambda: _make_status_error_subclass(UnprocessableEntityError, 422),
            LLMPermanentError,
        ),
        (
            lambda: ValueError("schema bad"),
            LLMPermanentError,
        ),
    ],
)
def test_translate_anthropic_error_mapping(
    exc_factory: Callable[[], Exception], expected_cls: type[LLMProviderError]
) -> None:
    """Vendor exceptions map to the correct LLM* class (B2 mapping lock + R1 expansion)."""
    translated = _translate_anthropic_error(exc_factory())
    assert isinstance(translated, expected_cls), (
        f"{type(exc_factory()).__name__} should map to {expected_cls.__name__}, "
        f"got {type(translated).__name__}"
    )


def test_translate_anthropic_error_exhaustive_guard() -> None:
    """Guard test: every top-level anthropic exception class has a defined LLM* shape.

    Walks every Exception subclass exposed at the anthropic top-level and
    asserts the translator returns a known LLM* subclass for an instance of it.
    Failure here means the SDK introduced a new exception type the mapping has
    not been audited against -- update _translate_anthropic_error and the
    expected map below before merging.
    """
    import anthropic

    expected: dict[str, type[LLMProviderError]] = {
        # Rate limit -> rate-limit transient
        "RateLimitError": LLMRateLimitError,
        # Connection / timeout -> transient
        "APIConnectionError": LLMTransientError,
        "APITimeoutError": LLMTransientError,
        # Response shape -> transient (often upstream flake)
        "APIResponseValidationError": LLMTransientError,
        # 5xx -> transient
        "InternalServerError": LLMTransientError,
        # 4xx -> permanent
        "BadRequestError": LLMPermanentError,
        "AuthenticationError": LLMPermanentError,
        "PermissionDeniedError": LLMPermanentError,
        "NotFoundError": LLMPermanentError,
        "ConflictError": LLMPermanentError,
        "UnprocessableEntityError": LLMPermanentError,
        # Catch-all 4xx (no specific subclass) -> permanent.
        # APIStatusError instance with 400 lands here for guard purposes.
        "APIStatusError": LLMPermanentError,
        # Base classes -- catch-all permanent (status not known).
        "APIError": LLMPermanentError,
        "AnthropicError": LLMPermanentError,
    }

    discovered: dict[str, type[Exception]] = {}
    for name in dir(anthropic):
        obj = getattr(anthropic, name)
        if isinstance(obj, type) and issubclass(obj, Exception):
            discovered[name] = obj

    # Make sure no NEW exception class slipped into the SDK without a planned mapping.
    unmapped = sorted(set(discovered) - set(expected))
    assert not unmapped, (
        f"anthropic SDK exposes new exception classes without a planned mapping: "
        f"{unmapped}. Update _translate_anthropic_error and this guard test."
    )

    # Build a representative instance for each known class and assert mapping.
    factories: dict[str, Callable[[], Exception]] = {
        "RateLimitError": lambda: RateLimitError(
            message="rl", response=MagicMock(status_code=429), body=None
        ),
        "APIConnectionError": lambda: APIConnectionError(request=MagicMock()),
        "APITimeoutError": lambda: APITimeoutError(request=MagicMock()),
        "APIResponseValidationError": _make_response_validation_error,
        "InternalServerError": lambda: InternalServerError(
            message="boom", response=MagicMock(status_code=500), body=None
        ),
        "BadRequestError": lambda: _make_status_error_subclass(BadRequestError, 400),
        "AuthenticationError": lambda: _make_status_error_subclass(AuthenticationError, 401),
        "PermissionDeniedError": lambda: _make_status_error_subclass(PermissionDeniedError, 403),
        "NotFoundError": lambda: _make_status_error_subclass(NotFoundError, 404),
        "ConflictError": lambda: _make_status_error_subclass(ConflictError, 409),
        "UnprocessableEntityError": lambda: _make_status_error_subclass(
            UnprocessableEntityError, 422
        ),
        "APIStatusError": lambda: _make_status_error(400),
        "APIError": lambda: anthropic.APIError(
            "api err",
            request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
            body=None,
        ),
        "AnthropicError": lambda: anthropic.AnthropicError("base"),
    }

    for cls_name, expected_cls in expected.items():
        if cls_name not in factories:
            continue  # skip if SDK does not expose constructor we know
        translated = _translate_anthropic_error(factories[cls_name]())
        assert isinstance(translated, expected_cls), (
            f"{cls_name} should map to {expected_cls.__name__}, " f"got {type(translated).__name__}"
        )


def test_rate_limit_is_also_transient() -> None:
    """LLMRateLimitError is a subclass of LLMTransientError (the retry loop relies on this)."""
    err = LLMRateLimitError("x")
    assert isinstance(err, LLMTransientError)
    assert isinstance(err, LLMProviderError)


def test_permanent_is_provider_error_but_not_transient() -> None:
    err = LLMPermanentError("x")
    assert isinstance(err, LLMProviderError)
    assert not isinstance(err, LLMTransientError)


# ---------------------------------------------------------------------------
# Retry loop translates + observes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retry_loop_raises_llm_permanent_on_non_retryable() -> None:
    """A 400 from the SDK should surface as LLMPermanentError (translated, not retried)."""
    from gubbi.config import LLMConfig
    from gubbi.extraction.llm.anthropic_provider import AnthropicProvider

    config = LLMConfig(api_key="test-key", model="claude-haiku-4-5-20251001")
    provider = AnthropicProvider(config)

    with patch.object(provider._client.messages, "create", new_callable=AsyncMock) as mock_create:
        mock_create.side_effect = _make_status_error(400)

        with pytest.raises(LLMPermanentError):
            await provider._call_with_retry({})

    # Permanent -- no retry.
    assert mock_create.call_count == 1


@pytest.mark.asyncio
async def test_retry_loop_raises_llm_rate_limit_on_exhaustion() -> None:
    """Rate-limit retried until budget exhausted, then raised as LLMRateLimitError."""
    from gubbi.config import LLMConfig
    from gubbi.extraction.llm.anthropic_provider import AnthropicProvider

    config = LLMConfig(api_key="test-key", model="claude-haiku-4-5-20251001")
    provider = AnthropicProvider(config)

    rate_limit = RateLimitError(
        message="rate limit",
        response=MagicMock(status_code=429),
        body=None,
    )

    async def no_sleep(_d: float) -> None:
        return None

    with (
        patch.object(provider._client.messages, "create", new_callable=AsyncMock) as mock_create,
        patch("asyncio.sleep", new=no_sleep),
    ):
        mock_create.side_effect = [rate_limit] * 10  # more than max_retries

        with pytest.raises(LLMRateLimitError):
            await provider._call_with_retry({})


@pytest.mark.asyncio
async def test_retry_loop_succeeds_after_transient() -> None:
    """LLMTransientError retries and ultimately succeeds; original SDK type translated cleanly."""
    from gubbi.config import LLMConfig
    from gubbi.extraction.llm.anthropic_provider import AnthropicProvider

    config = LLMConfig(api_key="test-key", model="claude-haiku-4-5-20251001")
    provider = AnthropicProvider(config)

    mock_message = Message(
        id="m",
        type="message",
        role="assistant",
        content=[{"type": "text", "text": "ok"}],
        model="claude-haiku-4-5-20251001",
        stop_reason="end_turn",
        usage=Usage(input_tokens=1, output_tokens=1),
    )

    async def no_sleep(_d: float) -> None:
        return None

    with (
        patch.object(provider._client.messages, "create", new_callable=AsyncMock) as mock_create,
        patch("asyncio.sleep", new=no_sleep),
    ):
        mock_create.side_effect = [
            APIConnectionError(request=MagicMock()),
            mock_message,
        ]
        result = await provider._call_with_retry({})

    assert result is mock_message
    assert mock_create.call_count == 2


@pytest.mark.asyncio
async def test_retry_observability_logs_and_counter() -> None:
    """Each retry logs an INFO event and increments the counter with attribute."""
    from gubbi.config import LLMConfig
    from gubbi.extraction.llm.anthropic_provider import AnthropicProvider

    config = LLMConfig(api_key="test-key", model="claude-haiku-4-5-20251001")
    provider = AnthropicProvider(config)

    mock_message = Message(
        id="m",
        type="message",
        role="assistant",
        content=[{"type": "text", "text": "ok"}],
        model="claude-haiku-4-5-20251001",
        stop_reason="end_turn",
        usage=Usage(input_tokens=1, output_tokens=1),
    )

    add_calls: list[tuple[int, dict[str, str]]] = []

    def _capture_add(amount: int, attributes: dict[str, str] | None = None) -> None:
        add_calls.append((amount, attributes or {}))

    info_calls: list[dict] = []

    async def _capture_info(event: str, **kw: object) -> None:
        info_calls.append({"event": event, **kw})

    async def no_sleep(_d: float) -> None:
        return None

    # Bind the counter to a local before patch.object, so the lru_cache
    # is sealed deliberately rather than as an implicit side effect of
    # evaluating the with-statement arguments.
    retry_counter = _get_anthropic_retry_counter()
    with (
        patch.object(provider._client.messages, "create", new_callable=AsyncMock) as mock_create,
        patch.object(retry_counter, "add", side_effect=_capture_add),
        patch("gubbi.extraction.llm.anthropic_provider.logger.info", side_effect=_capture_info),
        patch("asyncio.sleep", new=no_sleep),
    ):
        mock_create.side_effect = [
            APIConnectionError(request=MagicMock()),
            mock_message,
        ]
        await provider._call_with_retry({})

    # One retry attempt => one counter add with result=retried, one INFO log
    retried = [c for c in add_calls if c[1].get("result") == "retried"]
    assert len(retried) == 1
    assert retried[0][1].get("error_class") == "LLMTransientError"

    retry_attempt_logs = [c for c in info_calls if c["event"] == "anthropic_retry_attempt"]
    assert len(retry_attempt_logs) == 1
    assert retry_attempt_logs[0]["attempt"] == 1
    assert retry_attempt_logs[0]["error_class"] == "LLMTransientError"
    assert retry_attempt_logs[0]["exhausted"] is False
    assert retry_attempt_logs[0]["delay_seconds"] >= 1.0


@pytest.mark.asyncio
async def test_retry_observability_emits_exhausted_on_budget_end() -> None:
    """When the retry budget runs out the counter reads result=exhausted and logs anthropic_retry_exhausted."""
    from gubbi.config import LLMConfig
    from gubbi.constants import ANTHROPIC_MAX_RETRIES
    from gubbi.extraction.llm.anthropic_provider import AnthropicProvider

    config = LLMConfig(api_key="test-key", model="claude-haiku-4-5-20251001")
    provider = AnthropicProvider(config)

    add_calls: list[tuple[int, dict[str, str]]] = []
    info_calls: list[dict] = []

    def _capture_add(amount: int, attributes: dict[str, str] | None = None) -> None:
        add_calls.append((amount, attributes or {}))

    async def _capture_info(event: str, **kw: object) -> None:
        info_calls.append({"event": event, **kw})

    async def no_sleep(_d: float) -> None:
        return None

    # Bind the counter to a local (see helper-binding rationale at the
    # first patch.object site above).
    retry_counter = _get_anthropic_retry_counter()
    with (
        patch.object(provider._client.messages, "create", new_callable=AsyncMock) as mock_create,
        patch.object(retry_counter, "add", side_effect=_capture_add),
        patch("gubbi.extraction.llm.anthropic_provider.logger.info", side_effect=_capture_info),
        patch("asyncio.sleep", new=no_sleep),
    ):
        mock_create.side_effect = [
            APIConnectionError(request=MagicMock()) for _ in range(ANTHROPIC_MAX_RETRIES)
        ]
        with pytest.raises(LLMTransientError):
            await provider._call_with_retry({})

    exhausted_counter = [c for c in add_calls if c[1].get("result") == "exhausted"]
    assert len(exhausted_counter) == 1
    assert exhausted_counter[0][1].get("error_class") == "LLMTransientError"

    exhausted_logs = [c for c in info_calls if c["event"] == "anthropic_retry_exhausted"]
    assert len(exhausted_logs) == 1
    assert exhausted_logs[0]["error_class"] == "LLMTransientError"


# ---------------------------------------------------------------------------
# _classify_error in extract_conversation maps via the LLM* hierarchy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (LLMRateLimitError("x"), "llm_rate_limited"),
        (LLMTransientError("x"), "llm_provider_error"),
        (LLMPermanentError("x"), "llm_provider_error"),
        (LLMProviderError("x"), "llm_provider_error"),
        (RuntimeError("oops"), "internal_error"),
        (ValueError("v"), "internal_error"),
    ],
)
def test_classify_error_uses_llm_hierarchy(exc: Exception, expected: str) -> None:
    """Worker classify_error reads only LLM*; vendor types are never imported here."""
    from gubbi.extraction.jobs.extract_conversation import _classify_error

    assert _classify_error(exc) == expected


def test_extract_conversation_module_does_not_import_anthropic() -> None:
    """S7 M2 contract: extract_conversation.py must not import the vendor SDK."""
    import gubbi.extraction.jobs.extract_conversation as mod

    # The classify_error function is the only place that historically used
    # anthropic.*; verify the module dict has no `anthropic` symbol now.
    assert "anthropic" not in mod.__dict__, (
        "anthropic should not be imported in extract_conversation.py "
        "after B2 -- _classify_error uses LLM* abstraction"
    )


# Backward compat: ensure the original retry behavior tests still pass shape.
# The legacy test file (test_anthropic_retry.py) keeps its own coverage; this
# file focuses on the new abstraction.


@pytest.mark.asyncio
async def test_jitter_still_added_under_translation() -> None:
    """Backoff jitter is preserved after the translation refactor."""
    from gubbi.config import LLMConfig
    from gubbi.extraction.llm.anthropic_provider import AnthropicProvider

    config = LLMConfig(api_key="test-key", model="claude-haiku-4-5-20251001")
    provider = AnthropicProvider(config)

    mock_message = Message(
        id="m",
        type="message",
        role="assistant",
        content=[{"type": "text", "text": "ok"}],
        model="claude-haiku-4-5-20251001",
        stop_reason="end_turn",
        usage=Usage(input_tokens=1, output_tokens=1),
    )

    sleep_durations: list[float] = []
    original_sleep = asyncio.sleep

    async def record_sleep(duration: float) -> None:
        sleep_durations.append(duration)
        await original_sleep(0)

    with patch.object(provider._client.messages, "create", new_callable=AsyncMock) as mock_create:
        mock_create.side_effect = [
            RateLimitError(message="rl", response=MagicMock(status_code=429), body=None),
            mock_message,
        ]
        with patch("asyncio.sleep", new=record_sleep):
            await provider._call_with_retry({})

    assert len(sleep_durations) == 1
    assert sleep_durations[0] >= 1.0
    assert sleep_durations[0] <= 1.0 + (1.0 * 0.1)
