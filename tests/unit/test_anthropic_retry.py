"""Test AnthropicProvider only retries on transient errors.

Retries are gated by LLMTransientError (the provider-agnostic
hierarchy in gubbi.extraction.llm.provider), and vendor exceptions are
translated at the boundary. These tests assert the post-translation
behavior: vendor exceptions get retried when transient, surface as LLM*
when permanent or budget-exhausted.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from anthropic import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    InternalServerError,
)
from anthropic._exceptions import RateLimitError
from anthropic.types import Message, Usage

from gubbi.extraction.llm.provider import (
    LLMPermanentError,
    LLMRateLimitError,
)


@pytest.mark.asyncio
async def test_anthropic_retry_only_on_rate_limit_error() -> None:
    """Only RateLimitError triggers retries; other exceptions propagate immediately."""
    from gubbi.config import LLMConfig
    from gubbi.extraction.llm.anthropic_provider import AnthropicProvider

    config = LLMConfig(api_key="test-key", model="claude-haiku-4-5-20251001")
    provider = AnthropicProvider(config)

    mock_message = Message(
        id="msg_123",
        type="message",
        role="assistant",
        content=[{"type": "text", "text": "hello"}],
        model="claude-haiku-4-5-20251001",
        stop_reason="end_turn",
        usage=Usage(input_tokens=10, output_tokens=5),
    )

    # Patch client.messages.create to simulate failures.
    with patch.object(provider._client.messages, "create", new_callable=AsyncMock) as mock_create:
        # First call raises RateLimitError, second succeeds.
        mock_create.side_effect = [
            RateLimitError(
                message="Rate limit exceeded",
                response=MagicMock(status_code=429),
                body=None,
            ),
            mock_message,
        ]

        result = await provider._call_with_retry({})

    assert result == mock_message

    # Verify it retried exactly twice (1 failure + 1 success).
    assert mock_create.call_count == 2


@pytest.mark.asyncio
async def test_anthropic_unrecognized_exception_translated_to_permanent() -> None:
    """An unknown exception type should translate to LLMPermanentError and not be retried."""
    from gubbi.config import LLMConfig
    from gubbi.extraction.llm.anthropic_provider import AnthropicProvider

    config = LLMConfig(api_key="test-key", model="claude-haiku-4-5-20251001")
    provider = AnthropicProvider(config)

    class MyAPIError(Exception):
        pass

    with patch.object(provider._client.messages, "create", new_callable=AsyncMock) as mock_create:
        mock_create.side_effect = MyAPIError("something broke")

        with pytest.raises(LLMPermanentError):
            await provider._call_with_retry({})

    # Should have been called exactly once -- no retry.
    assert mock_create.call_count == 1


@pytest.mark.asyncio
async def test_anthropic_retry_jitter_is_added() -> None:
    """Retry backoff includes jitter to prevent thundering herd."""
    from gubbi.config import LLMConfig
    from gubbi.extraction.llm.anthropic_provider import AnthropicProvider

    config = LLMConfig(api_key="test-key", model="claude-haiku-4-5-20251001")
    provider = AnthropicProvider(config)

    mock_message = Message(
        id="msg_1",
        type="message",
        role="assistant",
        content=[{"type": "text", "text": "ok"}],
        model="claude-haiku-4-5-20251001",
        stop_reason="end_turn",
        usage=Usage(input_tokens=1, output_tokens=1),
    )

    # Rate limit 2 times, then succeed. Track sleep duration.
    sleep_durations: list[float] = []
    original_sleep = asyncio.sleep

    async def record_sleep(duration: float) -> None:
        sleep_durations.append(duration)
        await original_sleep(0)  # zero actual delay for speed

    with patch.object(provider._client.messages, "create", new_callable=AsyncMock) as mock_create:
        mock_create.side_effect = [
            RateLimitError(
                message="Rate limit exceeded",
                response=MagicMock(status_code=429),
                body=None,
            ),
            RateLimitError(
                message="Rate limit exceeded",
                response=MagicMock(status_code=429),
                body=None,
            ),
            mock_message,
        ]
        with patch("asyncio.sleep", new=record_sleep):
            result = await provider._call_with_retry({})

    assert result == mock_message
    assert len(sleep_durations) == 2  # two sleep calls for two retries

    # Verify jitter is non-zero on both retries (base delay * 2^attempt + random).
    for i, duration in enumerate(sleep_durations):
        base_delay = 1.0 * (2**i)
        jitter_range = base_delay * 0.1
        assert duration >= base_delay, f"delay {duration} < base {base_delay}"
        assert (
            duration <= base_delay + jitter_range
        ), f"delay {duration} exceeds max base+jitter ({base_delay + jitter_range})"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("exc_factory", "label"),
    [
        (
            lambda: APIConnectionError(request=MagicMock()),
            "APIConnectionError",
        ),
        (
            lambda: APITimeoutError(request=MagicMock()),
            "APITimeoutError",
        ),
        (
            lambda: InternalServerError(
                message="boom",
                response=MagicMock(status_code=500),
                body=None,
            ),
            "InternalServerError",
        ),
        (
            lambda: APIStatusError(
                message="service unavailable",
                response=httpx.Response(
                    status_code=503,
                    request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
                ),
                body=None,
            ),
            "APIStatusError(503)",
        ),
    ],
)
async def test_anthropic_retries_on_broadened_transient_errors(
    exc_factory: Callable[[], Exception], label: str
) -> None:
    """APIConnectionError / APITimeoutError / InternalServerError now retry."""
    from gubbi.config import LLMConfig
    from gubbi.extraction.llm.anthropic_provider import AnthropicProvider

    config = LLMConfig(api_key="test-key", model="claude-haiku-4-5-20251001")
    provider = AnthropicProvider(config)

    mock_message = Message(
        id="msg_1",
        type="message",
        role="assistant",
        content=[{"type": "text", "text": "ok"}],
        model="claude-haiku-4-5-20251001",
        stop_reason="end_turn",
        usage=Usage(input_tokens=1, output_tokens=1),
    )

    with patch.object(provider._client.messages, "create", new_callable=AsyncMock) as mock_create:
        mock_create.side_effect = [exc_factory(), mock_message]  # type: ignore[operator]

        async def no_sleep(_d: float) -> None:
            return None

        with patch("asyncio.sleep", new=no_sleep):
            result = await provider._call_with_retry({})

    assert result == mock_message, f"{label} should be retried and ultimately succeed"
    assert mock_create.call_count == 2, f"{label} should retry exactly once before success"


@pytest.mark.asyncio
async def test_anthropic_does_not_retry_api_status_client_error() -> None:
    """APIStatusError 4xx should propagate as LLMPermanentError without retry."""
    from gubbi.config import LLMConfig
    from gubbi.extraction.llm.anthropic_provider import AnthropicProvider

    config = LLMConfig(api_key="test-key", model="claude-haiku-4-5-20251001")
    provider = AnthropicProvider(config)
    bad_request = APIStatusError(
        message="bad request",
        response=httpx.Response(
            status_code=400,
            request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
        ),
        body=None,
    )

    with patch.object(provider._client.messages, "create", new_callable=AsyncMock) as mock_create:
        mock_create.side_effect = bad_request

        with pytest.raises(LLMPermanentError):
            await provider._call_with_retry({})

    assert mock_create.call_count == 1


@pytest.mark.asyncio
async def test_anthropic_rate_limit_exhaustion_surfaces_as_llm_rate_limit() -> None:
    """When the retry budget is exhausted on rate limits, callers see LLMRateLimitError."""
    from gubbi.config import LLMConfig
    from gubbi.constants import ANTHROPIC_MAX_RETRIES
    from gubbi.extraction.llm.anthropic_provider import AnthropicProvider

    config = LLMConfig(api_key="test-key", model="claude-haiku-4-5-20251001")
    provider = AnthropicProvider(config)

    async def no_sleep(_d: float) -> None:
        return None

    with (
        patch.object(provider._client.messages, "create", new_callable=AsyncMock) as mock_create,
        patch("asyncio.sleep", new=no_sleep),
    ):
        mock_create.side_effect = [
            RateLimitError(
                message="Rate limit",
                response=MagicMock(status_code=429),
                body=None,
            )
            for _ in range(ANTHROPIC_MAX_RETRIES)
        ]

        with pytest.raises(LLMRateLimitError):
            await provider._call_with_retry({})

    assert mock_create.call_count == ANTHROPIC_MAX_RETRIES
