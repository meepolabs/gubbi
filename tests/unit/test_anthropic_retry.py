"""Test AnthropicProvider only retries on RateLimitError (M-9.12)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from anthropic._exceptions import RateLimitError
from anthropic.types import Message, Usage


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
async def test_anthropic_non_rate_limit_raises_immediately() -> None:
    """Non-rate-limit exceptions should NOT be retried -- they propagate at once."""
    from gubbi.config import LLMConfig
    from gubbi.extraction.llm.anthropic_provider import AnthropicProvider

    config = LLMConfig(api_key="test-key", model="claude-haiku-4-5-20251001")
    provider = AnthropicProvider(config)

    class MyAPIError(Exception):
        pass

    with patch.object(provider._client.messages, "create", new_callable=AsyncMock) as mock_create:
        mock_create.side_effect = MyAPIError("something broke")

        with pytest.raises(MyAPIError):
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
