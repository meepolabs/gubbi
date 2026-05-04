import asyncio
import random
from collections.abc import Mapping
from typing import Any

from anthropic import AsyncAnthropic, RateLimitError

from gubbi.config import LLMConfig
from gubbi.extraction.llm.provider import LLMMessage, LLMProvider, LLMResponse

# Model pricing in $USD per million tokens (input, output).
# Values are approximate and should be updated when pricing changes.
_MODEL_PRICING: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    "claude-sonnet-4-20250514": (3.00, 15.00),
    "claude-opus-4-20250514": (15.00, 75.00),
}

# Fallback prices match Haiku 4.5 -- update both here and in _MODEL_PRICING if pricing changes.
_DEFAULT_INPUT_PRICE = 1.00
_DEFAULT_OUTPUT_PRICE = 5.00

# Default completion cap used when caller does not pass a token budget.
_DEFAULT_MAX_TOKENS = 4096


class AnthropicProvider(LLMProvider):
    def __init__(self, config: LLMConfig) -> None:
        self._api_key = config.api_key
        self._model = config.model or "claude-haiku-4-5-20251001"
        self._client = AsyncAnthropic(api_key=self._api_key)

    async def complete(
        self,
        messages: list[LLMMessage],
        system_prompt: str,
        output_schema: Mapping[str, Any] | None = None,
    ) -> LLMResponse:
        system_block: dict[str, Any] = {
            "type": "text",
            "text": system_prompt,
            "cache_control": {"type": "ephemeral"},
        }

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "system": [system_block],
            "max_tokens": _DEFAULT_MAX_TOKENS,
        }

        if output_schema is not None:
            kwargs["tools"] = [
                {
                    "name": "respond",
                    "description": ("Respond with structured output matching the requested schema"),
                    "input_schema": output_schema,
                },
            ]
            kwargs["tool_choice"] = {"type": "tool", "name": "respond"}

        response = await self._call_with_retry(kwargs)

        if output_schema is not None:
            content: str | dict[str, Any] = ""
            block_types: list[str] = []
            for block in response.content:
                block_types.append(getattr(block, "type", type(block).__name__))
                if hasattr(block, "name") and block.name == "respond":
                    content = block.input if isinstance(block.input, dict) else block.input
                    break
            if content == "":
                raise ValueError(
                    f"Anthropic model {response.model} returned no 'respond' tool block; "
                    f"blocks={block_types}"
                )
        else:
            content = response.content[0].text if response.content else ""

        return LLMResponse(
            content=content,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=response.model,
        )

    async def _call_with_retry(self, kwargs: dict[str, Any]) -> Any:
        max_retries = 5
        base_delay = 1.0
        for attempt in range(max_retries):
            try:
                return await self._client.messages.create(**kwargs)
            except RateLimitError:
                if attempt < max_retries - 1:
                    base = base_delay * (2**attempt)
                    # Not cryptographic -- simple jitter to prevent thundering herd.
                    jitter = random.uniform(0, base * 0.1)  # noqa: S311
                    delay = base + jitter
                    await asyncio.sleep(delay)
                    continue
                raise
        raise RuntimeError("Retry loop exited unexpectedly")

    def estimate_cost_cents(self, input_tokens: int, output_tokens: int) -> float:
        pricing = _MODEL_PRICING.get(self._model, (_DEFAULT_INPUT_PRICE, _DEFAULT_OUTPUT_PRICE))
        input_cost = (input_tokens / 1_000_000) * pricing[0] * 100
        output_cost = (output_tokens / 1_000_000) * pricing[1] * 100
        return round(input_cost + output_cost, 6)
