"""LLM provider for Anthropic's native Messages API.

Retries, the time budget, finish-reason checks and error translation are
shared with the other PydanticAI-backed providers in
:mod:`gubbi.extraction.llm.model_provider`; this module builds the Anthropic
model and maps its usage.

Structured output must reach the wire as ``output_config.format`` with the
JSON schema. PydanticAI decides per model name whether a model accepts that,
and its list lags new releases; a model it does not know would have native
output refused before any request. The model profile here declares the
capability for whichever model is configured.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, ClassVar, Literal

from anthropic import APITimeoutError, AsyncAnthropic
from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
from pydantic_ai.profiles.anthropic import AnthropicModelProfile
from pydantic_ai.providers.anthropic import AnthropicProvider as PydanticAIAnthropicProvider

from gubbi.constants import LLM_MAX_OUTPUT_TOKENS, LLM_REQUEST_TIMEOUT_SECS
from gubbi.extraction.llm.model_provider import PydanticAIModelProvider
from gubbi.extraction.llm.provider import LLMUsage

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    import httpx2
    from pydantic_ai.settings import ModelSettings
    from pydantic_ai.usage import RequestUsage

__all__: list[str] = [
    "AnthropicEffort",
    "AnthropicProvider",
]

AnthropicEffort = Literal["low", "medium", "high", "xhigh", "max"]

# Passed explicitly so an ``ANTHROPIC_BASE_URL`` in the process environment,
# which the SDK would otherwise read, cannot redirect requests.
_DEFAULT_BASE_URL = "https://api.anthropic.com"

_THINKING_DETAIL_KEY = "thinking_tokens"


class AnthropicProvider(PydanticAIModelProvider):
    """Structured completions against the Anthropic Messages API, with bounded retries.

    Retry, time-budget and error behavior are those of
    :class:`~gubbi.extraction.llm.model_provider.PydanticAIModelProvider`;
    529 (overloaded) is a 5xx and is retried like one.

    ``effort`` is sent as ``output_config.effort`` when set; ``None`` leaves
    the model's default. Thinking is never configured, so the model's
    default applies and its tokens bill as output.
    """

    _timeout_errors: ClassVar[tuple[type[BaseException], ...]] = (APITimeoutError,)

    def __init__(
        self,
        api_key: str,
        model: str,
        input_price_per_mtok: float,
        output_price_per_mtok: float,
        base_url: str = "",
        effort: AnthropicEffort | None = None,
        max_output_tokens: int = LLM_MAX_OUTPUT_TOKENS,
        timeout: float = LLM_REQUEST_TIMEOUT_SECS,
        http_client: httpx2.AsyncClient | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        client = AsyncAnthropic(
            api_key=api_key,
            base_url=base_url or _DEFAULT_BASE_URL,
            max_retries=0,
            timeout=timeout,
            http_client=http_client,
        )
        chat_model = AnthropicModel(
            model,
            provider=PydanticAIAnthropicProvider(anthropic_client=client),
            profile=AnthropicModelProfile(supports_json_schema_output=True),
        )
        super().__init__(
            model=model,
            chat_model=chat_model,
            input_price_per_mtok=input_price_per_mtok,
            output_price_per_mtok=output_price_per_mtok,
            max_output_tokens=max_output_tokens,
            sleep=sleep,
            clock=clock,
        )
        self._effort = effort

    def _model_settings(self) -> ModelSettings:
        settings: AnthropicModelSettings = {"max_tokens": self._max_output_tokens}
        if self._effort is not None:
            settings["anthropic_effort"] = self._effort
        return settings

    def _map_usage(self, usage: RequestUsage) -> LLMUsage:
        # PydanticAI's ``input_tokens`` for Anthropic already sums uncached
        # input, cache reads and cache writes, so cached input is a subset.
        return LLMUsage(
            input=usage.input_tokens,
            output=usage.output_tokens,
            cached_input=usage.cache_read_tokens,
            reasoning=usage.details.get(_THINKING_DETAIL_KEY, 0),
        )
