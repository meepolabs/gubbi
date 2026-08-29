import asyncio
import functools
import random
from collections.abc import Mapping
from typing import Any

import structlog
from anthropic import (
    APIConnectionError,
    APIResponseValidationError,
    APIStatusError,
    AsyncAnthropic,
    AuthenticationError,
    BadRequestError,
    ConflictError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
)
from opentelemetry.metrics import Counter, get_meter

from gubbi.config import LLMConfig
from gubbi.constants import ANTHROPIC_MAX_RETRIES, ANTHROPIC_REQUEST_TIMEOUT_SECS
from gubbi.extraction.llm.provider import (
    LLMMessage,
    LLMPermanentError,
    LLMProvider,
    LLMProviderError,
    LLMRateLimitError,
    LLMResponse,
    LLMTransientError,
)

logger = structlog.get_logger(__name__)


# Retry observability counter. Emitted per retry attempt with an attribute
# describing whether the attempt was followed by another retry or marked the
# end of the budget.
@functools.lru_cache(maxsize=1)
def _get_anthropic_retry_counter() -> Counter:
    """Lazily create the anthropic-retry counter against the live meter.

    The previous module-scope ``_meter.create_counter``
    bound at import time, well before ``configure_otel`` ran during the
    FastAPI lifespan -- so the counter held a NoOp instrument and silently
    discarded every ``.add(...)``. Deferring creation to first call (and
    re-priming via ``rebind_metrics_after_configure``) ensures the counter
    binds to the SDK provider configured at lifespan time. Mirrors the
    canonical pattern in ``gubbi.telemetry.metrics.initialize_metrics``.
    """
    return get_meter("gubbi").create_counter(
        name="anthropic.retry_count_total",
        description="Anthropic provider retry attempts, partitioned by result and error_class",
        unit="1",
    )


def _translate_anthropic_error(exc: Exception) -> LLMProviderError:
    """Translate an anthropic SDK exception to the provider-agnostic hierarchy.

    Mapping:
      RateLimitError                            -> LLMRateLimitError
      APIConnectionError (incl APITimeoutError) -> LLMTransientError
      APIResponseValidationError                -> LLMTransientError
        (response-shape failures often correlate with upstream flakes; retry)
      APIStatusError with status_code >= 500
        (incl InternalServerError)              -> LLMTransientError
      4xx APIStatusError subclasses             -> LLMPermanentError
        (BadRequestError, AuthenticationError, PermissionDeniedError,
         NotFoundError, UnprocessableEntityError, ConflictError)
      Anything else                             -> LLMPermanentError
    """
    if isinstance(exc, RateLimitError):
        return LLMRateLimitError(str(exc))
    if isinstance(exc, APIConnectionError):
        # APITimeoutError is a subclass of APIConnectionError; covered here.
        return LLMTransientError(str(exc))
    if isinstance(exc, APIResponseValidationError):
        # Response-shape validation failures often correlate with upstream
        # flakes (truncated/garbled responses). Treat as transient so retries
        # kick in.
        return LLMTransientError(str(exc))
    if isinstance(exc, APIStatusError):
        # APIStatusError.status_code is typed Any in the anthropic SDK stubs;
        # int() coercion keeps the comparison honest under mypy --strict.
        # InternalServerError is APIStatusError(status_code=500), also covered.
        if int(exc.status_code) >= 500:
            return LLMTransientError(str(exc))
        # 4xx subclasses (BadRequest/Auth/Permission/NotFound/Unprocessable/
        # Conflict) all fall through here. Explicit isinstance branches are
        # listed for clarity even though the >= 500 / else split already
        # routes them correctly.
        if isinstance(
            exc,
            BadRequestError
            | AuthenticationError
            | PermissionDeniedError
            | NotFoundError
            | UnprocessableEntityError
            | ConflictError,
        ):
            return LLMPermanentError(str(exc))
        return LLMPermanentError(str(exc))
    return LLMPermanentError(str(exc))


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
        self._client = AsyncAnthropic(
            api_key=self._api_key,
            timeout=ANTHROPIC_REQUEST_TIMEOUT_SECS,
        )

    async def complete(
        self,
        messages: list[LLMMessage],
        system_prompt: str,
        output_schema: Mapping[str, Any] | None = None,
    ) -> LLMResponse:
        """Call Anthropic Messages with prompt-cached system block; force tool-use for schemas."""
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
                    content = block.input if isinstance(block.input, dict) else str(block.input)
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
        """Invoke the Anthropic SDK with retry/backoff on transient errors.

        Retry policy:
          * Vendor exceptions are translated to LLM* at the boundary.
          * LLMTransientError (incl. LLMRateLimitError) -> retry with backoff.
          * LLMPermanentError -> raise immediately.

        Observability:
          * Per-retry INFO log `anthropic_retry_attempt` with attempt + delay
            + error_class.
          * On exhaustion INFO log `anthropic_retry_exhausted` with attempt
            + error_class.
          * Counter `anthropic.retry_count_total` with
            ``{result: retried|exhausted, error_class: <name>}``.
        """
        max_retries = ANTHROPIC_MAX_RETRIES
        base_delay = 1.0
        for attempt in range(max_retries):
            try:
                return await self._client.messages.create(**kwargs)
            except Exception as exc:
                translated = _translate_anthropic_error(exc)
                error_class = type(translated).__name__
                if not isinstance(translated, LLMTransientError):
                    raise translated from exc
                if attempt < max_retries - 1:
                    base = base_delay * (2**attempt)
                    # Not cryptographic -- simple jitter to prevent thundering herd.
                    jitter = random.uniform(0, base * 0.1)  # noqa: S311
                    delay = base + jitter
                    await logger.info(
                        "anthropic_retry_attempt",
                        attempt=attempt + 1,
                        delay_seconds=delay,
                        error_class=error_class,
                        exhausted=False,
                    )
                    _get_anthropic_retry_counter().add(
                        1,
                        attributes={"result": "retried", "error_class": error_class},
                    )
                    await asyncio.sleep(delay)
                    continue
                await logger.info(
                    "anthropic_retry_exhausted",
                    attempt=attempt + 1,
                    error_class=error_class,
                )
                _get_anthropic_retry_counter().add(
                    1,
                    attributes={"result": "exhausted", "error_class": error_class},
                )
                raise translated from exc
        raise RuntimeError("Retry loop exited unexpectedly")

    def estimate_cost_cents(self, input_tokens: int, output_tokens: int) -> float:
        """Compute call cost in cents from per-million-token model pricing."""
        pricing = _MODEL_PRICING.get(self._model, (_DEFAULT_INPUT_PRICE, _DEFAULT_OUTPUT_PRICE))
        input_cost = (input_tokens / 1_000_000) * pricing[0] * 100
        output_cost = (output_tokens / 1_000_000) * pricing[1] * 100
        return round(input_cost + output_cost, 6)


__all__: list[str] = [
    "AnthropicProvider",
]
