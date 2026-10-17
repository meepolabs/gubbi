"""Shared request machinery for providers built on a PydanticAI model.

The only retry loop is the one in :meth:`PydanticAIModelProvider.complete`:
concrete providers disable their SDK's own retries and PydanticAI is called
through its direct request API, so no output validation re-prompt can issue a
hidden extra call. Each attempt is one HTTP request.

A concrete provider supplies the PydanticAI model, the timeout exception its
SDK raises, and how the upstream's usage block maps onto :class:`LLMUsage`;
everything else lives here.

Translated errors never chain the PydanticAI or SDK exception they replace:
those carry upstream error bodies and response validation input, which can
echo conversation text into tracebacks.
"""

from __future__ import annotations

import abc
import asyncio
import dataclasses
import email.utils
import functools
import json
import random
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, ClassVar

import structlog
from opentelemetry.metrics import Counter, get_meter
from pydantic_ai.direct import model_request
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError, UnexpectedModelBehavior
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.output import OutputObjectDefinition

from gubbi.constants import (
    LLM_BACKOFF_BASE_SECS,
    LLM_BACKOFF_CAP_SECS,
    LLM_MAX_ATTEMPTS,
    LLM_MAX_OUTPUT_TOKENS,
)
from gubbi.extraction.llm.provider import (
    LLMDeadlineError,
    LLMMessage,
    LLMPermanentError,
    LLMProvider,
    LLMProviderError,
    LLMRateLimitError,
    LLMRefusalError,
    LLMResponse,
    LLMSchemaError,
    LLMTimeoutError,
    LLMTransientError,
    LLMTruncatedError,
    LLMUsage,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping

    from pydantic_ai.models import Model
    from pydantic_ai.settings import ModelSettings
    from pydantic_ai.usage import RequestUsage

__all__: list[str] = [
    "PydanticAIModelProvider",
]

logger = structlog.get_logger(__name__)

_OUTPUT_NAME = "final_result"

_RETRYABLE_STATUS_CODES = frozenset({408})
_RATE_LIMIT_STATUS = 429
_SERVER_ERROR_FLOOR = 500
_RETRY_AFTER_HEADER = "retry-after"


@functools.lru_cache(maxsize=1)
def _get_llm_retry_counter() -> Counter:
    """Create the retry counter against the meter current at first call.

    Deferred so it binds to the SDK provider installed at startup rather than
    the import-time NoOp; ``rebind_metrics_after_configure`` clears and
    re-primes it.
    """
    return get_meter("gubbi").create_counter(
        name="llm.retry_count_total",
        description="LLM provider retries, partitioned by result and error_class",
        unit="1",
    )


class PydanticAIModelProvider(LLMProvider, abc.ABC):
    """Structured completions through a PydanticAI model, with bounded retries.

    Timeouts, connection failures, 408, 429 and 5xx are retried up to
    ``LLM_MAX_ATTEMPTS`` total requests with jittered exponential backoff,
    waiting at least as long as a ``Retry-After`` header asks. Every other
    failure, including refusal, truncation and unparseable output, is raised
    after the request that produced it. Raised errors carry the number of
    requests made in ``attempts`` and the usage of every request that
    reported any in ``usage``.

    A ``time_budget`` passed to ``complete()`` caps each attempt at the time
    left and ends the call with :class:`LLMDeadlineError` when no time is
    left for another attempt, including after the wait a retry needs and
    after that wait has actually elapsed.

    Timed-out attempts report no usage although the upstream may have billed
    them; they are counted in ``unmetered_attempts`` on the response or the
    raised error.
    """

    _timeout_errors: ClassVar[tuple[type[BaseException], ...]]
    """SDK exceptions that mark a request timeout when they cause a ``ModelAPIError``."""

    def __init__(
        self,
        *,
        model: str,
        chat_model: Model,
        input_price_per_mtok: float,
        output_price_per_mtok: float,
        max_output_tokens: int = LLM_MAX_OUTPUT_TOKENS,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._model = model
        self._chat_model = chat_model
        self._sleep = sleep
        self._clock = clock
        self._input_price_per_mtok = input_price_per_mtok
        self._output_price_per_mtok = output_price_per_mtok
        self._max_output_tokens = max_output_tokens

    @abc.abstractmethod
    def _map_usage(self, usage: RequestUsage) -> LLMUsage:
        """Map the upstream's usage onto :class:`LLMUsage`; ``input`` includes cached input."""

    def _model_settings(self) -> ModelSettings:
        return {"max_tokens": self._max_output_tokens}

    async def complete(
        self,
        messages: list[LLMMessage],
        system_prompt: str,
        output_schema: Mapping[str, Any] | None = None,
        *,
        time_budget: float | None = None,
    ) -> LLMResponse:
        """Complete the chat, retrying transient failures; constrain to output_schema."""
        model_messages = _to_model_messages(messages, system_prompt)
        parameters = _request_parameters(output_schema)
        ends_at = None if time_budget is None else self._clock() + time_budget
        remaining = self._remaining(ends_at)
        if remaining is not None and remaining <= 0:
            raise LLMDeadlineError("time budget spent before the first attempt", attempts=0)
        unmetered = 0
        for attempt in range(1, LLM_MAX_ATTEMPTS + 1):
            try:
                response = await self._attempt(model_messages, parameters, output_schema, remaining)
            except LLMProviderError as exc:
                unmetered += isinstance(exc, LLMTimeoutError)
                exc.attempts = attempt
                exc.unmetered_attempts = unmetered
                if not isinstance(exc, LLMTransientError):
                    raise
                delay = _retry_delay(attempt, exc.retry_after)
                remaining = self._remaining(ends_at)
                if _budget_ends_retries(delay, remaining):
                    await self._record_retry(attempt, exc, "deadline", None)
                    raise LLMDeadlineError(
                        "time budget too short for another attempt",
                        attempts=attempt,
                        unmetered_attempts=unmetered,
                    ) from None
                await self._record_retry(
                    attempt, exc, "exhausted" if delay is None else "retried", delay
                )
                if delay is None:
                    raise
                await self._sleep(delay)
                remaining = self._remaining(ends_at)
                if remaining is not None and remaining <= 0:
                    raise LLMDeadlineError(
                        "time budget spent while waiting to retry",
                        attempts=attempt,
                        unmetered_attempts=unmetered,
                    ) from None
                continue
            return dataclasses.replace(response, attempts=attempt, unmetered_attempts=unmetered)
        raise AssertionError("retry loop exited without returning or raising")

    def estimate_cost_cents(self, input_tokens: int, output_tokens: int) -> float:
        """Compute call cost in cents from the configured per-million-token prices."""
        input_cost = input_tokens / 1_000_000 * self._input_price_per_mtok * 100
        output_cost = output_tokens / 1_000_000 * self._output_price_per_mtok * 100
        return round(input_cost + output_cost, 6)

    def _remaining(self, ends_at: float | None) -> float | None:
        return None if ends_at is None else ends_at - self._clock()

    async def _attempt(
        self,
        model_messages: list[ModelMessage],
        parameters: ModelRequestParameters,
        output_schema: Mapping[str, Any] | None,
        time_limit: float | None,
    ) -> LLMResponse:
        translated: LLMProviderError
        try:
            async with asyncio.timeout(time_limit):
                response = await model_request(
                    self._chat_model,
                    model_messages,
                    model_settings=self._model_settings(),
                    model_request_parameters=parameters,
                    instrument=False,
                )
        except TimeoutError:
            translated = LLMTimeoutError("upstream request timed out")
        except (ModelAPIError, UnexpectedModelBehavior) as exc:
            translated = _translate_error(exc, self._timeout_errors)
        else:
            return await self._to_response(response, output_schema)
        # Raised outside the except block so the upstream exception, whose
        # text can quote request content, is neither chained nor attached
        # as context.
        raise translated

    async def _to_response(
        self, response: ModelResponse, output_schema: Mapping[str, Any] | None
    ) -> LLMResponse:
        usage = self._map_usage(response.usage)
        await self._check_finish_reason(response, usage)
        return LLMResponse(
            content=_extract_content(response, usage, structured=output_schema is not None),
            input_tokens=usage.input,
            output_tokens=usage.output,
            model=response.model_name or self._model,
            finish_reason=response.finish_reason or "stop",
            cached_input_tokens=usage.cached_input,
            reasoning_tokens=usage.reasoning,
        )

    async def _record_retry(
        self, attempt: int, error: LLMTransientError, result: str, delay: float | None
    ) -> None:
        error_class = type(error).__name__
        _get_llm_retry_counter().add(1, attributes={"result": result, "error_class": error_class})
        await logger.info(
            "llm_retry_attempt" if result == "retried" else f"llm_retry_{result}",
            model=self._model,
            attempt=attempt,
            error_class=error_class,
            delay_seconds=delay,
        )

    async def _check_finish_reason(self, response: ModelResponse, usage: LLMUsage) -> None:
        if response.finish_reason == "content_filter":
            await self._log_unusable(response, usage)
            raise LLMRefusalError("model refused the request", usage=usage)
        if response.finish_reason == "length":
            await self._log_unusable(response, usage)
            raise LLMTruncatedError("model output hit the output-token cap", usage=usage)

    async def _log_unusable(self, response: ModelResponse, usage: LLMUsage) -> None:
        await logger.warning(
            "llm_response_unusable",
            model=self._model,
            finish_reason=response.finish_reason,
            input_tokens=usage.input,
            output_tokens=usage.output,
            cached_input_tokens=usage.cached_input,
            reasoning_tokens=usage.reasoning,
        )


def _retry_delay(attempt: int, retry_after: float | None) -> float | None:
    """Seconds to wait before the next attempt, or None when retries are over.

    A ``Retry-After`` longer than ``LLM_BACKOFF_CAP_SECS`` also ends the
    retries: waiting it out would hold the job past any useful deadline.
    """
    if attempt >= LLM_MAX_ATTEMPTS:
        return None
    if retry_after is not None and retry_after > LLM_BACKOFF_CAP_SECS:
        return None
    ceiling = min(LLM_BACKOFF_CAP_SECS, LLM_BACKOFF_BASE_SECS * 2 ** (attempt - 1))
    # Not cryptographic: jitter only spreads concurrent workers' retries apart.
    backoff = random.uniform(ceiling / 2, ceiling)  # noqa: S311
    return max(backoff, retry_after or 0.0)


def _budget_ends_retries(delay: float | None, remaining: float | None) -> bool:
    """True when the time budget is spent, or the next retry's wait would spend it.

    Checked even when retries are otherwise over, so a final attempt cut
    short by the budget reports the budget, not a timeout.
    """
    if remaining is None:
        return False
    return remaining <= 0 or (delay is not None and delay >= remaining)


def _retry_after_seconds(exc: ModelHTTPError) -> float | None:
    """Parse ``Retry-After`` (delta-seconds or HTTP-date) from an HTTP error."""
    if not exc.headers:
        return None
    value = exc.headers.get(_RETRY_AFTER_HEADER, "").strip()
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, (when - datetime.now(UTC)).total_seconds())


def _to_model_messages(messages: list[LLMMessage], system_prompt: str) -> list[ModelMessage]:
    history: list[ModelMessage] = []
    pending: list[ModelRequestPart] = [SystemPromptPart(content=system_prompt)]
    for message in messages:
        role = message["role"]
        content = message["content"]
        if role == "assistant":
            if pending:
                history.append(ModelRequest(parts=pending))
                pending = []
            history.append(ModelResponse(parts=[TextPart(content=content)]))
        elif role == "system":
            pending.append(SystemPromptPart(content=content))
        else:
            pending.append(UserPromptPart(content=content))
    if pending:
        history.append(ModelRequest(parts=pending))
    return history


def _request_parameters(output_schema: Mapping[str, Any] | None) -> ModelRequestParameters:
    if output_schema is None:
        return ModelRequestParameters(output_mode="text", allow_text_output=True)
    return ModelRequestParameters(
        output_mode="native",
        output_object=OutputObjectDefinition(
            json_schema=dict(output_schema),
            name=_OUTPUT_NAME,
            strict=True,
        ),
        allow_text_output=False,
    )


def _extract_content(
    response: ModelResponse, usage: LLMUsage, *, structured: bool
) -> str | dict[str, Any]:
    text = "".join(part.content for part in response.parts if isinstance(part, TextPart))
    if not structured:
        return text
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        pass
    else:
        return _json_object(parsed, usage)
    # Raised outside the except block: the decode error's ``doc`` holds the
    # model output, so it must not ride along as context.
    raise LLMSchemaError("model output is not valid JSON", usage=usage)


def _json_object(parsed: object, usage: LLMUsage) -> dict[str, Any]:
    if not isinstance(parsed, dict):
        raise LLMSchemaError("model output is not a JSON object", usage=usage)
    return parsed


def _translate_error(
    exc: ModelAPIError | UnexpectedModelBehavior,
    timeout_errors: tuple[type[BaseException], ...],
) -> LLMProviderError:
    """Map a PydanticAI error onto the provider-agnostic hierarchy.

    Messages name only the failure class and status code: upstream error
    bodies can echo request content, so they are never copied. The
    ``Retry-After`` wait is copied onto retryable errors.
    """
    if isinstance(exc, ModelHTTPError):
        status = exc.status_code
        message = f"upstream returned HTTP {status}"
        if status == _RATE_LIMIT_STATUS:
            return LLMRateLimitError(message, retry_after=_retry_after_seconds(exc))
        if status in _RETRYABLE_STATUS_CODES or status >= _SERVER_ERROR_FLOOR:
            return LLMTransientError(message, retry_after=_retry_after_seconds(exc))
        return LLMPermanentError(message)
    if isinstance(exc, ModelAPIError):
        if isinstance(exc.__cause__, timeout_errors):
            return LLMTimeoutError("upstream request timed out")
        return LLMTransientError("upstream connection failed")
    return LLMTransientError("upstream returned a malformed response")
