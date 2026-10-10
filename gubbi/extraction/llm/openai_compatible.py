"""LLM provider for any OpenAI-compatible chat-completions endpoint.

One ``complete()`` is one HTTP request: the OpenAI SDK's own retries are
disabled and PydanticAI is called through its direct request API, so no
output validation re-prompt can issue a hidden second call.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import structlog
from openai import APITimeoutError, AsyncOpenAI
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
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.output import OutputObjectDefinition
from pydantic_ai.providers.openai import OpenAIProvider

from gubbi.extraction.llm.provider import (
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
    from collections.abc import Mapping

    import httpx2
    from pydantic_ai.settings import ModelSettings
    from pydantic_ai.usage import RequestUsage

__all__: list[str] = [
    "OpenAICompatibleProvider",
]

logger = structlog.get_logger(__name__)

_OUTPUT_NAME = "final_result"

# Upstreams that report these outside the standard usage fields (for
# example a top-level ``cache_read_input_tokens``) land in
# ``RequestUsage.details`` under the wire name.
_CACHED_INPUT_DETAIL_KEY = "cache_read_input_tokens"
_REASONING_DETAIL_KEY = "reasoning_tokens"

_RETRYABLE_STATUS_CODES = frozenset({408})
_RATE_LIMIT_STATUS = 429
_SERVER_ERROR_FLOOR = 500


class OpenAICompatibleProvider(LLMProvider):
    """Single-attempt structured completions against an OpenAI-compatible API."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        input_price_per_mtok: float,
        output_price_per_mtok: float,
        max_output_tokens: int,
        timeout: float,
        http_client: httpx2.AsyncClient | None = None,
    ) -> None:
        self._model = model
        self._input_price_per_mtok = input_price_per_mtok
        self._output_price_per_mtok = output_price_per_mtok
        self._max_output_tokens = max_output_tokens
        client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            max_retries=0,
            timeout=timeout,
            http_client=http_client,
        )
        self._chat_model = OpenAIChatModel(model, provider=OpenAIProvider(openai_client=client))

    async def complete(
        self,
        messages: list[LLMMessage],
        system_prompt: str,
        output_schema: Mapping[str, Any] | None = None,
    ) -> LLMResponse:
        """Send one chat-completions request; constrain to output_schema when supplied."""
        settings: ModelSettings = {"max_tokens": self._max_output_tokens}
        try:
            response = await model_request(
                self._chat_model,
                _to_model_messages(messages, system_prompt),
                model_settings=settings,
                model_request_parameters=_request_parameters(output_schema),
                instrument=False,
            )
        except (ModelAPIError, UnexpectedModelBehavior) as exc:
            raise _translate_error(exc) from exc

        usage = _map_usage(response.usage)
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

    def estimate_cost_cents(self, input_tokens: int, output_tokens: int) -> float:
        """Compute call cost in cents from the configured per-million-token prices."""
        input_cost = input_tokens / 1_000_000 * self._input_price_per_mtok * 100
        output_cost = output_tokens / 1_000_000 * self._output_price_per_mtok * 100
        return round(input_cost + output_cost, 6)

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


def _map_usage(usage: RequestUsage) -> LLMUsage:
    details = usage.details
    return LLMUsage(
        input=usage.input_tokens,
        output=usage.output_tokens,
        cached_input=usage.cache_read_tokens or details.get(_CACHED_INPUT_DETAIL_KEY, 0),
        reasoning=details.get(_REASONING_DETAIL_KEY, 0),
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
        raise LLMSchemaError("model output is not valid JSON", usage=usage) from None
    if not isinstance(parsed, dict):
        raise LLMSchemaError("model output is not a JSON object", usage=usage)
    return parsed


def _translate_error(exc: ModelAPIError | UnexpectedModelBehavior) -> LLMProviderError:
    """Map a PydanticAI error onto the provider-agnostic hierarchy.

    Messages name only the failure class and status code: upstream error
    bodies can echo request content, so they are never copied.
    """
    if isinstance(exc, ModelHTTPError):
        status = exc.status_code
        message = f"upstream returned HTTP {status}"
        if status == _RATE_LIMIT_STATUS:
            return LLMRateLimitError(message)
        if status in _RETRYABLE_STATUS_CODES or status >= _SERVER_ERROR_FLOOR:
            return LLMTransientError(message)
        return LLMPermanentError(message)
    if isinstance(exc, ModelAPIError):
        if isinstance(exc.__cause__, APITimeoutError):
            return LLMTimeoutError("upstream request timed out")
        return LLMTransientError("upstream connection failed")
    return LLMTransientError("upstream returned a malformed response")
