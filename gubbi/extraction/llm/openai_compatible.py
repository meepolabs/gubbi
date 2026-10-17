"""LLM provider for any OpenAI-compatible chat-completions endpoint.

Retries, the time budget, finish-reason checks and error translation are
shared with the other PydanticAI-backed providers in
:mod:`gubbi.extraction.llm.model_provider`; this module builds the OpenAI
chat model and maps its usage.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, ClassVar, cast

from openai import APIStatusError, APITimeoutError, AsyncOpenAI, DefaultAsyncHttpxClient, omit
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from gubbi.constants import LLM_MAX_OUTPUT_TOKENS, LLM_REQUEST_TIMEOUT_SECS
from gubbi.extraction.llm.http_transport import pinned_transport
from gubbi.extraction.llm.model_provider import PydanticAIModelProvider
from gubbi.extraction.llm.provider import LLMUsage

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    import httpx2
    from pydantic_ai.usage import RequestUsage

__all__: list[str] = [
    "OpenAICompatibleProvider",
]

# Upstreams that report these outside the standard usage fields (for
# example a top-level ``cache_read_input_tokens``) land in
# ``RequestUsage.details`` under the wire name.
_CACHED_INPUT_DETAIL_KEY = "cache_read_input_tokens"
_REASONING_DETAIL_KEY = "reasoning_tokens"

# AsyncOpenAI refuses to construct without a key, but many self-hosted
# endpoints need none. This stand-in satisfies the client and is not a
# credential.
_PLACEHOLDER_API_KEY = "unused"

# ``OPENAI_LOG`` lowers this logger's level at SDK import; at debug the SDK
# logs each request's JSON body, which carries conversation text.
_SDK_LOGGER = "openai"


class OpenAICompatibleProvider(PydanticAIModelProvider):
    """Structured completions against an OpenAI-compatible API, with bounded retries.

    Retry, time-budget and error behavior are those of
    :class:`~gubbi.extraction.llm.model_provider.PydanticAIModelProvider`.

    An empty ``api_key`` sends a placeholder bearer, for endpoints that need
    no key.
    """

    _timeout_errors: ClassVar[tuple[type[BaseException], ...]] = (APITimeoutError,)
    _status_errors: ClassVar[tuple[type[BaseException], ...]] = (APIStatusError,)

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        input_price_per_mtok: float,
        output_price_per_mtok: float,
        max_output_tokens: int = LLM_MAX_OUTPUT_TOKENS,
        timeout: float = LLM_REQUEST_TIMEOUT_SECS,
        http_client: httpx2.AsyncClient | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        logging.getLogger(_SDK_LOGGER).setLevel(logging.WARNING)
        bearer_key = api_key or _PLACEHOLDER_API_KEY
        client = AsyncOpenAI(
            base_url=base_url,
            api_key=bearer_key,
            max_retries=0,
            timeout=timeout,
            http_client=http_client or _default_http_client(),
            default_headers=_pinned_headers(bearer_key),
        )
        super().__init__(
            model=model,
            chat_model=OpenAIChatModel(model, provider=OpenAIProvider(openai_client=client)),
            input_price_per_mtok=input_price_per_mtok,
            output_price_per_mtok=output_price_per_mtok,
            max_output_tokens=max_output_tokens,
            sleep=sleep,
            clock=clock,
        )

    def _map_usage(self, usage: RequestUsage) -> LLMUsage:
        details = usage.details
        return LLMUsage(
            input=usage.input_tokens,
            output=usage.output_tokens,
            cached_input=usage.cache_read_tokens or details.get(_CACHED_INPUT_DETAIL_KEY, 0),
            reasoning=details.get(_REASONING_DETAIL_KEY, 0),
        )


def _pinned_headers(api_key: str) -> dict[str, str]:
    """Headers that keep ``OPENAI_*`` environment variables off the wire.

    The SDK merges ``OPENAI_CUSTOM_HEADERS`` and adds organization and
    project headers from ``OPENAI_ORG_ID`` / ``OPENAI_PROJECT_ID``; only
    explicit defaults outrank them.
    """
    # The SDK types ``default_headers`` as ``str`` values but removes a
    # header whose value is ``omit``.
    absent = cast("str", omit)
    return {
        "Authorization": f"Bearer {api_key}",
        "OpenAI-Organization": absent,
        "OpenAI-Project": absent,
    }


def _default_http_client() -> httpx2.AsyncClient:
    """The SDK's default client, minus redirects and anything read from the environment."""
    return DefaultAsyncHttpxClient(
        follow_redirects=False, trust_env=False, transport=pinned_transport()
    )
