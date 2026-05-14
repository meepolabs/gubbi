from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, TypedDict, runtime_checkable


class LLMMessage(TypedDict):
    role: str
    content: str


@dataclass
class LLMResponse:
    content: str | dict[str, Any]
    input_tokens: int
    output_tokens: int
    model: str


# ---------------------------------------------------------------------------
# Provider-agnostic exception hierarchy.
#
# Concrete providers (anthropic_provider, future LLM gateway client) translate
# vendor SDK exceptions into these classes at the provider boundary so that
# callers (extract_conversation, retry loops, _classify_error) reason about
# transient/permanent classification without importing vendor SDKs.
# ---------------------------------------------------------------------------


class LLMProviderError(Exception):
    """Base for all errors raised by an LLMProvider implementation.

    Use directly only for errors that defy transient/permanent classification.
    Prefer the subclasses below for known categories.
    """


class LLMTransientError(LLMProviderError):
    """Transient failure: retry with backoff is appropriate.

    Examples: connection reset, timeout, 5xx server error, upstream overload.
    Retry budgets and backoff policy live in the provider's own retry loop.
    """


class LLMRateLimitError(LLMTransientError):
    """Upstream signalled a rate limit (HTTP 429 or equivalent).

    Always retryable; callers may wish to log this distinctly from generic
    transient errors for capacity planning.
    """


class LLMPermanentError(LLMProviderError):
    """Permanent failure: do not retry.

    Examples: authentication failure, schema-validation rejection, malformed
    request, unsupported model. The retry loop must propagate immediately.
    """


@runtime_checkable
class LLMProvider(Protocol):
    async def complete(
        self,
        messages: list[LLMMessage],
        system_prompt: str,
        output_schema: Mapping[str, Any] | None = None,
    ) -> LLMResponse:
        """Complete the chat sequence; constrain to output_schema when supplied."""
        ...

    def estimate_cost_cents(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate the call cost in cents from token counts."""
        ...
