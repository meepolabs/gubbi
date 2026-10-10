from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, NotRequired, Protocol, TypedDict, runtime_checkable


class LLMMessage(TypedDict):
    role: str
    content: str
    timestamp: NotRequired[str | None]


@dataclass(frozen=True)
class LLMUsage:
    """Token counts reported by the upstream for one call."""

    input: int = 0
    output: int = 0
    cached_input: int = 0
    reasoning: int = 0


@dataclass(frozen=True)
class LLMResponse:
    content: str | dict[str, Any]
    input_tokens: int
    output_tokens: int
    model: str
    finish_reason: str = "stop"
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0
    attempts: int = 1


# ---------------------------------------------------------------------------
# Provider-agnostic exception hierarchy.
#
# Concrete providers (openai_compatible) translate vendor SDK exceptions into
# these classes at the provider boundary so that callers (extract_conversation,
# retry loops, _classify_error) reason about transient/permanent
# classification without importing vendor SDKs.
# ---------------------------------------------------------------------------


class LLMProviderError(Exception):
    """Base for all errors raised by an LLMProvider implementation.

    Use directly only for errors that defy transient/permanent classification.
    Prefer the subclasses below for known categories.

    ``usage`` carries the token counts of the call that produced the error
    when the upstream reported any: a refused or truncated call still bills.
    Messages never carry model output or refusal text.
    """

    def __init__(self, message: str = "", *, usage: LLMUsage | None = None) -> None:
        super().__init__(message)
        self.usage = usage


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


class LLMTimeoutError(LLMTransientError):
    """The request exceeded its per-attempt timeout."""


class LLMPermanentError(LLMProviderError):
    """Permanent failure: do not retry.

    Examples: authentication failure, schema-validation rejection, malformed
    request, unsupported model. The retry loop must propagate immediately.
    """


class LLMRefusalError(LLMPermanentError):
    """The model declined to answer (finish reason ``content_filter``)."""


class LLMSchemaError(LLMPermanentError):
    """The model output does not parse as the requested structured output."""


class LLMTruncatedError(LLMProviderError):
    """The model hit its output-token cap (finish reason ``length``).

    Neither transient nor permanent: the caller decides whether a smaller
    input is worth another call. Partial output is never returned.
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
