"""Test-only LLMProvider stub for gubbi-testbench D-tier.

Selected by the worker via the ``JOURNAL_LLM_PROVIDER`` env var (factory
registry at ``gubbi/extraction/worker.py:_PROVIDER_FACTORIES``). The
testbench D-tier compose sets ``JOURNAL_LLM_PROVIDER=fake`` so the
worker boots against this in-process stub and the suite does not dial
``api.anthropic.com``. Set ``JOURNAL_LLM_PROVIDER=anthropic`` (with
``ANTHROPIC_API_KEY`` set) for occasional pre-release manual runs that
exercise the real provider.

Failure injection is via a marker substring in the conversation
content. Two markers:

  - ``FAKE_LLM_FAIL`` -- raises :class:`LLMTransientError`. The
    extraction retry loop classifies this as transient, retries, and
    eventually fails the job after the retry budget is exhausted.
    Exercises the pre-charge refund path in
    ``extraction/jobs/extract_conversation.py``.
  - ``FAKE_LLM_FAIL_PERMANENT`` -- raises :class:`LLMPermanentError`,
    skipping retries and going straight to the permanent-failure
    path.

Markers are checked against the JOINED content of all messages so the
test can plant the marker in any role (user message, assistant turn).
The strings are namespaced (``FAKE_LLM_*``) to avoid colliding with
real ingest content; in production this provider is never wired in
the first place, so collision is not a runtime risk.

The canned LLMResponse shape is keyed off the ``output_schema`` argument's
property set, so the provider returns the right canned dict regardless
of call order or worker concurrency. State-free / safe under ARQ
``max_jobs=10``.

Token-count and cost estimates mirror Haiku 4.5 pricing exactly
($1/MTok input, $5/MTok output) so the budget pre-charge math matches
what the production AnthropicProvider would report -- a regression in
``BudgetHelper`` accounting surfaces the same way under either provider.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from gubbi.extraction.llm.provider import (
    LLMMessage,
    LLMPermanentError,
    LLMProvider,
    LLMResponse,
    LLMTransientError,
)

__all__: list[str] = [
    "FakeLLMProvider",
]

_DEFAULT_MODEL = "fake-haiku-1"

# Canned token counts -- chosen to match a typical Haiku 4.5 categorize
# call (~100 input prompt tokens + ~50 output schema response tokens).
# Determinism is the contract: the budget pre-charge / refund math
# below depends on these being constant per call.
_CANNED_INPUT_TOKENS = 100
_CANNED_OUTPUT_TOKENS = 50

# Haiku 4.5 pricing per gubbi/extraction/llm/anthropic_provider.py:104-106
# ($1/MTok input, $5/MTok output). Hardcoded here rather than imported
# so a regression in AnthropicProvider's pricing constant does not
# silently change the fake's behaviour.
_INPUT_PRICE_PER_MILLION = 1.00
_OUTPUT_PRICE_PER_MILLION = 5.00


class FakeLLMProvider(LLMProvider):
    """Stub provider that returns canned dict responses + marker-driven failure injection.

    Constructor takes no arguments to match the factory call site in
    ``gubbi/extraction/worker.py:_PROVIDER_FACTORIES`` where the swap is
    one line. Subclasses or future per-test variants can override
    ``model`` if needed.
    """

    FAIL_MARKER: str = "FAKE_LLM_FAIL"
    PERMANENT_MARKER: str = "FAKE_LLM_FAIL_PERMANENT"

    def __init__(self, model: str = _DEFAULT_MODEL) -> None:
        self._model = model

    async def complete(
        self,
        messages: list[LLMMessage],
        system_prompt: str,
        output_schema: Mapping[str, Any] | None = None,
    ) -> LLMResponse:
        """Return a canned LLMResponse, or raise per the failure markers above."""
        joined_content = " ".join(m.get("content", "") for m in messages)
        # Permanent-marker check FIRST -- it is a strict superstring of
        # FAIL_MARKER, so checking FAIL_MARKER first would shadow it.
        if self.PERMANENT_MARKER in joined_content:
            raise LLMPermanentError(
                f"FakeLLMProvider: permanent-failure marker "
                f"{self.PERMANENT_MARKER!r} found in messages"
            )
        if self.FAIL_MARKER in joined_content:
            raise LLMTransientError(
                f"FakeLLMProvider: transient-failure marker "
                f"{self.FAIL_MARKER!r} found in messages"
            )

        content = _canned_content_for_schema(output_schema)
        return LLMResponse(
            content=content,
            input_tokens=_CANNED_INPUT_TOKENS,
            output_tokens=_CANNED_OUTPUT_TOKENS,
            model=self._model,
        )

    def estimate_cost_cents(self, input_tokens: int, output_tokens: int) -> float:
        """Mirror AnthropicProvider Haiku 4.5 pricing for deterministic budget math."""
        # ``round`` to 6 decimals matches the production rounding
        # convention. Determinism is the invariant the pre-charge /
        # refund test depends on.
        cents = (
            input_tokens / 1_000_000 * _INPUT_PRICE_PER_MILLION * 100
            + output_tokens / 1_000_000 * _OUTPUT_PRICE_PER_MILLION * 100
        )
        return round(cents, 6)


def _canned_content_for_schema(
    output_schema: Mapping[str, Any] | None,
) -> str | dict[str, Any]:
    """Return the dict that ExtractionService expects for this schema.

    Two schema shapes are exercised today (gubbi/extraction/service.py:37
    and :57): the categorize schema (topic_path / topic_title / summary
    / confidence) and the extract-entries schema (entries: [{content,
    reasoning, tags, ...}]). Distinguished by the property set so a
    future third schema with a fresh property name surfaces as
    "FakeLLMProvider canned response shape drift" rather than silently
    returning the wrong dict.

    A ``None`` schema (free-form completion) returns a string so the
    contract matches AnthropicProvider's text-mode return shape.
    """
    if output_schema is None:
        return "fake free-form response from FakeLLMProvider"

    properties = output_schema.get("properties", {})
    if "topic_path" in properties:
        return {
            "topic_path": "testbench/fake",
            "topic_title": "Testbench Fake Topic",
            "summary": "Canned categorization summary from FakeLLMProvider.",
            "confidence": 0.9,
        }
    if "entries" in properties:
        return {
            "entries": [
                {
                    "content": "fake decision marker from FakeLLMProvider",
                    "reasoning": "canned reasoning from FakeLLMProvider",
                    "tags": ["fake", "testbench"],
                }
            ]
        }
    raise ValueError(
        f"FakeLLMProvider has no canned response for output_schema with "
        f"properties {sorted(properties.keys())}; add a branch in "
        "_canned_content_for_schema."
    )
