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


@runtime_checkable
class LLMProvider(Protocol):
    async def complete(
        self,
        messages: list[LLMMessage],
        system_prompt: str,
        output_schema: Mapping[str, Any] | None = None,
    ) -> LLMResponse: ...

    def estimate_cost_cents(self, input_tokens: int, output_tokens: int) -> float: ...
