from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class LLMResponse:
    content: str | dict[str, Any]
    input_tokens: int
    output_tokens: int
    model: str


class LLMProvider(ABC):
    @abstractmethod
    async def complete(
        self,
        messages: list[dict],
        system_prompt: str,
        output_schema: dict | None = None,
    ) -> LLMResponse: ...

    @abstractmethod
    def estimate_cost_cents(self, input_tokens: int, output_tokens: int) -> float: ...
