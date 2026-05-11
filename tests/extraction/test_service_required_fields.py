"""Test ExtractionService.categorize_conversation validates required fields.

Item 4 from quick-fixes plan: a malformed LLM response (missing required
key like ``topic_path``) must raise ExtractionError before the dict reads
that would otherwise produce a confusing KeyError deep in the call site.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from gubbi.extraction.llm.provider import LLMResponse
from gubbi.extraction.service import ExtractionError, ExtractionService


@pytest.mark.unit
@pytest.mark.asyncio
async def test_categorize_raises_extraction_error_when_required_field_missing(
    tmp_path: Any,
) -> None:
    """LLM dict missing ``topic_path`` raises ExtractionError, not KeyError."""
    # Provide a stub prompts dir so the service _read_prompt call doesn't fail.
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "categorize.md").write_text("system prompt", encoding="utf-8")

    llm = MagicMock()
    # Response missing topic_path -- has only 3 of the 4 required fields.
    llm.complete = AsyncMock(
        return_value=LLMResponse(
            content={
                "topic_title": "Acme",
                "summary": "discussed Acme decisions",
                "confidence": 0.9,
            },
            input_tokens=10,
            output_tokens=5,
            model="claude-haiku-4-5-20251001",
        )
    )

    service = ExtractionService(llm)
    service._prompts_dir = prompts_dir  # type: ignore[attr-defined]

    with pytest.raises(ExtractionError, match="topic_path"):
        await service.categorize_conversation([], [])
