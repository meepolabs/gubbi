"""Live smoke: the extraction job end to end against a real LLM provider.

``JOURNAL_LLM_PROVIDER`` picks the provider: ``openai_compatible`` (default)
or ``anthropic``. Skipped unless ``GUBBI_LIVE_SMOKE_CONVERSATION`` and the
provider's own settings are set: ``JOURNAL_LLM_BASE_URL`` and
``JOURNAL_LLM_MODEL`` for ``openai_compatible`` (``JOURNAL_LLM_API_KEY`` may
stay unset for keyless gateways); ``JOURNAL_LLM_API_KEY`` and
``JOURNAL_LLM_MODEL`` for ``anthropic`` (``JOURNAL_LLM_BASE_URL`` and
``JOURNAL_LLM_EFFORT`` optional). ``GUBBI_LIVE_SMOKE_CONVERSATION`` names a
JSON file holding one conversation as ``{"platform": ..., "messages":
[{"role", "content", "timestamp"}, ...]}`` (``platform`` optional); a long one
(several chunks) is the point.

Everything is real: the provider, chunking, the pipeline, embeddings, dedupe
and persistence on the RLS-enforced test Postgres. The run's numbers are
logged as one ``live_extraction_smoke`` line (``-o log_cli=true``).
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

from gubbi.constants import LLM_MAX_OUTPUT_TOKENS
from gubbi.extraction.dedupe import normalize_text
from gubbi.extraction.jobs.extract_conversation import extract_conversation
from gubbi.extraction.llm.anthropic_messages import AnthropicEffort, AnthropicProvider
from gubbi.extraction.llm.openai_compatible import OpenAICompatibleProvider
from tests.fixtures.extraction_db import SeedMessage, job_ctx, seed_conversation, seed_job

if TYPE_CHECKING:
    from uuid import UUID

    import asyncpg

    from gubbi.crypto.cipher import ContentCipher
    from gubbi.extraction.llm.provider import LLMProvider

_PROVIDER = os.environ.get("JOURNAL_LLM_PROVIDER", "").strip().lower() or "openai_compatible"
_PROVIDER_ENV: dict[str, tuple[str, ...]] = {
    "openai_compatible": ("JOURNAL_LLM_BASE_URL", "JOURNAL_LLM_MODEL"),
    "anthropic": ("JOURNAL_LLM_API_KEY", "JOURNAL_LLM_MODEL"),
}
_REQUIRED_ENV = (*_PROVIDER_ENV.get(_PROVIDER, ()), "GUBBI_LIVE_SMOKE_CONVERSATION")
_MISSING_ENV = [name for name in _REQUIRED_ENV if not os.environ.get(name, "").strip()]
_SKIP_REASON = (
    f"live extraction smoke supports JOURNAL_LLM_PROVIDER in {sorted(_PROVIDER_ENV)}"
    if _PROVIDER not in _PROVIDER_ENV
    else f"live extraction smoke with {_PROVIDER} needs {', '.join(_REQUIRED_ENV)} set"
)

pytestmark = [
    pytest.mark.asyncio(loop_scope="session"),
    pytest.mark.integration,
    pytest.mark.skipif(
        _PROVIDER not in _PROVIDER_ENV or bool(_MISSING_ENV),
        reason=_SKIP_REASON,
    ),
]

logger = logging.getLogger(__name__)

_MIN_CHUNKS = 3
_REPORTED_COLUMNS = (
    "chunks",
    "llm_calls",
    "input_tokens",
    "output_tokens",
    "cached_input_tokens",
    "entries_created",
)
_KNOWN_PLATFORMS = frozenset({"chatgpt", "claude"})


def _load_conversation(path: Path) -> tuple[str | None, list[SeedMessage]]:
    """Read the conversation file into its platform and dated seed messages."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    platform = raw.get("platform")
    messages = [
        SeedMessage(
            role=str(item["role"]),
            content=str(item["content"]),
            timestamp=datetime.fromisoformat(item["timestamp"]) if item.get("timestamp") else None,
        )
        for item in raw["messages"]
    ]
    return (platform if platform in _KNOWN_PLATFORMS else None), messages


def _provider() -> LLMProvider:
    if _PROVIDER == "anthropic":
        effort = os.environ.get("JOURNAL_LLM_EFFORT", "").strip()
        return AnthropicProvider(
            api_key=os.environ["JOURNAL_LLM_API_KEY"],
            model=os.environ["JOURNAL_LLM_MODEL"],
            input_price_per_mtok=1.0,
            output_price_per_mtok=5.0,
            base_url=os.environ.get("JOURNAL_LLM_BASE_URL", ""),
            effort=cast("AnthropicEffort", effort) if effort else None,
            max_output_tokens=LLM_MAX_OUTPUT_TOKENS,
        )
    return OpenAICompatibleProvider(
        base_url=os.environ["JOURNAL_LLM_BASE_URL"],
        api_key=os.environ.get("JOURNAL_LLM_API_KEY", ""),
        model=os.environ["JOURNAL_LLM_MODEL"],
        input_price_per_mtok=1.0,
        output_price_per_mtok=5.0,
        max_output_tokens=LLM_MAX_OUTPUT_TOKENS,
    )


async def _stored_entries(
    admin_pool: asyncpg.Pool, cipher: ContentCipher, conversation_id: int
) -> list[tuple[str, date]]:
    async with admin_pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT content_encrypted, content_nonce, date FROM entries"
            " WHERE source_conversation_id = $1 ORDER BY id",
            conversation_id,
        )
    return [
        (cipher.decrypt(row["content_encrypted"], row["content_nonce"]), row["date"])
        for row in rows
    ]


async def _job_row(admin_pool: asyncpg.Pool, job_id: UUID) -> asyncpg.Record:
    async with admin_pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM extraction_jobs WHERE id = $1", job_id)
    assert row is not None
    return row


async def test_a_long_conversation_extracts_through_the_live_endpoint(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    cipher: ContentCipher,
    tenant_a: UUID,
) -> None:
    platform, messages = _load_conversation(Path(os.environ["GUBBI_LIVE_SMOKE_CONVERSATION"]))
    last_spoken = max(m.timestamp.date() for m in messages if m.timestamp is not None)
    conv_id = await seed_conversation(admin_pool, cipher, tenant_a, messages, platform=platform)
    job_id = await seed_job(app_pool, tenant_a, conv_id)
    started = time.monotonic()

    result = await extract_conversation(
        job_ctx(app_pool, cipher, _provider()), conv_id, str(tenant_a), str(job_id)
    )

    wall_seconds = round(time.monotonic() - started, 1)
    row = await _job_row(admin_pool, job_id)
    entries = await _stored_entries(admin_pool, cipher, conv_id)
    numbers = {name: row[name] for name in _REPORTED_COLUMNS}
    logger.info(
        "live_extraction_smoke %s",
        " ".join(
            f"{key}={value}"
            for key, value in {
                **numbers,
                "provider": _PROVIDER,
                "model": os.environ["JOURNAL_LLM_MODEL"],
                "topic_path": result["topic_path"],
                "wall_seconds": wall_seconds,
            }.items()
        ),
    )

    assert row["status"] == "completed", row["error_code"]
    assert row["chunks"] >= _MIN_CHUNKS
    assert row["llm_calls"] >= row["chunks"]
    keys = [(normalize_text(content), day) for content, day in entries]
    assert len(keys) == len(set(keys)), "two stored entries share normalized content and date"
    late = [(content, day) for content, day in entries if day > last_spoken]
    assert not late, f"entries dated after the last message ({last_spoken}): {late}"
