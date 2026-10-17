"""Acceptance tests: extraction of a thread that grew after it was last extracted.

The job runs end to end on an RLS-enforced Postgres: it reads the messages
above the conversation's extraction watermark plus a context tail, and moves
the watermark in the transaction that completes the job.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest
import structlog

from gubbi.constants import EXTRACTION_VERSION
from gubbi.extraction import charging
from gubbi.extraction.jobs.extract_conversation import extract_conversation
from gubbi.extraction.llm.fake_provider import FakeLLMProvider
from gubbi.extraction.llm.provider import LLMPermanentError
from gubbi.extraction.pipeline import PipelineFailure
from gubbi.extraction.slice_run import WatermarkMovedError
from tests.fixtures.extraction_db import SeedMessage, job_ctx, seed_conversation, seed_job
from tests.fixtures.extraction_job import ScriptedProvider, chunk_output, entry

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID

    import asyncpg

    from gubbi.crypto.cipher import ContentCipher
    from gubbi.extraction.llm.provider import LLMProvider

pytestmark = [pytest.mark.asyncio(loop_scope="session"), pytest.mark.integration]

_PLATFORM = "chatgpt"
_TOPIC = "travel/lisbon"
_FIRST_DAY = datetime(2025, 11, 1, 9, 0, tzinfo=UTC)


def _daily(contents: Sequence[str], roles: Sequence[str] | None = None) -> list[SeedMessage]:
    """One message per day starting at ``_FIRST_DAY``; all user turns unless ``roles`` says."""
    return [
        SeedMessage(
            role="user" if roles is None else roles[i],
            content=content,
            timestamp=_FIRST_DAY + timedelta(days=i),
        )
        for i, content in enumerate(contents)
    ]


def _plain(count: int) -> list[str]:
    return [f"Day {i}: I kept planning the Lisbon trip in detail." for i in range(count)]


async def _seed(
    admin_pool: asyncpg.Pool,
    app_pool: asyncpg.Pool,
    cipher: ContentCipher,
    user: UUID,
    messages: Sequence[SeedMessage],
    watermark: int | None,
) -> tuple[int, UUID]:
    conv_id = await seed_conversation(admin_pool, cipher, user, messages, platform=_PLATFORM)
    await admin_pool.execute(
        "UPDATE conversations SET extracted_through_position = $2 WHERE id = $1",
        conv_id,
        watermark,
    )
    return conv_id, await seed_job(app_pool, user, conv_id)


async def _run(
    app_pool: asyncpg.Pool,
    cipher: ContentCipher,
    provider: LLMProvider,
    user: UUID,
    conv_id: int,
    job_id: UUID,
) -> dict[str, Any]:
    return await extract_conversation(
        job_ctx(app_pool, cipher, provider), conv_id, str(user), str(job_id)
    )


async def _watermark(admin_pool: asyncpg.Pool, conv_id: int) -> int | None:
    return await admin_pool.fetchval(  # type: ignore[no-any-return]
        "SELECT extracted_through_position FROM conversations WHERE id = $1", conv_id
    )


async def _entries(admin_pool: asyncpg.Pool, conv_id: int) -> list[asyncpg.Record]:
    return list(
        await admin_pool.fetch(
            "SELECT id, date FROM entries WHERE source_conversation_id = $1"
            " AND conversation_id IS DISTINCT FROM $1 ORDER BY id",
            conv_id,
        )
    )


async def _job_status(admin_pool: asyncpg.Pool, job_id: UUID) -> tuple[str, str | None]:
    row = await admin_pool.fetchrow(
        "SELECT status, error_code FROM extraction_jobs WHERE id = $1", job_id
    )
    assert row is not None
    return row["status"], row["error_code"]


async def _jobs_of(admin_pool: asyncpg.Pool, conv_id: int) -> list[asyncpg.Record]:
    return list(
        await admin_pool.fetch(
            "SELECT id, status FROM extraction_jobs WHERE conversation_id = $1 ORDER BY created_at",
            conv_id,
        )
    )


async def _append_message(
    admin_pool: asyncpg.Pool, cipher: ContentCipher, user: UUID, conv_id: int, text: str
) -> None:
    """Store one more user message at the next position, as a merge append does."""
    ciphertext, nonce = cipher.encrypt(text)
    await admin_pool.execute(
        """
        INSERT INTO messages
            (conversation_id, user_id, role, timestamp, position,
             content_encrypted, content_nonce, search_vector)
        SELECT $1, $2, 'user', now(), max(position) + 1, $3, $4, to_tsvector('english', 'x')
        FROM messages WHERE conversation_id = $1
        """,
        conv_id,
        user,
        ciphertext,
        nonce,
    )


class _DuringCall(ScriptedProvider):
    """Runs ``action`` before the first model call, then replays the script."""

    def __init__(self, steps: list[Any], action: Any) -> None:
        super().__init__(steps)
        self._action = action

    async def complete(self, *args: Any, **kwargs: Any) -> Any:
        if self._action is not None:
            action, self._action = self._action, None
            await action()
        return await super().complete(*args, **kwargs)


# -- reading the slice ----------------------------------------------------------


async def test_only_the_new_slice_entry_persists_dated_by_its_own_message(
    app_pool: asyncpg.Pool, admin_pool: asyncpg.Pool, cipher: ContentCipher, tenant_a: UUID
) -> None:
    contents = _plain(10)
    contents[4] = "Day 4: I chose the hotel. FAKE_LLM_ENTRY:tail-hotel"
    contents[8] = "Day 8: I booked the flights. FAKE_LLM_ENTRY:new-flights"
    conv_id, job_id = await _seed(admin_pool, app_pool, cipher, tenant_a, _daily(contents), 7)

    result = await _run(app_pool, cipher, FakeLLMProvider(), tenant_a, conv_id, job_id)

    (row,) = await _entries(admin_pool, conv_id)
    assert result["entries_created"] == 1
    assert row["date"] == (_FIRST_DAY + timedelta(days=8)).date()
    assert row["date"] != (_FIRST_DAY + timedelta(days=6)).date()
    assert await _watermark(admin_pool, conv_id) == 9
    assert await _job_status(admin_pool, job_id) == ("completed", None)


async def test_a_never_extracted_thread_is_read_whole_and_advances_to_its_last_position(
    app_pool: asyncpg.Pool, admin_pool: asyncpg.Pool, cipher: ContentCipher, tenant_a: UUID
) -> None:
    conv_id, job_id = await _seed(admin_pool, app_pool, cipher, tenant_a, _daily(_plain(5)), None)
    provider = ScriptedProvider([chunk_output(_TOPIC, [entry("Planned Lisbon", source=0)])])

    await _run(app_pool, cipher, provider, tenant_a, conv_id, job_id)

    (sent,) = provider.user_contents
    assert "Boundary:" not in sent
    assert all(f"Day {i}:" in sent for i in range(5))
    (row,) = await _entries(admin_pool, conv_id)
    assert row["date"] == _FIRST_DAY.date()
    assert await _watermark(admin_pool, conv_id) == 4


# -- the watermark moves only with a completed job ------------------------------


async def test_the_watermark_moves_in_the_transaction_that_completes_the_job(
    app_pool: asyncpg.Pool, admin_pool: asyncpg.Pool, cipher: ContentCipher, tenant_a: UUID
) -> None:
    conv_id, job_id = await _seed(admin_pool, app_pool, cipher, tenant_a, _daily(_plain(6)), 3)
    seen: dict[str, Any] = {}
    real_settle = charging.settle_attempt

    async def observing_settle(conn: asyncpg.Connection, *args: Any, **kwargs: Any) -> Any:
        seen["inside"] = await conn.fetchval(
            "SELECT extracted_through_position FROM conversations WHERE id = $1", conv_id
        )
        seen["outside"] = await _watermark(admin_pool, conv_id)
        return await real_settle(conn, *args, **kwargs)

    provider = ScriptedProvider([chunk_output(_TOPIC, [entry("Booked flights", source=5)])])
    with patch("gubbi.extraction.persist.settle_attempt", observing_settle):
        await _run(app_pool, cipher, provider, tenant_a, conv_id, job_id)

    assert seen == {"inside": 5, "outside": 3}
    assert await _watermark(admin_pool, conv_id) == 5
    assert await _job_status(admin_pool, job_id) == ("completed", None)


async def test_a_failure_after_the_advance_rolls_the_watermark_back(
    app_pool: asyncpg.Pool, admin_pool: asyncpg.Pool, cipher: ContentCipher, tenant_a: UUID
) -> None:
    conv_id, job_id = await _seed(admin_pool, app_pool, cipher, tenant_a, _daily(_plain(6)), 3)
    provider = ScriptedProvider([chunk_output(_TOPIC, [entry("Booked flights", source=5)])])

    async def failing_settle(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("settle failed")

    with (
        patch("gubbi.extraction.persist.settle_attempt", failing_settle),
        pytest.raises(RuntimeError, match="settle failed"),
    ):
        await _run(app_pool, cipher, provider, tenant_a, conv_id, job_id)

    assert await _watermark(admin_pool, conv_id) == 3
    assert await _entries(admin_pool, conv_id) == []
    assert await _job_status(admin_pool, job_id) == ("failed", "internal_error")


async def test_a_provider_error_leaves_the_watermark_unchanged(
    app_pool: asyncpg.Pool, admin_pool: asyncpg.Pool, cipher: ContentCipher, tenant_a: UUID
) -> None:
    conv_id, job_id = await _seed(admin_pool, app_pool, cipher, tenant_a, _daily(_plain(6)), 3)
    provider = ScriptedProvider([LLMPermanentError("down")])

    with pytest.raises(PipelineFailure):
        await _run(app_pool, cipher, provider, tenant_a, conv_id, job_id)

    assert await _watermark(admin_pool, conv_id) == 3
    assert await _job_status(admin_pool, job_id) == ("failed", "llm_provider_error")


async def test_a_cancelled_run_leaves_the_watermark_unchanged(
    app_pool: asyncpg.Pool, admin_pool: asyncpg.Pool, cipher: ContentCipher, tenant_a: UUID
) -> None:
    conv_id, job_id = await _seed(admin_pool, app_pool, cipher, tenant_a, _daily(_plain(6)), 3)
    called = asyncio.Event()

    async def hang() -> None:
        called.set()
        await asyncio.Event().wait()

    provider = _DuringCall([chunk_output(_TOPIC)], hang)
    run = asyncio.create_task(_run(app_pool, cipher, provider, tenant_a, conv_id, job_id))
    await asyncio.wait_for(called.wait(), 10)
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run

    assert await _watermark(admin_pool, conv_id) == 3
    assert await _job_status(admin_pool, job_id) == ("failed", "deadline_exceeded")


# -- never backward ---------------------------------------------------------------


@pytest.mark.parametrize(
    "found",
    [
        pytest.param([entry("Booked flights", source=5)], id="entries"),
        pytest.param([], id="nothing-to-record"),
    ],
)
async def test_a_watermark_moved_by_a_concurrent_run_makes_this_run_write_nothing(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    cipher: ContentCipher,
    tenant_a: UUID,
    found: list[dict[str, Any]],
) -> None:
    conv_id, job_id = await _seed(admin_pool, app_pool, cipher, tenant_a, _daily(_plain(6)), 3)

    async def concurrent_advance() -> None:
        await admin_pool.execute(
            "UPDATE conversations SET extracted_through_position = 4 WHERE id = $1", conv_id
        )

    provider = _DuringCall([chunk_output(_TOPIC, found)], concurrent_advance)

    with pytest.raises(WatermarkMovedError):
        await _run(app_pool, cipher, provider, tenant_a, conv_id, job_id)

    state = await admin_pool.fetchrow(
        "SELECT processed_at, extracted_content_hash FROM conversations WHERE id = $1", conv_id
    )
    assert state is not None
    assert await _watermark(admin_pool, conv_id) == 4
    assert await _entries(admin_pool, conv_id) == []
    assert (state["processed_at"], state["extracted_content_hash"]) == (None, None)
    assert await admin_pool.fetchval("SELECT count(*) FROM topics WHERE path = $1", _TOPIC) == 0
    assert await _job_status(admin_pool, job_id) == ("failed", "internal_error")


# -- follow-up when the thread grows during a run ---------------------------------


@pytest.mark.parametrize(
    ("appended", "expected_follow_ups"),
    [pytest.param(True, 1, id="append-during-run"), pytest.param(False, 0, id="no-append")],
)
async def test_exactly_one_follow_up_when_messages_arrive_during_the_run(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    cipher: ContentCipher,
    tenant_a: UUID,
    appended: bool,
    expected_follow_ups: int,
) -> None:
    conv_id, job_id = await _seed(admin_pool, app_pool, cipher, tenant_a, _daily(_plain(6)), 3)

    async def append() -> None:
        await _append_message(admin_pool, cipher, tenant_a, conv_id, "I also rented a car.")

    provider = _DuringCall(
        [chunk_output(_TOPIC, [entry("Booked flights", source=5)])], append if appended else None
    )

    await _run(app_pool, cipher, provider, tenant_a, conv_id, job_id)

    jobs = await _jobs_of(admin_pool, conv_id)
    follow_ups = [j for j in jobs if j["id"] != job_id]
    assert await _job_status(admin_pool, job_id) == ("completed", None)
    assert [j["status"] for j in follow_ups] == ["pending"] * expected_follow_ups
    assert await _watermark(admin_pool, conv_id) == 5


# -- nothing new above the watermark ---------------------------------------------


async def test_an_edit_below_the_watermark_completes_unchanged_without_a_model_call(
    app_pool: asyncpg.Pool, admin_pool: asyncpg.Pool, cipher: ContentCipher, tenant_a: UUID
) -> None:
    conv_id, job_id = await _seed(admin_pool, app_pool, cipher, tenant_a, _daily(_plain(6)), 5)
    edited, nonce = cipher.encrypt("Day 2: I changed my mind about the hotel entirely.")
    await admin_pool.execute(
        "UPDATE messages SET content_encrypted = $2, content_nonce = $3"
        " WHERE conversation_id = $1 AND position = 2",
        conv_id,
        edited,
        nonce,
    )
    await admin_pool.execute(
        "UPDATE conversations SET extracted_content_hash = 'stale', extraction_version = $2"
        " WHERE id = $1",
        conv_id,
        EXTRACTION_VERSION,
    )
    provider = ScriptedProvider([])

    with structlog.testing.capture_logs() as logs:
        result = await _run(app_pool, cipher, provider, tenant_a, conv_id, job_id)

    assert result["skipped"] is True
    assert provider.user_contents == []
    assert [log["reason"] for log in logs if log["event"] == "extraction_skipped"] == ["unchanged"]
    assert await _job_status(admin_pool, job_id) == ("completed", None)
    assert await _watermark(admin_pool, conv_id) == 5


async def test_new_messages_the_role_filter_drops_complete_unchanged_and_advance(
    app_pool: asyncpg.Pool, admin_pool: asyncpg.Pool, cipher: ContentCipher, tenant_a: UUID
) -> None:
    roles = ["user", "assistant", "user", "assistant", "system", "tool"]
    messages = _daily(_plain(6), roles)
    conv_id, job_id = await _seed(admin_pool, app_pool, cipher, tenant_a, messages, 3)
    provider = ScriptedProvider([])

    with structlog.testing.capture_logs() as logs:
        await _run(app_pool, cipher, provider, tenant_a, conv_id, job_id)

    assert provider.user_contents == []
    assert [log["reason"] for log in logs if log["event"] == "extraction_skipped"] == ["unchanged"]
    assert await _job_status(admin_pool, job_id) == ("completed", None)
    assert await _watermark(admin_pool, conv_id) == 5
