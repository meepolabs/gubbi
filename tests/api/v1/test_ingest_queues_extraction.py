"""Integration tests: ingest records queued extraction jobs.

Verifies that POST /api/v1/ingest/conversations:
  - Creates one pending extraction_jobs row per SAVED conversation, carrying
    its lane, run_after, activity_at and filtered token count
  - Never enqueues an Arq job (the scheduler dispatches) and never refuses work
  - Assigns lanes from the user's uploads recorded before the request
  - Reports budget_exhausted when the user's monthly allowance is used up
  - Skipped-by-dedup conversations produce no row
  - An audit row is written for each created job
  - Replay (re-POST same payload) does NOT duplicate in-flight rows (idempotency)

Run:
    pytest tests/api/v1/test_ingest_queues_extraction.py -v
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import asyncpg
import pytest
import pytest_asyncio
import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from httpx import ASGITransport, AsyncClient

from gubbi.api.v1.ingest import router as ingest_router
from gubbi.app_context import AppContext
from gubbi.auth.strategies import TrustGatewayStrategy
from gubbi.config import Settings, get_settings
from gubbi.crypto.cipher import ContentCipher
from gubbi.extraction.intake import TOKEN_COUNT_VERSION, count_filtered_tokens, next_nightly_window
from gubbi.models.conversation import Message
from gubbi.storage.embedding_service import EmbeddingService
from gubbi.storage.repositories.extraction_jobs import ExtractionJobAlreadyInFlight
from gubbi.storage.repositories.search import fts_search

pytestmark = pytest.mark.asyncio(loop_scope="session")

API_PREFIX = "/api/v1"
ENDPOINT = f"{API_PREFIX}/ingest/conversations"

TEST_USER_ID = UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")


NIGHTLY_HOUR = 3
_OLD_ACTIVITY = datetime(2020, 1, 1, tzinfo=UTC)
_PAST_ONBOARDING = timedelta(days=8)


def _build_conv(
    platform_id: str,
    platform: str = "chatgpt",
    title: str = "Test conversation",
    msg_count: int = 2,
    message_times: list[datetime] | None = None,
) -> dict[str, Any]:
    now = datetime.now(UTC)
    messages: list[dict[str, Any]] = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"Message {i}"}
        for i in range(msg_count)
    ]
    for message, stamp in zip(messages, message_times or [], strict=False):
        message["timestamp"] = stamp.isoformat()
    return {
        "platform": platform,
        "platform_id": platform_id,
        "title": title,
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
        "messages": messages,
    }


def _headers() -> dict[str, str]:
    return {"X-Auth-User-Id": str(TEST_USER_ID)}


async def _post(client: AsyncClient, source: str, *convs: dict[str, Any]) -> dict[str, Any]:
    resp = await client.post(
        ENDPOINT, json={"source": source, "conversations": list(convs)}, headers=_headers()
    )
    assert resp.status_code == 200, resp.text
    body: dict[str, Any] = resp.json()
    return body


async def _job_for(pool: asyncpg.Pool, platform_id: str) -> asyncpg.Record:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT ej.*
            FROM extraction_jobs ej
            JOIN conversations c ON c.id = ej.conversation_id
            WHERE c.user_id = $1 AND c.platform_id = $2
            """,
            TEST_USER_ID,
            platform_id,
        )
    assert row is not None, f"no extraction job for {platform_id}"
    return row


async def _backdate_jobs(pool: asyncpg.Pool, age: timedelta) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE extraction_jobs SET created_at = created_at - $2::interval WHERE user_id = $1",
            TEST_USER_ID,
            age,
        )


def _period_start() -> date:
    return datetime.now(UTC).date().replace(day=1)


async def _admit_all_jobs(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE extraction_jobs
            SET admitted_at = now(), budget_period_start = $2, lane = 'live'
            WHERE user_id = $1
            """,
            TEST_USER_ID,
            _period_start(),
        )


@pytest.fixture
def budget_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Enable budgets with a one-conversation monthly allowance."""
    monkeypatch.setenv("JOURNAL_LLM_BUDGET_ENABLED", "true")
    monkeypatch.setenv("JOURNAL_EXTRACTION_GLOBAL_MONTHLY_TOKEN_CEILING", "1000000000")
    monkeypatch.setenv("JOURNAL_EXTRACTION_MONTHLY_CONVERSATIONS", "1")
    get_settings.cache_clear()
    return monkeypatch


@pytest_asyncio.fixture
async def mock_arq_pool() -> AsyncMock:
    """An Arq pool mock; ingest must never call it."""
    pool = AsyncMock()
    pool.enqueue_job = AsyncMock(return_value=None)
    return pool


@pytest_asyncio.fixture
async def test_user(
    pool: asyncpg.Pool,
    clean_rls_db: asyncpg.Pool,
) -> AsyncGenerator[UUID, None]:
    """Ensure TEST_USER_ID exists in users table; tear down after test."""
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO users (id, email, timezone, created_at, updated_at)
            VALUES ($1, 'ingest-enqueue-test@test.local', 'UTC', now(), now())
            ON CONFLICT (id) DO NOTHING
            """,
            TEST_USER_ID,
        )
    yield TEST_USER_ID
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM extraction_jobs WHERE user_id = $1", TEST_USER_ID)
        await conn.execute("DELETE FROM entries WHERE user_id = $1", TEST_USER_ID)
        await conn.execute("DELETE FROM conversations WHERE user_id = $1", TEST_USER_ID)
        await conn.execute("DELETE FROM topics WHERE user_id = $1", TEST_USER_ID)
        # audit_log is append-only at the DB layer (immutability trigger);
        # clean_rls_db's per-test TRUNCATE handles audit_log cleanup via
        # journal_admin (BYPASSRLS) -- see tests/conftest.py:463.
        await conn.execute("DELETE FROM users WHERE id = $1", TEST_USER_ID)


@pytest_asyncio.fixture
async def app_with_arq(
    pool: asyncpg.Pool,
    tmp_path: Path,
    mock_arq_pool: AsyncMock,
    test_user: UUID,
) -> FastAPI:
    """Minimal FastAPI with an AppContext backed by the test pool and mock arq."""
    settings = Settings(
        db={"app_url": ""},
        auth={
            "api_key": "test-api-key-for-unit-tests-only",
            "operator_email": "ingest-enqueue-test@test.local",
            "trust_gateway": True,
        },
        server={"url": "http://localhost:8100"},
        data_dir=str(tmp_path),
    )
    cipher = ContentCipher({1: bytes([1]) * 32})
    app_ctx = AppContext(
        pool=pool,
        embedding_service=EmbeddingService(),
        settings=settings,
        logger=structlog.get_logger("test"),
        admin_pool=None,
        operator_user_id=TEST_USER_ID,
        cipher=cipher,
        arq_pool=mock_arq_pool,
    )
    app = FastAPI()
    app.state.app_ctx = app_ctx
    app.state.arq_pool = mock_arq_pool
    app.state.auth_strategies = [
        TrustGatewayStrategy(gateway_secret=None, gateway_require_signature=False),
    ]
    app.include_router(ingest_router, prefix=API_PREFIX)

    @app.exception_handler(Exception)
    async def _handler(request: Request, exc: Exception) -> JSONResponse:
        raise exc

    return app


@pytest_asyncio.fixture
async def client(app_with_arq: FastAPI) -> AsyncGenerator[AsyncClient, None]:
    transport = ASGITransport(app=app_with_arq)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class TestIngestQueuesExtraction:
    """Ingest saves conversations and records queued extraction jobs."""

    async def test_saved_conversation_creates_pending_row(
        self,
        client: AsyncClient,
        pool: asyncpg.Pool,
    ) -> None:
        """Each saved conversation produces one extraction_jobs row with status='pending'."""
        data = await _post(
            client,
            "extension_chatgpt",
            _build_conv("enq-001", title="Enqueue test 1"),
            _build_conv("enq-002", title="Enqueue test 2"),
        )

        assert data == {
            "conversations_saved": 2,
            "conversations_skipped_dedupe": 0,
            "extractions_enqueued": 2,
            "extractions_skipped_budget": 0,
            "extractions_skipped_error": 0,
            "budget_exhausted": False,
        }
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT status, admitted_at FROM extraction_jobs WHERE user_id = $1",
                TEST_USER_ID,
            )
        assert [(r["status"], r["admitted_at"]) for r in rows] == [("pending", None)] * 2

    async def test_ingest_never_enqueues_to_arq(
        self,
        client: AsyncClient,
        mock_arq_pool: AsyncMock,
    ) -> None:
        """Dispatch belongs to the scheduler: ingest makes no call on the Arq pool."""
        data = await _post(client, "paste_memories", _build_conv("no-arq-001", title="No arq"))

        assert data["extractions_enqueued"] == 1
        assert mock_arq_pool.mock_calls == []

    async def test_job_records_filtered_tokens_and_latest_activity(
        self,
        client: AsyncClient,
        pool: asyncpg.Pool,
    ) -> None:
        """Token count covers only kept turns; activity_at is the latest message time."""
        early = datetime(2026, 3, 1, 9, tzinfo=UTC)
        late = datetime(2026, 3, 1, 17, tzinfo=UTC)
        conv = _build_conv("tokens-001", title="Token count")
        conv["messages"] = [
            {"role": "system", "content": "x" * 400, "timestamp": late.isoformat()},
            {"role": "user", "content": "hello there", "timestamp": early.isoformat()},
            {"role": "assistant", "content": "a" * 37, "timestamp": late.isoformat()},
        ]

        await _post(client, "extension_chatgpt", conv)

        job = await _job_for(pool, "tokens-001")
        kept = [
            Message(role="user", content="hello there"),
            Message(role="assistant", content="a" * 37),
        ]
        assert job["filtered_tokens"] == count_filtered_tokens(kept)
        assert job["filtered_tokens"] > 0
        assert job["token_count_version"] == TOKEN_COUNT_VERSION
        assert job["activity_at"] == late

    async def test_activity_falls_back_to_created_at_without_timestamps(
        self,
        client: AsyncClient,
        pool: asyncpg.Pool,
    ) -> None:
        conv = _build_conv("no-stamps-001", title="No stamps")

        await _post(client, "extension_chatgpt", conv)

        job = await _job_for(pool, "no-stamps-001")
        assert job["activity_at"] == datetime.fromisoformat(conv["created_at"])

    async def test_naive_message_times_are_read_as_utc(
        self,
        client: AsyncClient,
        pool: asyncpg.Pool,
    ) -> None:
        conv = _build_conv("naive-001", title="Naive stamps")
        conv["messages"][0]["timestamp"] = "2026-03-01T17:00:00"

        await _post(client, "extension_chatgpt", conv)

        job = await _job_for(pool, "naive-001")
        assert job["activity_at"] == datetime(2026, 3, 1, 17, tzinfo=UTC)

    async def test_paste_is_queued_to_run_now(
        self,
        client: AsyncClient,
        pool: asyncpg.Pool,
    ) -> None:
        await _post(client, "paste_memories", _build_conv("paste-001", title="Paste"))

        job = await _job_for(pool, "paste-001")
        assert job["lane"] == "paste"
        assert job["run_after"] <= datetime.now(UTC)

    async def test_first_ever_upload_is_wholly_backfill(
        self,
        client: AsyncClient,
        pool: asyncpg.Pool,
    ) -> None:
        """Every conversation of a first upload is history, however recent its activity."""
        recent = [datetime.now(UTC) + timedelta(hours=1)]
        await _post(
            client,
            "extension_chatgpt",
            _build_conv("first-001", title="Old thread", message_times=[_OLD_ACTIVITY]),
            _build_conv("first-002", title="Recent thread", message_times=recent),
        )

        for platform_id in ("first-001", "first-002"):
            job = await _job_for(pool, platform_id)
            assert job["lane"] == "backfill"
            assert job["run_after"] <= datetime.now(UTC)

    async def test_fresh_conversation_after_onboarding_window_waits_for_nightly_window(
        self,
        client: AsyncClient,
        pool: asyncpg.Pool,
    ) -> None:
        await _post(client, "extension_chatgpt", _build_conv("live-seed", title="Seed"))
        await _backdate_jobs(pool, _PAST_ONBOARDING)

        before = datetime.now(UTC)
        fresh = [datetime.now(UTC) + timedelta(hours=1)]
        await _post(
            client,
            "extension_chatgpt",
            _build_conv("live-001", title="Fresh", message_times=fresh),
        )
        after = datetime.now(UTC)

        job = await _job_for(pool, "live-001")
        assert job["lane"] == "live"
        assert job["run_after"] in {
            next_nightly_window(before, NIGHTLY_HOUR),
            next_nightly_window(after, NIGHTLY_HOUR),
        }
        assert job["run_after"] > after

    async def test_conversation_older_than_first_platform_upload_stays_backfill(
        self,
        client: AsyncClient,
        pool: asyncpg.Pool,
    ) -> None:
        await _post(client, "extension_chatgpt", _build_conv("old-seed", title="Seed"))
        await _backdate_jobs(pool, _PAST_ONBOARDING)

        await _post(
            client,
            "extension_chatgpt",
            _build_conv("old-001", title="Older", message_times=[_OLD_ACTIVITY]),
        )

        assert (await _job_for(pool, "old-001"))["lane"] == "backfill"

    async def test_new_platform_request_is_wholly_backfill_after_onboarding_window(
        self,
        client: AsyncClient,
        pool: asyncpg.Pool,
    ) -> None:
        """The first upload from a platform is judged on history read before the request.

        A conversation later in the same request must not see the jobs recorded
        for earlier ones as the user's first upload from that platform.
        """
        await _post(client, "extension_chatgpt", _build_conv("plat-seed", title="Seed"))
        await _backdate_jobs(pool, _PAST_ONBOARDING)

        fresh = [datetime.now(UTC) + timedelta(hours=1)]
        await _post(
            client,
            "extension_claude",
            _build_conv("plat-001", platform="claude", title="Claude A", message_times=fresh),
            _build_conv("plat-002", platform="claude", title="Claude B", message_times=fresh),
        )

        lanes = [(await _job_for(pool, pid))["lane"] for pid in ("plat-001", "plat-002")]
        assert lanes == ["backfill", "backfill"]

    async def test_over_limit_user_is_saved_searchable_queued_and_flagged(
        self,
        client: AsyncClient,
        pool: asyncpg.Pool,
        budget_env: pytest.MonkeyPatch,
    ) -> None:
        """Over the monthly allowance, work is still saved and queued, never refused."""
        await _post(client, "paste_memories", _build_conv("over-seed", title="Seed"))
        await _admit_all_jobs(pool)

        data = await _post(
            client, "paste_memories", _build_conv("over-001", title="Zanzibar itinerary")
        )

        assert data["conversations_saved"] == 1
        assert data["extractions_enqueued"] == 1
        assert data["extractions_skipped_budget"] == 0
        assert data["budget_exhausted"] is True
        job = await _job_for(pool, "over-001")
        assert (job["status"], job["admitted_at"]) == ("pending", None)
        async with pool.acquire() as conn:
            hits = await fts_search(conn, "zanzibar", None, None, None, 50)
        assert job["conversation_id"] in {h.conversation_id for h in hits}

    async def test_user_within_allowance_is_not_flagged(
        self,
        client: AsyncClient,
        budget_env: pytest.MonkeyPatch,
    ) -> None:
        data = await _post(client, "paste_memories", _build_conv("under-001", title="Under"))

        assert data["budget_exhausted"] is False

    async def test_budgets_disabled_never_flags_exhaustion(
        self,
        client: AsyncClient,
        pool: asyncpg.Pool,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("JOURNAL_EXTRACTION_MONTHLY_CONVERSATIONS", "1")
        get_settings.cache_clear()
        await _post(client, "paste_memories", _build_conv("off-seed", title="Seed"))
        await _admit_all_jobs(pool)

        data = await _post(client, "paste_memories", _build_conv("off-001", title="Off"))

        assert data["budget_exhausted"] is False

    async def test_skipped_conversation_produces_no_row(
        self,
        client: AsyncClient,
        pool: asyncpg.Pool,
    ) -> None:
        """Conversations skipped by dedup produce no extraction_jobs row."""
        conv = _build_conv("enq-dedup-001", title="Dedup enqueue test")
        await _post(client, "extension_chatgpt", conv)

        data = await _post(client, "extension_chatgpt", conv)

        assert data["conversations_skipped_dedupe"] == 1
        assert data["extractions_enqueued"] == 0
        async with pool.acquire() as conn:
            count = await conn.fetchval(
                "SELECT COUNT(*) FROM extraction_jobs WHERE user_id = $1", TEST_USER_ID
            )
        assert count == 1

    async def test_audit_row_written_for_each_created_job(
        self,
        client: AsyncClient,
        pool: asyncpg.Pool,
    ) -> None:
        """An audit_log row with action='extraction_job.created' is written per job."""
        await _post(client, "extension_chatgpt", _build_conv("enq-audit-001", title="Audit test"))

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT action, target_kind
                FROM audit_log
                WHERE actor_id = $1 AND action = 'extraction_job.created'
                """,
                str(TEST_USER_ID),
            )
        assert len(rows) >= 1
        assert rows[0]["target_kind"] == "extraction_job"

    async def test_resave_with_pending_job_reuses_job(
        self,
        client: AsyncClient,
        pool: asyncpg.Pool,
    ) -> None:
        """A re-saved conversation whose job is still pending keeps that one job.

        The second upload carries a new platform_id, so the platform-level dedupe
        lets it through; the same title upserts the same conversation row, and
        the job INSERT conflicts with the pending job inside the ingest transaction.
        """
        await _post(
            client, "extension_chatgpt", _build_conv("enq-resave-001", title="Resave pending job")
        )
        async with pool.acquire() as conn:
            first_ids = await conn.fetch(
                "SELECT id FROM extraction_jobs WHERE user_id = $1", TEST_USER_ID
            )

        data = await _post(
            client, "extension_chatgpt", _build_conv("enq-resave-002", title="Resave pending job")
        )

        assert data["conversations_saved"] == 1
        assert data["extractions_skipped_error"] == 0
        assert data["extractions_enqueued"] == 1
        async with pool.acquire() as conn:
            job_ids = await conn.fetch(
                "SELECT id FROM extraction_jobs WHERE user_id = $1", TEST_USER_ID
            )
        assert len(first_ids) == 1
        assert job_ids == first_ids

    async def test_job_insert_failure_keeps_conversation_and_counts_error(
        self,
        client: AsyncClient,
        pool: asyncpg.Pool,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """TXN 2 failure is counted, not propagated: HTTP 200 and the conversation persists."""

        async def _boom(*args: Any, **kwargs: Any) -> Any:
            raise asyncpg.PostgresError("simulated insert failure")

        monkeypatch.setattr("gubbi.api.v1.ingest.extraction_jobs.create_pending", _boom)

        data = await _post(client, "extension_chatgpt", _build_conv("txn2-001", title="Txn2"))

        assert data["conversations_saved"] == 1
        assert data["extractions_enqueued"] == 0
        assert data["extractions_skipped_error"] == 1
        async with pool.acquire() as conn:
            conv_count = await conn.fetchval(
                "SELECT COUNT(*) FROM conversations WHERE user_id = $1 AND platform_id = 'txn2-001'",
                TEST_USER_ID,
            )
            job_count = await conn.fetchval(
                "SELECT COUNT(*) FROM extraction_jobs WHERE user_id = $1", TEST_USER_ID
            )
        assert (conv_count, job_count) == (1, 0)

    async def test_already_in_flight_job_counts_as_queued(
        self,
        client: AsyncClient,
        pool: asyncpg.Pool,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An in-flight job for the conversation is reused: queued, not an error."""
        existing_job_id = uuid4()

        async def _already_in_flight(*args: object, **kwargs: object) -> UUID:
            raise ExtractionJobAlreadyInFlight(existing_job_id)

        monkeypatch.setattr(
            "gubbi.api.v1.ingest.extraction_jobs.create_pending", _already_in_flight
        )

        data = await _post(
            client, "extension_chatgpt", _build_conv("inflight-001", title="In flight")
        )

        assert data["conversations_saved"] == 1
        assert data["extractions_enqueued"] == 1
        assert data["extractions_skipped_error"] == 0
        async with pool.acquire() as conn:
            audited = await conn.fetchval(
                "SELECT COUNT(*) FROM audit_log WHERE target_id = $1",
                str(existing_job_id),
            )
        assert audited == 1
