"""Integration tests: ingest endpoint enqueues extraction jobs.

Verifies that POST /api/v1/ingest/conversations:
  - Creates one extraction_jobs row per SAVED conversation (status='pending')
  - Calls arq_pool.enqueue_job with the matching _job_id
  - Skipped-by-dedup conversations produce neither a row nor an enqueue call
  - An audit row is written for each created job
  - Replay (re-POST same payload) does NOT duplicate in-flight rows (idempotency)
  - Pre-charge denial: conversation saves but no extraction row / no enqueue (D1, D2, D3)
  - Mid-batch exhaustion: correct per-conversation counters
  - Pre-charge succeeds + savepoint 2 fails: refund called with actual=0 (D4)
  - helper=None (self-host): all conversations enqueue unconditionally

The arq pool is mocked -- no Redis required.

Run:
    pytest tests/api/v1/test_ingest_enqueues_extraction.py -v
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import asyncpg
import pytest
import pytest_asyncio
import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from httpx import ASGITransport, AsyncClient

from gubbi.api.v1.ingest import router as ingest_router
from gubbi.app_context import AppContext
from gubbi.config import Settings
from gubbi.crypto.cipher import ContentCipher
from gubbi.storage.embedding_service import EmbeddingService
from gubbi.storage.repositories.extraction_jobs import ExtractionJobAlreadyInFlight

pytestmark = pytest.mark.asyncio(loop_scope="session")

API_PREFIX = "/api/v1"
ENDPOINT = f"{API_PREFIX}/ingest/conversations"

TEST_USER_ID = UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")


def _build_conv(
    platform_id: str,
    platform: str = "chatgpt",
    title: str = "Test conversation",
    msg_count: int = 2,
) -> dict[str, Any]:
    now = datetime.now(UTC)
    return {
        "platform": platform,
        "platform_id": platform_id,
        "title": title,
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
        "messages": [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"Message {i}"}
            for i in range(msg_count)
        ],
    }


@pytest_asyncio.fixture
async def mock_arq_pool() -> AsyncMock:
    """A minimal arq pool mock with an awaitable enqueue_job."""
    pool = AsyncMock()
    pool.enqueue_job = AsyncMock(return_value=None)
    return pool


@pytest_asyncio.fixture
async def test_user(pool: asyncpg.Pool) -> UUID:
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
        await conn.execute("DELETE FROM conversations WHERE user_id = $1", TEST_USER_ID)
        await conn.execute("DELETE FROM topics WHERE user_id = $1", TEST_USER_ID)
        await conn.execute("DELETE FROM audit_log WHERE actor_id = $1", str(TEST_USER_ID))
        await conn.execute("DELETE FROM users WHERE id = $1", TEST_USER_ID)


@pytest_asyncio.fixture
async def app_with_arq(
    pool: asyncpg.Pool,
    clean_pool: asyncpg.Pool,  # noqa: ARG001 -- ensures clean tables
    tmp_path: Path,
    mock_arq_pool: AsyncMock,
    test_user: UUID,  # noqa: ARG001 -- ensures user exists
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
        budget_helper=None,  # default: self-host semantics; tests may override
    )
    app = FastAPI()
    app.state.app_ctx = app_ctx
    app.state.arq_pool = mock_arq_pool
    app.include_router(ingest_router, prefix=API_PREFIX)

    @app.exception_handler(Exception)
    async def _handler(request: Request, exc: Exception) -> JSONResponse:
        raise exc

    return app


@pytest_asyncio.fixture
async def client(app_with_arq: FastAPI) -> AsyncClient:
    transport = ASGITransport(app=app_with_arq)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class TestIngestEnqueuesExtraction:
    """Assertions that ingest wires up extraction job rows and arq enqueues."""

    async def test_saved_conversation_creates_pending_row(
        self,
        client: AsyncClient,
        pool: asyncpg.Pool,
    ) -> None:
        """Each saved conversation produces one extraction_jobs row with status='pending'."""
        payload = {
            "source": "extension_chatgpt",
            "conversations": [
                _build_conv("enq-001", title="Enqueue test 1"),
                _build_conv("enq-002", title="Enqueue test 2"),
            ],
        }
        resp = await client.post(
            ENDPOINT,
            json=payload,
            headers={"X-Auth-User-Id": str(TEST_USER_ID)},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["conversations_saved"] == 2
        assert data["conversations_skipped_dedupe"] == 0
        assert data["extractions_enqueued"] == 2
        assert data["extractions_skipped_budget"] == 0
        assert data["extractions_skipped_error"] == 0
        assert data["budget_exhausted"] is False

        # Verify two pending extraction_jobs rows were created.
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT status FROM extraction_jobs WHERE user_id = $1 ORDER BY created_at",
                TEST_USER_ID,
            )
        assert len(rows) == 2  # noqa: PLR2004
        assert all(r["status"] == "pending" for r in rows)

    async def test_enqueue_job_called_per_saved_conversation(
        self,
        client: AsyncClient,
        pool: asyncpg.Pool,
        app_with_arq: FastAPI,
    ) -> None:
        """arq_pool.enqueue_job is called once per saved conversation with matching _job_id."""
        mock_arq: AsyncMock = app_with_arq.state.arq_pool
        mock_arq.enqueue_job.reset_mock()

        payload = {
            "source": "extension_claude",
            "conversations": [
                _build_conv("enq-arq-001", platform="claude", title="Arq test"),
            ],
        }
        resp = await client.post(
            ENDPOINT,
            json=payload,
            headers={"X-Auth-User-Id": str(TEST_USER_ID)},
        )
        assert resp.status_code == 200, resp.text

        assert mock_arq.enqueue_job.call_count == 1
        call_args = mock_arq.enqueue_job.call_args
        assert call_args[0][0] == "extract_conversation"
        # ``_job_id=`` is the arq-internal tracking id (queue + result
        # key, used for dedup); it is NOT forwarded to the worker
        # function. The worker function reads ``job_id`` only from its
        # 4th positional argument. A regression that drops the 4th arg
        # makes the worker run with the default sentinel ``job_id="unknown"``
        # and silently skips ``mark_running`` / ``mark_completed`` /
        # ``mark_failed``, leaving ``extraction_jobs.status`` stuck at
        # ``'pending'``. Lock both slots independently so future drift
        # in either path surfaces here.
        assert len(call_args[0]) == 4, (  # noqa: PLR2004
            f"expected 4 positional args (function_name, conversation_id, "
            f"user_id, job_id); got {len(call_args[0])}: {call_args[0]!r}"
        )
        positional_job_id = call_args[0][3]
        kwarg_job_id = call_args[1].get("_job_id")
        assert kwarg_job_id is not None
        UUID(kwarg_job_id)  # raises if not a valid UUID
        UUID(positional_job_id)  # raises if not a valid UUID
        assert positional_job_id == kwarg_job_id, (
            "4th positional ``job_id`` must equal ``_job_id`` kwarg so "
            "the FSM row updated by the worker is the same row arq is "
            "tracking for dedup + result lookup"
        )

        # Verify it matches the DB row.
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id FROM extraction_jobs WHERE user_id = $1",
                TEST_USER_ID,
            )
        assert row is not None
        assert str(row["id"]) == kwarg_job_id

    async def test_skipped_conversation_produces_no_row_no_enqueue(
        self,
        client: AsyncClient,
        pool: asyncpg.Pool,
        app_with_arq: FastAPI,
    ) -> None:
        """Conversations skipped by dedup produce no extraction_jobs row and no enqueue call."""
        mock_arq: AsyncMock = app_with_arq.state.arq_pool
        mock_arq.enqueue_job.reset_mock()

        conv = _build_conv("enq-dedup-001", title="Dedup enqueue test")
        payload = {"source": "extension_chatgpt", "conversations": [conv]}

        # First POST -- saves and enqueues.
        resp1 = await client.post(
            ENDPOINT,
            json=payload,
            headers={"X-Auth-User-Id": str(TEST_USER_ID)},
        )
        assert resp1.status_code == 200

        # Second POST -- dedupe skips; no additional row or enqueue.
        mock_arq.enqueue_job.reset_mock()
        resp2 = await client.post(
            ENDPOINT,
            json=payload,
            headers={"X-Auth-User-Id": str(TEST_USER_ID)},
        )
        assert resp2.status_code == 200
        assert resp2.json()["conversations_skipped_dedupe"] == 1
        assert mock_arq.enqueue_job.call_count == 0

    async def test_audit_row_written_for_each_created_job(
        self,
        client: AsyncClient,
        pool: asyncpg.Pool,
    ) -> None:
        """An audit_log row with action='extraction_job.created' is written per job."""
        payload = {
            "source": "extension_chatgpt",
            "conversations": [
                _build_conv("enq-audit-001", title="Audit test"),
            ],
        }
        resp = await client.post(
            ENDPOINT,
            json=payload,
            headers={"X-Auth-User-Id": str(TEST_USER_ID)},
        )
        assert resp.status_code == 200

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

    async def test_replay_does_not_duplicate_in_flight_rows(
        self,
        client: AsyncClient,
        pool: asyncpg.Pool,
    ) -> None:
        """Replaying an ingest while a job is in-flight does not create duplicate rows.

        The partial unique index on (user_id, conversation_id, source) WHERE
        status NOT IN ('completed','failed') enforces one in-flight row.
        The ExtractionJobAlreadyInFlight path in ingest re-uses the existing job_id
        rather than failing.
        """
        conv = _build_conv("enq-replay-001", title="Replay test")
        payload = {"source": "extension_chatgpt", "conversations": [conv]}

        # First POST -- creates conversation + pending job.
        resp1 = await client.post(
            ENDPOINT,
            json=payload,
            headers={"X-Auth-User-Id": str(TEST_USER_ID)},
        )
        assert resp1.status_code == 200
        assert resp1.json()["conversations_saved"] == 1

        # Simulate a "new" upload with the same conversation (different platform_id to
        # bypass the conv-level dedup, but same combination of user/conv/source).
        # This tests the extraction_jobs partial-unique path directly by calling
        # the repository function.
        async with pool.acquire() as conn:
            count_before = await conn.fetchval(
                "SELECT COUNT(*) FROM extraction_jobs WHERE user_id = $1",
                TEST_USER_ID,
            )

        # Second POST with same payload -- conv dedup fires, so extraction row stays 1.
        resp2 = await client.post(
            ENDPOINT,
            json=payload,
            headers={"X-Auth-User-Id": str(TEST_USER_ID)},
        )
        assert resp2.status_code == 200

        async with pool.acquire() as conn:
            count_after = await conn.fetchval(
                "SELECT COUNT(*) FROM extraction_jobs WHERE user_id = $1",
                TEST_USER_ID,
            )

        assert count_after == count_before

    async def test_pre_charge_denies_save_commits_no_enqueue(
        self,
        client: AsyncClient,
        pool: asyncpg.Pool,
        app_with_arq: FastAPI,
    ) -> None:
        """Pre-charge returns False -- conversation saves; no extraction row; no enqueue.

        Asserts D1 + D2 + D3: save commits unconditionally, response carries
        extractions_skipped_budget=1 and budget_exhausted=True, HTTP 200.
        """
        # Arrange: install a mock helper that denies pre_charge.
        helper_mock = MagicMock()
        helper_mock.pre_charge = AsyncMock(return_value=False)
        helper_mock.record_actual_cost = AsyncMock()
        app_with_arq.state.app_ctx.budget_helper = helper_mock

        mock_arq: AsyncMock = app_with_arq.state.arq_pool
        mock_arq.enqueue_job.reset_mock()

        payload = {
            "source": "extension_chatgpt",
            "conversations": [_build_conv("denied-001", title="Denied")],
        }

        # Act
        resp = await client.post(
            ENDPOINT,
            json=payload,
            headers={"X-Auth-User-Id": str(TEST_USER_ID)},
        )

        # Assert
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["conversations_saved"] == 1
        assert data["extractions_enqueued"] == 0
        assert data["extractions_skipped_budget"] == 1
        assert data["budget_exhausted"] is True

        # No extraction_jobs row (scoped by platform_id to avoid cross-test pollution).
        async with pool.acquire() as conn:
            count = await conn.fetchval(
                """
                SELECT COUNT(*)
                FROM extraction_jobs ej
                JOIN conversations c ON c.id = ej.conversation_id
                WHERE c.user_id = $1
                  AND c.platform = 'chatgpt'
                  AND c.platform_id = 'denied-001'
                """,
                TEST_USER_ID,
            )
        assert count == 0

        # No arq enqueue.
        assert mock_arq.enqueue_job.call_count == 0

        # No refund (pre_charge was False, nothing to refund).
        helper_mock.record_actual_cost.assert_not_called()

        # Cleanup: reset helper so later tests in same class are unaffected.
        app_with_arq.state.app_ctx.budget_helper = None

    async def test_mid_batch_budget_exhaustion(
        self,
        client: AsyncClient,
        pool: asyncpg.Pool,
        app_with_arq: FastAPI,
    ) -> None:
        """3 conversations; pre_charge True/True/False -- exact counter values."""
        helper_mock = MagicMock()
        helper_mock.pre_charge = AsyncMock(side_effect=[True, True, False])
        helper_mock.record_actual_cost = AsyncMock()
        app_with_arq.state.app_ctx.budget_helper = helper_mock

        mock_arq: AsyncMock = app_with_arq.state.arq_pool
        mock_arq.enqueue_job.reset_mock()

        payload = {
            "source": "extension_chatgpt",
            "conversations": [
                _build_conv("mid-001", title="A"),
                _build_conv("mid-002", title="B"),
                _build_conv("mid-003", title="C"),
            ],
        }

        resp = await client.post(
            ENDPOINT,
            json=payload,
            headers={"X-Auth-User-Id": str(TEST_USER_ID)},
        )
        assert resp.status_code == 200, resp.text

        data = resp.json()
        assert data["conversations_saved"] == 3
        assert data["conversations_skipped_dedupe"] == 0
        assert data["extractions_enqueued"] == 2
        assert data["extractions_skipped_budget"] == 1
        assert data["budget_exhausted"] is True

        assert mock_arq.enqueue_job.call_count == 2

        # Cleanup.
        app_with_arq.state.app_ctx.budget_helper = None

    async def test_pre_charge_succeeds_inner_savepoint_fails_refund_called(
        self,
        client: AsyncClient,
        pool: asyncpg.Pool,
        app_with_arq: FastAPI,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Pre-charge True -> TXN 2 INSERT raises -> refund + continue (200).

        Asserts B3-H1 + D4:
        - HTTP 200 (not 500; TXN 2 failure is counted, not propagated).
        - extractions_skipped_error == 1.
        - conversations_saved == 1 (TXN 1 committed independently).
        - Refund is called with actual_cents=0, estimated_cents=PRE_CHARGE_CENTS.
        """
        from gubbi_common.budget import PRE_CHARGE_CENTS  # noqa: PLC0415

        helper_mock = MagicMock()
        helper_mock.pre_charge = AsyncMock(return_value=True)
        helper_mock.record_actual_cost = AsyncMock()
        app_with_arq.state.app_ctx.budget_helper = helper_mock

        # Force extraction_jobs.create_pending to raise an unexpected error
        # (NOT ExtractionJobAlreadyInFlight, which is caught and reused).
        async def _boom(*args: Any, **kwargs: Any) -> Any:
            raise asyncpg.PostgresError("simulated insert failure")

        monkeypatch.setattr(
            "gubbi.api.v1.ingest.extraction_jobs.create_pending",
            _boom,
        )

        payload = {
            "source": "extension_chatgpt",
            "conversations": [_build_conv("refund-001", title="Refund")],
        }

        resp = await client.post(
            ENDPOINT,
            json=payload,
            headers={"X-Auth-User-Id": str(TEST_USER_ID)},
        )

        # B3-H1: TXN 2 failure continues; HTTP 200 (not 500).
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["conversations_saved"] == 1
        assert data["extractions_enqueued"] == 0
        assert data["extractions_skipped_error"] == 1
        assert data["extractions_skipped_budget"] == 0

        # Conversation row IS committed (TXN 1 succeeded independently).
        # Scoped to platform_id to avoid interference from other tests (B3-H3).
        async with pool.acquire() as conn:
            conv_count = await conn.fetchval(
                """
                SELECT COUNT(*) FROM conversations
                WHERE user_id = $1
                  AND platform = 'chatgpt'
                  AND platform_id = 'refund-001'
                """,
                TEST_USER_ID,
            )
            job_count = await conn.fetchval(
                """
                SELECT COUNT(*)
                FROM extraction_jobs ej
                JOIN conversations c ON c.id = ej.conversation_id
                WHERE c.user_id = $1
                  AND c.platform = 'chatgpt'
                  AND c.platform_id = 'refund-001'
                """,
                TEST_USER_ID,
            )
        assert conv_count == 1
        assert job_count == 0  # TXN 2 rolled back

        # Refund called with actual=0, estimated=50.
        helper_mock.record_actual_cost.assert_called_once()
        kwargs = helper_mock.record_actual_cost.call_args.kwargs
        assert kwargs["actual_cents"] == 0
        assert kwargs["estimated_cents"] == PRE_CHARGE_CENTS

        # Cleanup.
        app_with_arq.state.app_ctx.budget_helper = None

    async def test_helper_is_none_self_host_unconditional_enqueue(
        self,
        client: AsyncClient,
        pool: asyncpg.Pool,
        app_with_arq: FastAPI,
    ) -> None:
        """helper=None (self-host / budget disabled) -- no pre_charge, all enqueue."""
        # Default fixture has budget_helper=None already.
        assert app_with_arq.state.app_ctx.budget_helper is None

        mock_arq: AsyncMock = app_with_arq.state.arq_pool
        mock_arq.enqueue_job.reset_mock()

        payload = {
            "source": "extension_chatgpt",
            "conversations": [
                _build_conv("self-001", title="A"),
                _build_conv("self-002", title="B"),
            ],
        }

        resp = await client.post(
            ENDPOINT,
            json=payload,
            headers={"X-Auth-User-Id": str(TEST_USER_ID)},
        )
        assert resp.status_code == 200

        data = resp.json()
        assert data["conversations_saved"] == 2
        assert data["extractions_enqueued"] == 2
        assert data["extractions_skipped_budget"] == 0
        assert data["extractions_skipped_error"] == 0
        assert data["budget_exhausted"] is False
        assert mock_arq.enqueue_job.call_count == 2

    async def test_conv_level_dedupe_does_not_call_pre_charge(
        self,
        client: AsyncClient,
        pool: asyncpg.Pool,
        app_with_arq: FastAPI,
    ) -> None:
        """Re-POST of the same (platform, platform_id) is caught by conv-level dedupe.

        exists_by_platform_id fires BEFORE TXN 1, so pre_charge is never called
        on the second POST and the existing extraction row is untouched.
        """
        # Arrange
        helper_mock = MagicMock()
        helper_mock.pre_charge = AsyncMock(return_value=True)
        helper_mock.record_actual_cost = AsyncMock()
        app_with_arq.state.app_ctx.budget_helper = helper_mock

        mock_arq: AsyncMock = app_with_arq.state.arq_pool
        mock_arq.enqueue_job.reset_mock()

        conv = _build_conv("conv-dedupe-budget-001", title="Conv-level dedupe test")
        payload = {"source": "extension_chatgpt", "conversations": [conv]}

        # Act: first POST saves + pre-charges + enqueues
        resp1 = await client.post(
            ENDPOINT,
            json=payload,
            headers={"X-Auth-User-Id": str(TEST_USER_ID)},
        )
        assert resp1.status_code == 200
        data1 = resp1.json()
        assert data1["conversations_saved"] == 1
        assert data1["extractions_enqueued"] == 1
        assert data1["extractions_skipped_budget"] == 0
        assert helper_mock.pre_charge.call_count == 1

        mock_arq.enqueue_job.reset_mock()
        helper_mock.pre_charge.reset_mock()

        # Act: second POST with identical payload -- conv-level dedupe fires
        resp2 = await client.post(
            ENDPOINT,
            json=payload,
            headers={"X-Auth-User-Id": str(TEST_USER_ID)},
        )
        assert resp2.status_code == 200
        data2 = resp2.json()

        # Assert: dedupe counter increments, no new pre-charge, no new enqueue
        assert data2["conversations_saved"] == 0
        assert data2["conversations_skipped_dedupe"] == 1
        assert data2["extractions_enqueued"] == 0
        assert data2["extractions_skipped_budget"] == 0
        assert data2["extractions_skipped_error"] == 0
        helper_mock.pre_charge.assert_not_called()
        mock_arq.enqueue_job.assert_not_called()

        # Exactly one extraction_jobs row for this conversation
        async with pool.acquire() as conn:
            job_count = await conn.fetchval(
                """
                SELECT COUNT(*)
                FROM extraction_jobs ej
                JOIN conversations c ON c.id = ej.conversation_id
                WHERE c.user_id = $1
                  AND c.platform = 'chatgpt'
                  AND c.platform_id = 'conv-dedupe-budget-001'
                """,
                TEST_USER_ID,
            )
        assert job_count == 1

        # Cleanup
        app_with_arq.state.app_ctx.budget_helper = None

    async def test_already_in_flight_budget_enabled_does_not_refund(
        self,
        client: AsyncClient,
        app_with_arq: FastAPI,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When ExtractionJobAlreadyInFlight fires inside TXN 2, audit runs but no refund.

        The intentional 'preserve-debit-for-idempotent-re-POST' semantic: the
        existing extraction job will eventually consume the pre-charge, so
        refunding would over-credit the user.

        Uses monkeypatch to inject ExtractionJobAlreadyInFlight from create_pending
        so the integration test exercises the TXN 2 in-flight branch directly,
        bypassing the conv-level dedupe that normally prevents this branch from
        being reached in normal flow (race-condition-only code path).
        """
        # Arrange
        helper_mock = MagicMock()
        helper_mock.pre_charge = AsyncMock(return_value=True)
        helper_mock.record_actual_cost = AsyncMock()
        app_with_arq.state.app_ctx.budget_helper = helper_mock

        mock_arq: AsyncMock = app_with_arq.state.arq_pool
        mock_arq.enqueue_job.reset_mock()

        existing_job_id = UUID("eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee")

        async def _already_in_flight(*args: object, **kwargs: object) -> UUID:
            raise ExtractionJobAlreadyInFlight(existing_job_id)

        monkeypatch.setattr(
            "gubbi.api.v1.ingest.extraction_jobs.create_pending",
            _already_in_flight,
        )

        conv = _build_conv("inflight-monkeypatch-001", title="AlreadyInFlight monkeypatch test")
        payload = {"source": "extension_chatgpt", "conversations": [conv]}

        # Act
        resp = await client.post(
            ENDPOINT,
            json=payload,
            headers={"X-Auth-User-Id": str(TEST_USER_ID)},
        )

        # Assert: ingest succeeds and the existing job gets enqueued
        assert resp.status_code == 200
        data = resp.json()
        # The conversation saves (TXN 1 succeeds before create_pending is called).
        assert data["conversations_saved"] == 1
        # The existing job is treated as the enqueued job -- no error counter.
        assert data["extractions_skipped_error"] == 0
        assert data["extractions_skipped_budget"] == 0
        # pre_charge fired once (before create_pending); NO record_actual_cost refund.
        assert helper_mock.pre_charge.call_count == 1
        helper_mock.record_actual_cost.assert_not_called()
        # arq.enqueue_job called with the existing_job_id (reuse semantics).
        mock_arq.enqueue_job.assert_called_once()
        call_args = mock_arq.enqueue_job.call_args
        # Structured check matching test_enqueue_job_called_per_saved_conversation:
        # 4th positional must be the existing_job_id string, _job_id kwarg too.
        assert call_args[0][3] == str(existing_job_id), (
            f"expected positional job_id arg to be the existing in-flight UUID; "
            f"got {call_args[0][3]!r}"
        )
        assert call_args[1]["_job_id"] == str(existing_job_id), (
            f"expected _job_id kwarg to be the existing in-flight UUID; "
            f"got {call_args[1]['_job_id']!r}"
        )

        # Cleanup
        app_with_arq.state.app_ctx.budget_helper = None
