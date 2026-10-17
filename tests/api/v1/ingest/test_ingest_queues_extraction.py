"""Integration tests: ingest records queued extraction jobs.

Verifies that POST /api/v1/ingest/conversations:
  - Creates one pending extraction_jobs row per SAVED conversation, carrying
    its lane, run_after and activity_at (never later than the server's now)
  - Never enqueues an Arq job (the scheduler dispatches) and never refuses work
  - Assigns lanes from the user's uploads recorded before the request
  - A re-send that adds no message produces no row
  - An audit row is written for each created job, none for a reused one
  - A conversation and its job are stored together or not at all, each
    conversation independently of the others in its request
  - A re-upload refreshes the queue inputs of a job not yet dispatched
  - Replay (re-POST same payload) does NOT duplicate in-flight rows (idempotency)

Run:
    pytest tests/api/v1/ingest/test_ingest_queues_extraction.py -v
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import asyncpg
import pytest
from httpx import AsyncClient

from gubbi.extraction.intake import next_nightly_window
from gubbi.storage.repositories import extraction_jobs
from gubbi.storage.repositories.extraction_jobs import ExtractionJobAlreadyInFlight
from tests.fixtures.ingest_app import (
    CREATED,
    NIGHTLY_HOUR,
    OLD_ACTIVITY,
    PAST_ONBOARDING,
    TEST_USER_ID,
    UPDATED,
    admit_all_jobs,
    backdate_jobs,
    build_conv,
    counts_for,
    job_by_id,
    job_for,
    post,
)

pytestmark = pytest.mark.asyncio(loop_scope="session")


class TestIngestQueuesExtraction:
    """Ingest saves conversations and records queued extraction jobs."""

    async def test_saved_conversation_creates_pending_row(
        self,
        client: AsyncClient,
        pool: asyncpg.Pool,
    ) -> None:
        """Each saved conversation produces one extraction_jobs row with status='pending'."""
        data = await post(
            client,
            "extension_chatgpt",
            build_conv("enq-001", title="Enqueue test 1"),
            build_conv("enq-002", title="Enqueue test 2"),
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
        data = await post(client, "paste_memories", build_conv("no-arq-001", title="No arq"))

        assert data["extractions_enqueued"] == 1
        assert mock_arq_pool.mock_calls == []

    async def test_job_records_the_latest_kept_message_time_as_activity(
        self,
        client: AsyncClient,
        pool: asyncpg.Pool,
    ) -> None:
        """activity_at is the latest timestamp among the kept (non-system) turns."""
        early = datetime(2026, 3, 1, 9, tzinfo=UTC)
        late = datetime(2026, 3, 1, 17, tzinfo=UTC)
        latest_system = datetime(2026, 3, 1, 23, tzinfo=UTC)
        conv = build_conv("activity-001", title="Activity")
        conv["messages"] = [
            {"role": "system", "content": "x" * 400, "timestamp": latest_system.isoformat()},
            {"role": "user", "content": "hello there", "timestamp": early.isoformat()},
            {"role": "assistant", "content": "a" * 37, "timestamp": late.isoformat()},
        ]

        await post(client, "extension_chatgpt", conv)

        assert (await job_for(pool, "activity-001"))["activity_at"] == late

    async def test_a_future_message_time_is_clamped_to_now(
        self,
        client: AsyncClient,
        pool: asyncpg.Pool,
    ) -> None:
        """A client clock ahead of the server cannot date activity in the future."""
        future = datetime.now(UTC) + timedelta(days=30)
        before = datetime.now(UTC)

        await post(
            client,
            "extension_chatgpt",
            build_conv("future-001", title="Future", message_times=[future]),
        )

        activity = (await job_for(pool, "future-001"))["activity_at"]
        assert before <= activity <= datetime.now(UTC)

    @pytest.mark.parametrize(
        ("updated_at", "expected"),
        [
            pytest.param(UPDATED, UPDATED, id="updated_at"),
            pytest.param(None, CREATED, id="created_at"),
        ],
    )
    async def test_activity_falls_back_without_message_timestamps(
        self,
        client: AsyncClient,
        pool: asyncpg.Pool,
        updated_at: datetime | None,
        expected: datetime,
    ) -> None:
        """Without message times, activity is the update time, else the creation time."""
        conv = build_conv("no-stamps-001", title="No stamps")
        conv["created_at"] = CREATED.isoformat()
        conv["updated_at"] = updated_at.isoformat() if updated_at else None

        await post(client, "extension_chatgpt", conv)

        job = await job_for(pool, "no-stamps-001")
        assert job["activity_at"] == expected

    async def test_naive_message_times_are_read_as_utc(
        self,
        client: AsyncClient,
        pool: asyncpg.Pool,
    ) -> None:
        conv = build_conv("naive-001", title="Naive stamps")
        conv["messages"][0]["timestamp"] = "2026-03-01T17:00:00"

        await post(client, "extension_chatgpt", conv)

        job = await job_for(pool, "naive-001")
        assert job["activity_at"] == datetime(2026, 3, 1, 17, tzinfo=UTC)

    async def test_offset_message_times_are_stored_as_the_same_utc_instant(
        self,
        client: AsyncClient,
        pool: asyncpg.Pool,
    ) -> None:
        conv = build_conv("offset-001", title="Offset stamps")
        conv["messages"][0]["timestamp"] = "2026-03-01T17:00:00+05:30"

        await post(client, "extension_chatgpt", conv)

        job = await job_for(pool, "offset-001")
        assert job["activity_at"] == datetime(2026, 3, 1, 11, 30, tzinfo=UTC)

    async def test_paste_is_queued_to_run_now(
        self,
        client: AsyncClient,
        pool: asyncpg.Pool,
    ) -> None:
        await post(client, "paste_memories", build_conv("paste-001", title="Paste"))

        job = await job_for(pool, "paste-001")
        assert job["lane"] == "paste"
        assert job["run_after"] <= datetime.now(UTC)

    async def test_first_ever_upload_is_wholly_backfill(
        self,
        client: AsyncClient,
        pool: asyncpg.Pool,
    ) -> None:
        """Every conversation of a first upload is history, however recent its activity."""
        recent = [datetime.now(UTC) + timedelta(hours=1)]
        await post(
            client,
            "extension_chatgpt",
            build_conv("first-001", title="Old thread", message_times=[OLD_ACTIVITY]),
            build_conv("first-002", title="Recent thread", message_times=recent),
        )

        for platform_id in ("first-001", "first-002"):
            job = await job_for(pool, platform_id)
            assert job["lane"] == "backfill"
            assert job["run_after"] <= datetime.now(UTC)

    async def test_fresh_conversation_after_onboarding_window_waits_for_nightly_window(
        self,
        client: AsyncClient,
        pool: asyncpg.Pool,
    ) -> None:
        await post(client, "extension_chatgpt", build_conv("live-seed", title="Seed"))
        await backdate_jobs(pool, PAST_ONBOARDING)

        before = datetime.now(UTC)
        fresh = [datetime.now(UTC) + timedelta(hours=1)]
        await post(
            client,
            "extension_chatgpt",
            build_conv("live-001", title="Fresh", message_times=fresh),
        )
        after = datetime.now(UTC)

        job = await job_for(pool, "live-001")
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
        await post(client, "extension_chatgpt", build_conv("old-seed", title="Seed"))
        await backdate_jobs(pool, PAST_ONBOARDING)

        await post(
            client,
            "extension_chatgpt",
            build_conv("old-001", title="Older", message_times=[OLD_ACTIVITY]),
        )

        assert (await job_for(pool, "old-001"))["lane"] == "backfill"

    async def test_new_platform_request_is_wholly_backfill_after_onboarding_window(
        self,
        client: AsyncClient,
        pool: asyncpg.Pool,
    ) -> None:
        """The first upload from a platform is judged on history read before the request.

        A conversation later in the same request must not see the jobs recorded
        for earlier ones as the user's first upload from that platform.
        """
        await post(client, "extension_chatgpt", build_conv("plat-seed", title="Seed"))
        await backdate_jobs(pool, PAST_ONBOARDING)

        fresh = [datetime.now(UTC) + timedelta(hours=1)]
        await post(
            client,
            "extension_claude",
            build_conv("plat-001", platform="claude", title="Claude A", message_times=fresh),
            build_conv("plat-002", platform="claude", title="Claude B", message_times=fresh),
        )

        lanes = [(await job_for(pool, pid))["lane"] for pid in ("plat-001", "plat-002")]
        assert lanes == ["backfill", "backfill"]

    async def test_platforms_are_judged_on_their_own_history_across_requests(
        self,
        client: AsyncClient,
        pool: asyncpg.Pool,
    ) -> None:
        """After onboarding, a known platform's fresh work is live; a new platform's is backfill."""
        await post(client, "extension_chatgpt", build_conv("mix-seed", title="Seed"))
        await backdate_jobs(pool, PAST_ONBOARDING)
        fresh = [datetime.now(UTC) + timedelta(hours=1)]

        await post(
            client,
            "extension_chatgpt",
            build_conv("mix-001", title="ChatGPT fresh", message_times=fresh),
        )
        await post(
            client,
            "extension_claude",
            build_conv("mix-002", platform="claude", title="Claude fresh", message_times=fresh),
        )

        lanes = [(await job_for(pool, pid))["lane"] for pid in ("mix-001", "mix-002")]
        assert lanes == ["live", "backfill"]
        assert (await job_for(pool, "mix-002"))["run_after"] <= datetime.now(UTC)

    async def test_skipped_conversation_produces_no_row(
        self,
        client: AsyncClient,
        pool: asyncpg.Pool,
    ) -> None:
        """An identical re-send is skipped and produces no extraction_jobs row."""
        conv = build_conv("enq-dedup-001", title="Dedup enqueue test")
        await post(client, "extension_chatgpt", conv)

        data = await post(client, "extension_chatgpt", conv)

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
        await post(client, "extension_chatgpt", build_conv("enq-audit-001", title="Audit test"))

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
        """A thread re-sent with new messages while its job is pending keeps that one job."""
        await post(
            client, "extension_chatgpt", build_conv("enq-resave-001", title="Resave pending job")
        )
        async with pool.acquire() as conn:
            first_ids = await conn.fetch(
                "SELECT id FROM extraction_jobs WHERE user_id = $1", TEST_USER_ID
            )

        data = await post(
            client,
            "extension_chatgpt",
            build_conv("enq-resave-001", title="Resave pending job", msg_count=4),
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
        assert await counts_for(pool, "enq-resave-001") == (1, 1)

    async def test_job_insert_failure_stores_nothing_and_counts_failed_save(
        self,
        client: AsyncClient,
        pool: asyncpg.Pool,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A job insert failure rolls back the conversation; HTTP 200, one failed save."""

        async def _boom(*args: Any, **kwargs: Any) -> Any:
            raise asyncpg.PostgresError("simulated insert failure")

        monkeypatch.setattr("gubbi.api.v1.ingest.extraction_jobs.create_pending", _boom)

        data = await post(client, "extension_chatgpt", build_conv("atomic-001", title="Atomic"))

        assert (data["conversations_saved"], data["extractions_enqueued"]) == (0, 0)
        assert data["extractions_skipped_error"] == 1
        assert await counts_for(pool, "atomic-001") == (0, 0)

    async def test_failed_job_insert_rolls_back_only_its_own_conversation(
        self,
        client: AsyncClient,
        pool: asyncpg.Pool,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A failure on the second conversation leaves the first stored with its job."""
        create_pending = extraction_jobs.create_pending
        calls = 0

        async def _fail_second(*args: Any, **kwargs: Any) -> UUID:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise asyncpg.PostgresError("simulated insert failure")
            return await create_pending(*args, **kwargs)

        monkeypatch.setattr("gubbi.api.v1.ingest.extraction_jobs.create_pending", _fail_second)

        data = await post(
            client,
            "extension_chatgpt",
            build_conv("pair-001", title="Pair first"),
            build_conv("pair-002", title="Pair second"),
        )

        assert (data["conversations_saved"], data["extractions_enqueued"]) == (1, 1)
        assert data["extractions_skipped_error"] == 1
        assert await counts_for(pool, "pair-001") == (1, 1)
        assert await counts_for(pool, "pair-002") == (0, 1)
        assert (await job_for(pool, "pair-001"))["status"] == "pending"

    async def test_resend_after_failed_save_stores_conversation_and_job(
        self,
        client: AsyncClient,
        pool: asyncpg.Pool,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Replaying a request whose save failed is accepted, not dropped as a duplicate."""

        async def _boom(*args: Any, **kwargs: Any) -> Any:
            raise asyncpg.PostgresError("simulated insert failure")

        conv = build_conv("atomic-002", title="Atomic replay")
        monkeypatch.setattr("gubbi.api.v1.ingest.extraction_jobs.create_pending", _boom)
        await post(client, "extension_chatgpt", conv)
        monkeypatch.undo()

        data = await post(client, "extension_chatgpt", conv)

        assert data["conversations_saved"] == 1
        assert data["conversations_skipped_dedupe"] == 0
        assert data["extractions_enqueued"] == 1
        assert await counts_for(pool, "atomic-002") == (1, 1)

    async def test_resave_refreshes_queue_inputs_of_an_undispatched_job(
        self,
        client: AsyncClient,
        pool: asyncpg.Pool,
    ) -> None:
        """A grown thread re-uploaded before dispatch updates its queued job in place."""
        await post(client, "extension_chatgpt", build_conv("grow-seed", title="Seed"))
        await backdate_jobs(pool, PAST_ONBOARDING)
        await post(
            client,
            "extension_chatgpt",
            build_conv("grow-001", title="Growing", message_times=[OLD_ACTIVITY]),
        )
        before = await job_for(pool, "grow-001")

        fresh = datetime.now(UTC) - timedelta(minutes=1)
        data = await post(
            client,
            "extension_chatgpt",
            build_conv("grow-001", title="Growing", msg_count=6, message_times=[fresh]),
        )

        assert (data["conversations_saved"], data["extractions_enqueued"]) == (1, 1)
        assert await counts_for(pool, "grow-001") == (1, 2)
        after = await job_by_id(pool, before["id"])
        assert (before["lane"], after["lane"]) == ("backfill", "live")
        assert after["run_after"] > before["run_after"]
        assert after["activity_at"] == fresh

    async def test_resave_leaves_a_dispatched_job_unchanged(
        self,
        client: AsyncClient,
        pool: asyncpg.Pool,
    ) -> None:
        """A dispatched job keeps its queue inputs; the worker queues a follow-up instead."""
        await post(client, "extension_chatgpt", build_conv("adm-001", title="Admitted"))
        await admit_all_jobs(pool)
        before = await job_for(pool, "adm-001")

        fresh = datetime.now(UTC) - timedelta(minutes=1)
        data = await post(
            client,
            "extension_chatgpt",
            build_conv("adm-001", title="Admitted", msg_count=6, message_times=[fresh]),
        )

        assert (data["conversations_saved"], data["extractions_enqueued"]) == (1, 1)
        assert await counts_for(pool, "adm-001") == (1, 1)
        assert dict(await job_by_id(pool, before["id"])) == dict(before)

    async def test_reused_job_is_not_audited_as_created(
        self,
        client: AsyncClient,
        pool: asyncpg.Pool,
    ) -> None:
        """Only the upload that created the job writes extraction_job.created."""
        await post(client, "extension_chatgpt", build_conv("reuse-001", title="Reuse audit"))
        job = await job_for(pool, "reuse-001")

        data = await post(
            client, "extension_chatgpt", build_conv("reuse-001", title="Reuse audit", msg_count=4)
        )

        assert (data["conversations_saved"], data["extractions_enqueued"]) == (1, 1)
        assert await counts_for(pool, "reuse-001") == (1, 1)
        async with pool.acquire() as conn:
            created = await conn.fetchval(
                "SELECT COUNT(*) FROM audit_log"
                " WHERE action = 'extraction_job.created' AND target_id = $1",
                str(job["id"]),
            )
        assert created == 1

    async def test_already_in_flight_job_counts_as_queued(
        self,
        client: AsyncClient,
        pool: asyncpg.Pool,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An in-flight job for the conversation is reused: queued, not an error, not audited."""
        existing_job_id = uuid4()

        async def _already_in_flight(*args: object, **kwargs: object) -> UUID:
            raise ExtractionJobAlreadyInFlight(existing_job_id)

        monkeypatch.setattr(
            "gubbi.api.v1.ingest.extraction_jobs.create_pending", _already_in_flight
        )

        data = await post(
            client, "extension_chatgpt", build_conv("inflight-001", title="In flight")
        )

        assert data["conversations_saved"] == 1
        assert data["extractions_enqueued"] == 1
        assert data["extractions_skipped_error"] == 0
        async with pool.acquire() as conn:
            audited = await conn.fetchval(
                "SELECT COUNT(*) FROM audit_log WHERE target_id = $1",
                str(existing_job_id),
            )
        assert audited == 0
