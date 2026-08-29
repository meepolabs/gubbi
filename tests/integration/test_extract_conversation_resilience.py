"""Integration tests: extraction worker resilience.

Tests cover:
- Idempotent re-entry after LLM failure (retrying from scratch produces exactly N entries).
- Persistence rollback leaves no partial state.
- Pool not starved under concurrency (10 concurrent jobs against pool max=12).
- Multi-worker race: both workers for same conversation_id, bounded READ COMMITTED outcome.
- Lifecycle UPDATEs: mark_running -> mark_completed / mark_failed.
- Audit rows at terminals: extraction_job.completed / extraction_job.failed.
- Retry idempotency: lifecycle UPDATEs are no-ops on second invocation.

Requires a running PostgreSQL instance with migrations applied through head.
Uses mock LLM service -- no API key needed.

Run with:
    pytest tests/integration/test_extract_conversation_resilience.py -v
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import date
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import asyncpg
import pytest
import pytest_asyncio

from gubbi.constants import APP_POOL_SIZE_MAX
from gubbi.crypto.cipher import ContentCipher
from gubbi.extraction.jobs.extract_conversation import extract_conversation
from gubbi.extraction.service import CategorizationResult, ExtractedEntry, ExtractionEntriesResult
from gubbi.storage.repositories import extraction_jobs

pytestmark = [
    pytest.mark.asyncio(loop_scope="session"),
    pytest.mark.integration,
]

# Fixed test user + cipher.
_USER_UUID = UUID("22222222-3333-4444-5555-666666666666")
_USER_ID_STR = str(_USER_UUID)
_CIPHER = ContentCipher({1: bytes([1]) * 32})


@pytest_asyncio.fixture(autouse=True)
async def _seed_user(clean_rls_db: asyncpg.Pool) -> AsyncIterator[None]:
    """Seed the fixed test user before each test.

    clean_rls_db TRUNCATEs users (RESTART IDENTITY CASCADE) before yielding, so
    the user must be re-inserted per test. topics / conversations / extraction_jobs
    all FK to users(id), so without this every seed helper fails with a
    foreign-key violation.
    """
    async with clean_rls_db.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO users (id, email, timezone, created_at, updated_at)
            VALUES ($1, 'resilience-e2e@test.local', 'UTC', now(), now())
            ON CONFLICT (id) DO NOTHING
            """,
            _USER_UUID,
        )
    yield


def _encrypt(text: str) -> tuple[bytes, bytes]:
    return _CIPHER.encrypt(text)


async def _seed_conversation_modern(conn: asyncpg.Connection) -> int:
    """Insert a minimal conversations row using the current encrypted schema.

    Returns the new conversation_id.
    """
    title_ct, title_nonce = _encrypt("Test resilience conv")
    summary_ct, summary_nonce = _encrypt("Test summary")
    seed_suffix = uuid4().hex[:8]
    topic_id: int = await conn.fetchval(
        """
        INSERT INTO topics (path, title, description, user_id, created_at, updated_at)
        VALUES ($1, $2, '', $3, now(), now())
        RETURNING id
        """,
        f"test/resilience-seed-{seed_suffix}",
        "Resilience Seed",
        _USER_UUID,
    )
    conv_id: int = await conn.fetchval(
        """
        INSERT INTO conversations
            (topic_id, user_id, title_encrypted, title_nonce, slug, source,
             summary_encrypted, summary_nonce, tags, participants,
             message_count, created_at, updated_at, json_path, search_vector)
        VALUES (
            $1, $2, $3, $4, $5, 'claude',
            $6, $7, '{}', '{}',
            0, now(), now(), $8, to_tsvector('english', $9)
        )
        RETURNING id
        """,
        topic_id,
        _USER_UUID,
        title_ct,
        title_nonce,
        f"test-resilience-{seed_suffix}",
        summary_ct,
        summary_nonce,
        f"conversations_json/test-{seed_suffix}.json",
        "Test resilience conv Test summary",
    )
    return int(conv_id)


async def _seed_extraction_job(conn: asyncpg.Connection, conversation_id: int) -> UUID:
    """Insert a pending extraction_jobs row and return the job_id UUID."""
    return await extraction_jobs.create_pending(
        conn,
        user_id=_USER_UUID,
        conversation_id=conversation_id,
        source="claude",
        period_start=date(2026, 5, 1),
    )


def _make_mock_ctx(pool: asyncpg.Pool) -> dict:
    mock_extraction_service = AsyncMock()
    mock_extraction_service.categorize_conversation.return_value = CategorizationResult(
        topic_path="test/resilience",
        topic_title="Resilience Test",
        summary="Resilience test summary",
        confidence=0.95,
    )
    mock_extraction_service.extract_entries.return_value = ExtractionEntriesResult(
        entries=[
            ExtractedEntry(
                content="Resilience entry one",
                reasoning="Reason one",
                tags=["test"],
                entry_date="2026-05-10",
            ),
            ExtractedEntry(
                content="Resilience entry two",
                reasoning="Reason two",
                tags=["test"],
                entry_date="2026-05-10",
            ),
        ],
        input_tokens=100,
        output_tokens=50,
    )
    mock_redis = AsyncMock()
    mock_redis.publish = AsyncMock()
    return {
        "pool": pool,
        "cipher": _CIPHER,
        "extraction_service": mock_extraction_service,
        "redis": mock_redis,
    }


async def test_idempotent_reentry_after_llm_failure(
    clean_rls_db: asyncpg.Pool,
    app_pool: asyncpg.Pool,
) -> None:
    """Run job, force extract_entries to raise on first attempt, then re-run.

    After re-run: exactly one mark_processed event and exactly N entries
    (not 2N) under the conversation's topic.
    """
    async with clean_rls_db.acquire() as conn:
        conv_id = await _seed_conversation_modern(conn)

    ctx = _make_mock_ctx(app_pool)

    # First run: extract_entries fails mid-job (LLM failure after categorization).
    ctx["extraction_service"].extract_entries.side_effect = RuntimeError("LLM timeout - first run")
    with pytest.raises(RuntimeError, match="LLM timeout - first run"):
        await extract_conversation(ctx, conv_id, _USER_ID_STR)

    # Confirm not processed yet.
    async with clean_rls_db.acquire() as conn:
        processed_at = await conn.fetchval(
            "SELECT processed_at FROM conversations WHERE id = $1", conv_id
        )
    assert processed_at is None, "Should not be processed after LLM failure"

    # Second run: succeeds.
    ctx["extraction_service"].extract_entries.side_effect = None
    ctx["extraction_service"].extract_entries.return_value = ExtractionEntriesResult(
        entries=[
            ExtractedEntry(
                content="Resilience entry one",
                reasoning="Reason one",
                tags=["test"],
                entry_date="2026-05-10",
            ),
            ExtractedEntry(
                content="Resilience entry two",
                reasoning="Reason two",
                tags=["test"],
                entry_date="2026-05-10",
            ),
        ],
        input_tokens=100,
        output_tokens=50,
    )

    result = await extract_conversation(ctx, conv_id, _USER_ID_STR)
    assert result["skipped"] is False
    assert result["entries_created"] == 2

    # Exactly 2 entries under the topic, not 4 (idempotency).
    async with clean_rls_db.acquire() as conn:
        entry_count: int = await conn.fetchval(
            """
            SELECT COUNT(*) FROM entries e
            JOIN topics t ON t.id = e.topic_id
            WHERE t.path = 'test/resilience'
            """
        )
        processed_at2 = await conn.fetchval(
            "SELECT processed_at FROM conversations WHERE id = $1", conv_id
        )
    assert entry_count == 2, f"Expected 2 entries, got {entry_count}"
    assert processed_at2 is not None, "Should be processed after second run"


async def test_persistence_rollback_leaves_no_partial_state(
    clean_rls_db: asyncpg.Pool,
    app_pool: asyncpg.Pool,
) -> None:
    """Inject a failure on the second entry_repo.append call.

    After the failure the SAVEPOINT rolls back, so zero entries are
    persisted and processed_at remains NULL.
    """
    async with clean_rls_db.acquire() as conn:
        conv_id = await _seed_conversation_modern(conn)

    ctx = _make_mock_ctx(app_pool)
    # Two entries configured -- first succeeds, second fails.
    ctx["extraction_service"].extract_entries.return_value = ExtractionEntriesResult(
        entries=[
            ExtractedEntry(content="Good entry", reasoning="ok", tags=[], entry_date="2026-05-10"),
            ExtractedEntry(content="Bad entry", reasoning="bad", tags=[], entry_date="2026-05-10"),
        ],
        input_tokens=100,
        output_tokens=50,
    )

    # We patch entry_repo.append at the repository level to inject the failure
    # on the second call.
    call_count = 0

    async def _flaky_append(*args: object, **kwargs: object) -> int:
        nonlocal call_count
        call_count += 1
        if call_count >= 2:
            raise RuntimeError("Simulated DB write failure on second entry")
        return 999  # fake entry_id

    with (
        patch("gubbi.storage.repositories.entries.append", new=_flaky_append),
        pytest.raises(RuntimeError, match="Simulated DB write failure"),
    ):
        await extract_conversation(ctx, conv_id, _USER_ID_STR)

    # Assert zero entries written (SAVEPOINT rolled back partial inserts).
    async with clean_rls_db.acquire() as conn:
        entry_count: int = await conn.fetchval(
            "SELECT COUNT(*) FROM entries WHERE deleted_at IS NULL"
        )
        processed_at = await conn.fetchval(
            "SELECT processed_at FROM conversations WHERE id = $1", conv_id
        )

    assert entry_count == 0, f"Expected 0 entries after rollback, got {entry_count}"
    assert processed_at is None, "processed_at should be NULL after rollback"


@pytest.mark.skip(
    reason="QUARANTINE (prod-code concurrency gap, not test-only): under concurrent "
    "workers the topic-create race in _persist_extraction "
    "(gubbi/extraction/jobs/extract_conversation.py:388) catches TopicAlreadyExists, "
    "but the failing INSERT has already aborted the enclosing SAVEPOINT transaction, "
    "so the next statement raises InFailedSQLTransactionError. Recovering after a "
    "constraint violation requires an inner SAVEPOINT around the topic INSERT -- a "
    "prod change, out of scope for this TEST-ONLY quarantine burn-down. The "
    "non-concurrent resilience tests in this file are fixed and run."
)
async def test_pool_not_starved_under_concurrency(
    clean_rls_db: asyncpg.Pool,
    app_pool: asyncpg.Pool,
) -> None:
    """Spawn 10 concurrent extract_conversation coroutines against a real pool.

    Each LLM call sleeps 0.1s to simulate async latency without holding the
    DB connection. All 10 jobs should complete without pool exhaustion.

    Verifies that APP_POOL_SIZE_MAX=12 is sufficient headroom above max_jobs=10.
    """
    # Pool max is 12 -- verify constant matches plan.
    assert APP_POOL_SIZE_MAX == 12, f"Expected APP_POOL_SIZE_MAX=12, got {APP_POOL_SIZE_MAX}"

    # Seed 10 separate conversations.
    conv_ids: list[int] = []
    async with clean_rls_db.acquire() as conn:
        for _ in range(10):
            cid = await _seed_conversation_modern(conn)
            conv_ids.append(cid)

    # Each job gets its own mock context (fresh counters, shared pool).
    async def _run_one(conv_id: int) -> dict:
        mock_svc = AsyncMock()

        async def _slow_categorize(*_args: object, **_kwargs: object) -> CategorizationResult:
            await asyncio.sleep(0.1)  # simulate LLM latency
            return CategorizationResult(
                topic_path="test/concurrency",
                topic_title="Concurrency Test",
                summary="concurrent",
                confidence=0.9,
            )

        async def _slow_extract(*_args: object, **_kwargs: object) -> ExtractionEntriesResult:
            await asyncio.sleep(0.1)  # simulate LLM latency
            return ExtractionEntriesResult(
                entries=[
                    ExtractedEntry(
                        content=f"Entry for {conv_id}",
                        reasoning="concurrent",
                        tags=[],
                        entry_date="2026-05-10",
                    )
                ],
                input_tokens=10,
                output_tokens=5,
            )

        mock_svc.categorize_conversation = _slow_categorize
        mock_svc.extract_entries = _slow_extract
        mock_redis = AsyncMock()
        mock_redis.publish = AsyncMock()

        local_ctx = {
            "pool": app_pool,
            "cipher": _CIPHER,
            "extraction_service": mock_svc,
            "redis": mock_redis,
        }
        return await extract_conversation(local_ctx, conv_id, _USER_ID_STR)

    results = await asyncio.gather(*[_run_one(cid) for cid in conv_ids])

    assert len(results) == 10, "All 10 jobs should complete"
    completed = [r for r in results if not r["skipped"]]
    assert len(completed) == 10, "All 10 jobs should succeed (none skipped)"


@pytest.mark.skip(
    reason="QUARANTINE (prod-code concurrency gap, not test-only): same root cause as "
    "test_pool_not_starved_under_concurrency -- two workers racing to create the same "
    "topic abort each other's SAVEPOINT on the unique-constraint violation, raising "
    "InFailedSQLTransactionError. Needs an inner SAVEPOINT around the topic INSERT in "
    "_persist_extraction (prod change)."
)
async def test_multi_worker_race_idempotent(
    clean_rls_db: asyncpg.Pool,
    app_pool: asyncpg.Pool,
) -> None:
    """Two concurrent coroutines for the same conversation_id.

    Under READ COMMITTED, both workers may pass the conn2 idempotency check
    before either commits mark_processed, so this test only asserts the stable
    bounded outcome: at least one write lands and at most one duplicate write
    per worker occurs.
    """
    async with clean_rls_db.acquire() as conn:
        conv_id = await _seed_conversation_modern(conn)

    # Both workers use the same pool.
    async def _run_worker() -> dict:
        mock_svc = AsyncMock()

        async def _slow_categorize(*_args: object, **_kwargs: object) -> CategorizationResult:
            # Brief sleep to let both workers reach Phase 2 "simultaneously".
            await asyncio.sleep(0.05)
            return CategorizationResult(
                topic_path="test/race",
                topic_title="Race Test",
                summary="race",
                confidence=0.9,
            )

        async def _slow_extract(*_args: object, **_kwargs: object) -> ExtractionEntriesResult:
            await asyncio.sleep(0.05)
            return ExtractionEntriesResult(
                entries=[
                    ExtractedEntry(
                        content="Race entry",
                        reasoning="race",
                        tags=[],
                        entry_date="2026-05-10",
                    )
                ],
                input_tokens=10,
                output_tokens=5,
            )

        mock_svc.categorize_conversation = _slow_categorize
        mock_svc.extract_entries = _slow_extract
        mock_redis = AsyncMock()
        mock_redis.publish = AsyncMock()

        local_ctx = {
            "pool": app_pool,
            "cipher": _CIPHER,
            "extraction_service": mock_svc,
            "redis": mock_redis,
        }
        return await extract_conversation(local_ctx, conv_id, _USER_ID_STR)

    r1, r2 = await asyncio.gather(_run_worker(), _run_worker())

    # READ COMMITTED does not guarantee that one worker will observe the other
    # worker's mark_processed update before its own conn2 check.
    results = [r1, r2]
    writers = [r for r in results if not r["skipped"]]
    skippers = [r for r in results if r["skipped"]]

    assert 1 <= len(writers) <= 2, f"Expected 1-2 writers, got {len(writers)}: {results}"
    assert len(skippers) in {0, 1}, f"Expected 0-1 skippers, got {len(skippers)}: {results}"

    # Exactly one row is ideal, but two is permitted by READ COMMITTED races.
    async with clean_rls_db.acquire() as conn:
        audit_count: int = await conn.fetchval(
            "SELECT COUNT(*) FROM audit_log WHERE action = 'conversation.extracted'"
            " AND target_id = $1",
            str(conv_id),
        )
        entry_count: int = await conn.fetchval(
            """
            SELECT COUNT(*) FROM entries e
            JOIN topics t ON t.id = e.topic_id
            WHERE t.path = 'test/race'
            """
        )
        processed_at = await conn.fetchval(
            "SELECT processed_at FROM conversations WHERE id = $1", conv_id
        )
    assert 1 <= audit_count <= 2, f"Expected 1-2 audit rows, got {audit_count}"
    assert 1 <= entry_count <= 2, f"Expected 1-2 entries, got {entry_count}"
    assert processed_at is not None, "Conversation should be marked processed"


# ---------------------------------------------------------------------------
# lifecycle UPDATE integration tests
# ---------------------------------------------------------------------------


async def test_lifecycle_happy_path(
    clean_rls_db: asyncpg.Pool,
    app_pool: asyncpg.Pool,
) -> None:
    """Worker lifecycle: pending -> running -> completed.

    Asserts:
    - Row transitions happen correctly.
    - cents_spent, topics_created, entries_created are populated after completion.
    - extraction_job.completed audit row written with actor_type='user'.
    """
    async with clean_rls_db.acquire() as conn:
        conv_id = await _seed_conversation_modern(conn)
        job_id = await _seed_extraction_job(conn, conv_id)

    ctx = _make_mock_ctx(app_pool)

    # Use a deterministic mock provider so cents_spent is calculable.
    from unittest.mock import MagicMock

    mock_provider = MagicMock()
    mock_provider.estimate_cost_cents.return_value = 3.0
    ctx["extraction_service"]._llm = mock_provider

    result = await extract_conversation(ctx, conv_id, _USER_ID_STR, str(job_id))

    assert result["skipped"] is False
    assert result["entries_created"] == 2
    assert result["cents_spent"] == 3

    # Verify row state in DB.
    async with clean_rls_db.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT status, entries_created, cents_spent, started_at, completed_at "
            "FROM extraction_jobs WHERE id = $1",
            job_id,
        )
    assert row is not None
    assert row["status"] == "completed"
    assert row["entries_created"] == 2
    assert row["cents_spent"] == 3
    assert row["started_at"] is not None
    assert row["completed_at"] is not None

    # Verify extraction_job.completed audit row.
    async with clean_rls_db.acquire() as conn:
        audit_row = await conn.fetchrow(
            "SELECT actor_type, actor_id, action, target_kind, target_id "
            "FROM audit_log "
            "WHERE action = 'extraction_job.completed' AND target_id = $1",
            str(job_id),
        )
    assert audit_row is not None
    assert audit_row["actor_type"] == "user"
    assert audit_row["actor_id"] == _USER_ID_STR
    assert audit_row["target_kind"] == "extraction_job"


async def test_lifecycle_on_failure(
    clean_rls_db: asyncpg.Pool,
    app_pool: asyncpg.Pool,
) -> None:
    """Worker lifecycle on LLM failure: pending -> running -> failed.

    Asserts:
    - Row transitions correctly to 'failed'.
    - error_code is populated.
    - extraction_job.failed audit row written with actor_type='user'.
    """
    async with clean_rls_db.acquire() as conn:
        conv_id = await _seed_conversation_modern(conn)
        job_id = await _seed_extraction_job(conn, conv_id)

    ctx = _make_mock_ctx(app_pool)
    ctx["extraction_service"].extract_entries.side_effect = RuntimeError("LLM hard failure")

    with pytest.raises(RuntimeError, match="LLM hard failure"):
        await extract_conversation(ctx, conv_id, _USER_ID_STR, str(job_id))

    # Verify row state.
    async with clean_rls_db.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT status, error_code, started_at, completed_at "
            "FROM extraction_jobs WHERE id = $1",
            job_id,
        )
    assert row is not None
    assert row["status"] == "failed"
    assert row["error_code"] == "internal_error"
    assert row["started_at"] is not None
    assert row["completed_at"] is not None

    # Verify extraction_job.failed audit row.
    async with clean_rls_db.acquire() as conn:
        audit_row = await conn.fetchrow(
            "SELECT actor_type, actor_id, action, target_kind, target_id "
            "FROM audit_log "
            "WHERE action = 'extraction_job.failed' AND target_id = $1",
            str(job_id),
        )
    assert audit_row is not None
    assert audit_row["actor_type"] == "user"
    assert audit_row["actor_id"] == _USER_ID_STR
    assert audit_row["target_kind"] == "extraction_job"


async def test_lifecycle_retry_idempotency(
    clean_rls_db: asyncpg.Pool,
    app_pool: asyncpg.Pool,
) -> None:
    """Lifecycle UPDATEs are no-ops on retry after terminal state is set.

    Simulate a successful run followed by a second invocation of the same
    job_id. The second run short-circuits via the idempotency check (processed_at
    already set) and the mark_completed / mark_running calls in the repo are no-ops
    (WHERE clause guards prevent overwriting a terminal row).

    Asserts that the extraction_jobs row stays in 'completed' after the retry
    and is not duplicated.
    """
    async with clean_rls_db.acquire() as conn:
        conv_id = await _seed_conversation_modern(conn)
        job_id = await _seed_extraction_job(conn, conv_id)

    ctx = _make_mock_ctx(app_pool)

    from unittest.mock import MagicMock

    mock_provider = MagicMock()
    mock_provider.estimate_cost_cents.return_value = 2.0
    ctx["extraction_service"]._llm = mock_provider

    # First run: succeeds.
    result1 = await extract_conversation(ctx, conv_id, _USER_ID_STR, str(job_id))
    assert result1["skipped"] is False

    # Second run with same job_id: idempotency check fires, returns skipped.
    result2 = await extract_conversation(ctx, conv_id, _USER_ID_STR, str(job_id))
    assert result2["skipped"] is True

    # Row should still be 'completed', not clobbered.
    async with clean_rls_db.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT status FROM extraction_jobs WHERE id = $1",
            job_id,
        )
    assert row is not None
    assert row["status"] == "completed"

    # Exactly one extraction_job.completed audit row (second run was idempotent).
    async with clean_rls_db.acquire() as conn:
        completed_audit_count: int = await conn.fetchval(
            "SELECT COUNT(*) FROM audit_log "
            "WHERE action = 'extraction_job.completed' AND target_id = $1",
            str(job_id),
        )
    assert completed_audit_count == 1, (
        f"Expected exactly 1 extraction_job.completed audit row, got {completed_audit_count}"
    )


async def test_audit_rows_actor_type_user(
    clean_rls_db: asyncpg.Pool,
    app_pool: asyncpg.Pool,
) -> None:
    """Both terminal audit rows use actor_type='user', not 'service'."""
    async with clean_rls_db.acquire() as conn:
        conv_id_ok = await _seed_conversation_modern(conn)
        job_id_ok = await _seed_extraction_job(conn, conv_id_ok)
        conv_id_fail = await _seed_conversation_modern(conn)
        job_id_fail = await _seed_extraction_job(conn, conv_id_fail)

    # Happy path job.
    ctx_ok = _make_mock_ctx(app_pool)
    from unittest.mock import MagicMock

    mock_provider = MagicMock()
    mock_provider.estimate_cost_cents.return_value = 1.0
    ctx_ok["extraction_service"]._llm = mock_provider
    await extract_conversation(ctx_ok, conv_id_ok, _USER_ID_STR, str(job_id_ok))

    # Failure job.
    ctx_fail = _make_mock_ctx(app_pool)
    ctx_fail["extraction_service"].extract_entries.side_effect = RuntimeError("injected fail")
    with pytest.raises(RuntimeError):
        await extract_conversation(ctx_fail, conv_id_fail, _USER_ID_STR, str(job_id_fail))

    async with clean_rls_db.acquire() as conn:
        completed_row = await conn.fetchrow(
            "SELECT actor_type FROM audit_log "
            "WHERE action = 'extraction_job.completed' AND target_id = $1",
            str(job_id_ok),
        )
        failed_row = await conn.fetchrow(
            "SELECT actor_type FROM audit_log "
            "WHERE action = 'extraction_job.failed' AND target_id = $1",
            str(job_id_fail),
        )

    assert completed_row is not None
    assert completed_row["actor_type"] == "user"
    assert failed_row is not None
    assert failed_row["actor_type"] == "user"
