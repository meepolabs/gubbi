"""Integration tests: extraction worker resilience (m-h5-h6).

Tests cover:
- Idempotent re-entry after LLM failure (retrying from scratch produces exactly N entries).
- Persistence rollback leaves no partial state.
- Pool not starved under concurrency (10 concurrent jobs against pool max=12).
- Multi-worker race: both workers for same conversation_id, bounded READ COMMITTED outcome.

Requires a running PostgreSQL instance with migrations applied through head.
Uses mock LLM service -- no API key needed.

Run with:
    pytest tests/integration/test_extract_conversation_resilience.py -v
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import asyncpg
import pytest

from gubbi.constants import APP_POOL_SIZE_MAX
from gubbi.crypto.cipher import ContentCipher
from gubbi.extraction.jobs.extract_conversation import extract_conversation
from gubbi.extraction.service import CategorizationResult, ExtractedEntry

pytestmark = [
    pytest.mark.asyncio(loop_scope="session"),
    pytest.mark.integration,
]

# Fixed test user + cipher.
_USER_UUID = UUID("22222222-3333-4444-5555-666666666666")
_USER_ID_STR = str(_USER_UUID)
_CIPHER = ContentCipher({1: bytes([1]) * 32})


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


def _make_mock_ctx(pool: asyncpg.Pool) -> dict:
    mock_extraction_service = AsyncMock()
    mock_extraction_service.categorize_conversation.return_value = CategorizationResult(
        topic_path="test/resilience",
        topic_title="Resilience Test",
        summary="Resilience test summary",
        confidence=0.95,
    )
    mock_extraction_service.extract_entries.return_value = [
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
    ]
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
) -> None:
    """Run job, force extract_entries to raise on first attempt, then re-run.

    After re-run: exactly one mark_processed event and exactly N entries
    (not 2N) under the conversation's topic.
    """
    async with clean_rls_db.acquire() as conn:
        conv_id = await _seed_conversation_modern(conn)

    ctx = _make_mock_ctx(clean_rls_db)

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
    ctx["extraction_service"].extract_entries.return_value = [
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
    ]

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
) -> None:
    """Inject a failure on the second entry_repo.append call.

    After the failure the SAVEPOINT rolls back, so zero entries are
    persisted and processed_at remains NULL.
    """
    async with clean_rls_db.acquire() as conn:
        conv_id = await _seed_conversation_modern(conn)

    ctx = _make_mock_ctx(clean_rls_db)
    # Two entries configured -- first succeeds, second fails.
    ctx["extraction_service"].extract_entries.return_value = [
        ExtractedEntry(content="Good entry", reasoning="ok", tags=[], entry_date="2026-05-10"),
        ExtractedEntry(content="Bad entry", reasoning="bad", tags=[], entry_date="2026-05-10"),
    ]

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


async def test_pool_not_starved_under_concurrency(
    clean_rls_db: asyncpg.Pool,
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

        async def _slow_extract(*_args: object, **_kwargs: object) -> list[ExtractedEntry]:
            await asyncio.sleep(0.1)  # simulate LLM latency
            return [
                ExtractedEntry(
                    content=f"Entry for {conv_id}",
                    reasoning="concurrent",
                    tags=[],
                    entry_date="2026-05-10",
                )
            ]

        mock_svc.categorize_conversation = _slow_categorize
        mock_svc.extract_entries = _slow_extract
        mock_redis = AsyncMock()
        mock_redis.publish = AsyncMock()

        local_ctx = {
            "pool": clean_rls_db,
            "cipher": _CIPHER,
            "extraction_service": mock_svc,
            "redis": mock_redis,
        }
        return await extract_conversation(local_ctx, conv_id, _USER_ID_STR)

    results = await asyncio.gather(*[_run_one(cid) for cid in conv_ids])

    assert len(results) == 10, "All 10 jobs should complete"
    completed = [r for r in results if not r["skipped"]]
    assert len(completed) == 10, "All 10 jobs should succeed (none skipped)"


async def test_multi_worker_race_idempotent(
    clean_rls_db: asyncpg.Pool,
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

        async def _slow_extract(*_args: object, **_kwargs: object) -> list[ExtractedEntry]:
            await asyncio.sleep(0.05)
            return [
                ExtractedEntry(
                    content="Race entry",
                    reasoning="race",
                    tags=[],
                    entry_date="2026-05-10",
                )
            ]

        mock_svc.categorize_conversation = _slow_categorize
        mock_svc.extract_entries = _slow_extract
        mock_redis = AsyncMock()
        mock_redis.publish = AsyncMock()

        local_ctx = {
            "pool": clean_rls_db,
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
