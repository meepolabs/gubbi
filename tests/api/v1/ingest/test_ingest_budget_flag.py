"""Integration tests: the ingest response's budget_exhausted flag reads the user's wallet.

Work is never refused for budget: an exhausted user's conversations are still
saved, searchable and queued.
"""

from __future__ import annotations

import asyncpg
import pytest
from httpx import AsyncClient

from gubbi.config import get_settings
from gubbi.storage.repositories.search import fts_search
from tests.fixtures.ingest_app import (
    MONTHLY,
    TEST_USER_ID,
    build_conv,
    job_for,
    post,
    spend,
)

pytestmark = pytest.mark.asyncio(loop_scope="session")


class TestIngestBudgetFlag:
    """budget_exhausted is True exactly when the user has nothing available."""

    async def test_an_exhausted_user_is_saved_searchable_queued_and_flagged(
        self,
        client: AsyncClient,
        pool: asyncpg.Pool,
        budget_env: pytest.MonkeyPatch,
    ) -> None:
        """With nothing available, work is still saved and queued, never refused."""
        await spend(pool, MONTHLY)

        data = await post(
            client, "paste_memories", build_conv("over-001", title="Zanzibar itinerary")
        )

        assert data["conversations_saved"] == 1
        assert data["extractions_enqueued"] == 1
        assert data["extractions_skipped_budget"] == 0
        assert data["budget_exhausted"] is True
        job = await job_for(pool, "over-001")
        assert (job["status"], job["admitted_at"]) == ("pending", None)
        async with pool.acquire() as conn:
            hits = await fts_search(conn, "zanzibar", None, None, None, 50)
        assert job["conversation_id"] in {h.conversation_id for h in hits}

    @pytest.mark.parametrize(
        ("spent", "granted", "exhausted"),
        [
            pytest.param(0, 0, False, id="untouched"),
            pytest.param(MONTHLY - 1, 0, False, id="one-left"),
            pytest.param(MONTHLY, 0, True, id="exactly-zero"),
            pytest.param(MONTHLY + 50, 0, True, id="negative"),
            pytest.param(MONTHLY, 10, False, id="monthly-gone-credit-left"),
            pytest.param(MONTHLY + 10, 10, True, id="credit-used-up"),
        ],
    )
    async def test_budget_exhausted_is_true_once_nothing_is_available(
        self,
        client: AsyncClient,
        pool: asyncpg.Pool,
        budget_env: pytest.MonkeyPatch,
        spent: int,
        granted: int,
        exhausted: bool,
    ) -> None:
        if spent:
            await spend(pool, spent)
        if granted:
            async with pool.acquire() as conn:
                await conn.execute(
                    "INSERT INTO credit_grants (user_id, amount, reason, idempotency_key)"
                    " VALUES ($1, $2, 'test', 'test-grant')",
                    TEST_USER_ID,
                    granted,
                )

        data = await post(client, "paste_memories", build_conv("wallet-001", title="Wallet"))

        assert data["budget_exhausted"] is exhausted

    async def test_budgets_disabled_never_flags_exhaustion(
        self,
        client: AsyncClient,
        pool: asyncpg.Pool,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("JOURNAL_LLM_WALLET_MONTHLY_AMOUNT", str(MONTHLY))
        get_settings.cache_clear()
        await spend(pool, MONTHLY * 5)

        data = await post(client, "paste_memories", build_conv("off-001", title="Off"))

        assert data["budget_exhausted"] is False
