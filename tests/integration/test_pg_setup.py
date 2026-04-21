"""Tests for storage/pg_setup.py — pool initialisation and schema bootstrap."""

import asyncpg
import pytest

# Session-scoped asyncpg pool requires tests to share its event loop.
pytestmark = pytest.mark.asyncio(loop_scope="session")


class TestPgSetup:
    """Pool connects and schema is idempotent."""

    async def test_pool_connects_and_schema_exists(self, pool: asyncpg.Pool) -> None:
        """Verify the session-scoped pool is usable and schema tables exist."""
        async with pool.acquire() as conn:
            tables = await conn.fetch("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
        table_names = {r["tablename"] for r in tables}
        assert "topics" in table_names
        assert "entries" in table_names
        assert "conversations" in table_names
        assert "messages" in table_names
        assert "entry_embeddings" in table_names
