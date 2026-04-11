"""Tests for storage/pg_setup.py — pool initialisation and schema bootstrap."""

import asyncpg


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

    async def test_setup_schema_is_idempotent(self, pool: asyncpg.Pool) -> None:
        """Running setup_schema twice must not raise."""
        from journalctl.storage.pg_setup import setup_schema

        await setup_schema(pool)  # Second call — should not fail
        async with pool.acquire() as conn:
            count = await conn.fetchval("SELECT COUNT(*) FROM topics")
        assert count is not None
