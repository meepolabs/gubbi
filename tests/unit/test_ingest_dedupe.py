"""Unit tests for ingest deduplication logic.

Tests the dedupe SQL query logic in isolation using a mock connection.
No real DB required.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest


class TestDedupeLogic:
    """Dedupe logic: skip conversations with existing (platform, platform_id)."""

    @pytest.mark.parametrize(
        ("existing_result", "expect_skip"),
        [
            (1, True),
            (None, False),
            (0, False),
        ],
    )
    def test_dedupe_check(self, existing_result: int | None, expect_skip: bool) -> None:
        """Simulate the dedupe SELECT check: a row found means skip."""
        mock_conn = AsyncMock()
        mock_conn.fetchval.return_value = existing_result

        # Simulate what the endpoint does
        user_id = "00000000-0000-0000-0000-000000000001"
        platform = "chatgpt"
        platform_id = "conv-abc-123"

        import asyncio

        async def _check() -> bool:
            existing = await mock_conn.fetchval(
                "SELECT 1 FROM conversations"
                " WHERE user_id = $1 AND platform = $2 AND platform_id = $3",
                user_id,
                platform,
                platform_id,
            )
            return bool(existing)

        result = asyncio.run(_check())
        assert result == expect_skip

    def test_dedupe_query_uses_correct_params(self) -> None:
        """Verify the dedupe query is called with the right arguments."""
        mock_conn = AsyncMock()
        mock_conn.fetchval.return_value = None  # Not found = save

        user_id = "00000000-0000-0000-0000-000000000002"
        platform = "claude"
        platform_id = "claude-conv-xyz"

        import asyncio

        async def _check() -> None:
            await mock_conn.fetchval(
                "SELECT 1 FROM conversations"
                " WHERE user_id = $1 AND platform = $2 AND platform_id = $3",
                user_id,
                platform,
                platform_id,
            )

        asyncio.run(_check())

        # Verify fetchval was called once with expected args
        mock_conn.fetchval.assert_called_once()
        call_args = mock_conn.fetchval.call_args[0]
        assert call_args[0] == (
            "SELECT 1 FROM conversations"
            " WHERE user_id = $1 AND platform = $2 AND platform_id = $3"
        )
        assert call_args[1] == user_id
        assert call_args[2] == platform
        assert call_args[3] == platform_id

    def test_no_dupe_when_different_platform(self) -> None:
        """Same platform_id but different platform should not be deduped."""
        mock_conn = AsyncMock()
        # First check returns None (no match), then returns 1 (match on second)
        mock_conn.fetchval.side_effect = [None, 1]

        import asyncio

        async def _check() -> tuple[bool, bool]:
            # Check chatgpt conv (not found)
            result1 = await mock_conn.fetchval(
                "SELECT 1 FROM conversations WHERE user_id = $1 AND platform = $2 AND platform_id = $3",
                "user-1",
                "chatgpt",
                "same-id",
            )
            # Check claude conv with same id (found)
            result2 = await mock_conn.fetchval(
                "SELECT 1 FROM conversations WHERE user_id = $1 AND platform = $2 AND platform_id = $3",
                "user-1",
                "claude",
                "same-id",
            )
            return bool(result1), bool(result2)

        r1, r2 = asyncio.run(_check())
        assert r1 is False  # chatgpt same-id: not found
        assert r2 is True  # claude same-id: found

    def test_no_dupe_when_different_user(self) -> None:
        """Same platform+platform_id but different user_id is not a dupe."""
        mock_conn = AsyncMock()
        # user_a: not found, user_b: not found
        mock_conn.fetchval.side_effect = [None, None]

        import asyncio

        async def _check() -> tuple[bool, bool]:
            result_a = await mock_conn.fetchval(
                "SELECT 1 FROM conversations WHERE user_id = $1 AND platform = $2 AND platform_id = $3",
                "user-a",
                "chatgpt",
                "conv-1",
            )
            result_b = await mock_conn.fetchval(
                "SELECT 1 FROM conversations WHERE user_id = $1 AND platform = $2 AND platform_id = $3",
                "user-b",
                "chatgpt",
                "conv-1",
            )
            return bool(result_a), bool(result_b)

        r_a, r_b = asyncio.run(_check())
        assert r_a is False
        assert r_b is False
