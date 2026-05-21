"""Tests for gubbi.storage.connection.safe_acquire and safe_user_scoped_connection."""

from unittest.mock import AsyncMock, patch
from uuid import UUID

import asyncpg
import pytest

from gubbi.storage.connection import safe_acquire, safe_user_scoped_connection
from gubbi.storage.exceptions import DatabaseUnavailable

_SAMPLE_USER_ID = UUID("12345678-1234-5678-1234-567812345678")


class TestSafeAcquire:
    """Tests for the safe_acquire pool-acquisition helper."""

    async def test_safe_acquire_translates_connection_error(
        self,
    ) -> None:
        """pool.acquire raising PostgresConnectionError becomes DatabaseUnavailable."""
        error = asyncpg.PostgresConnectionError("connection refused")
        mock_pool = AsyncMock(spec=asyncpg.Pool)
        mock_pool.acquire.return_value.__aenter__.side_effect = error
        mock_pool.acquire.return_value.__aexit__.return_value = False

        with pytest.raises(DatabaseUnavailable, match="connection refused"):
            async with safe_acquire(mock_pool):
                pass  # pragma: no cover

    async def test_safe_acquire_success_path(self) -> None:
        """Normal pool.acquire yields a connection without error."""
        mock_conn = AsyncMock()
        mock_pool = AsyncMock(spec=asyncpg.Pool)
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_pool.acquire.return_value.__aexit__.return_value = False

        yielded: object | None = None
        async with safe_acquire(mock_pool) as conn:
            yielded = conn

        assert yielded is mock_conn
        mock_pool.acquire.return_value.__aenter__.assert_called_once()
        mock_pool.acquire.return_value.__aexit__.assert_called_once()

    async def test_safe_acquire_translates_cannot_connect_now(
        self,
    ) -> None:
        """pool.acquire raising CannotConnectNowError becomes DatabaseUnavailable."""
        error = asyncpg.CannotConnectNowError("starting up")
        mock_pool = AsyncMock(spec=asyncpg.Pool)
        mock_pool.acquire.return_value.__aenter__.side_effect = error
        mock_pool.acquire.return_value.__aexit__.return_value = False

        with pytest.raises(DatabaseUnavailable, match="starting up"):
            async with safe_acquire(mock_pool):
                pass  # pragma: no cover

    async def test_safe_acquire_translates_os_error(self) -> None:
        """pool.acquire raising OSError becomes DatabaseUnavailable."""
        error = OSError("network down")
        mock_pool = AsyncMock(spec=asyncpg.Pool)
        mock_pool.acquire.return_value.__aenter__.side_effect = error
        mock_pool.acquire.return_value.__aexit__.return_value = False

        with pytest.raises(DatabaseUnavailable, match="network down"):
            async with safe_acquire(mock_pool):
                pass  # pragma: no cover

    async def test_safe_acquire_translates_asyncio_timeout(self) -> None:
        """pool.acquire raising asyncio.TimeoutError (acquire budget exhausted) becomes DatabaseUnavailable."""
        # gubbi-common 0.13.1 invokes ``pool.acquire(timeout=5.0)``; asyncpg
        # raises ``asyncio.TimeoutError`` when that budget expires.
        error = TimeoutError("acquire timed out")
        mock_pool = AsyncMock(spec=asyncpg.Pool)
        mock_pool.acquire.return_value.__aenter__.side_effect = error
        mock_pool.acquire.return_value.__aexit__.return_value = False

        with pytest.raises(DatabaseUnavailable, match="acquire timed out"):
            async with safe_acquire(mock_pool):
                pass  # pragma: no cover

    async def test_safe_acquire_falls_back_to_class_name_on_bare_exception(self) -> None:
        """Bare ``TimeoutError()`` produces non-empty DatabaseUnavailable message."""
        error = TimeoutError()
        mock_pool = AsyncMock(spec=asyncpg.Pool)
        mock_pool.acquire.return_value.__aenter__.side_effect = error
        mock_pool.acquire.return_value.__aexit__.return_value = False

        with pytest.raises(DatabaseUnavailable, match="TimeoutError"):
            async with safe_acquire(mock_pool):
                pass  # pragma: no cover

    async def test_safe_acquire_does_not_translate_other_errors(
        self,
    ) -> None:
        """Non-connection exceptions propagate unchanged."""
        error = asyncpg.UniqueViolationError("duplicate key")
        mock_pool = AsyncMock(spec=asyncpg.Pool)
        mock_pool.acquire.return_value.__aenter__.side_effect = error
        mock_pool.acquire.return_value.__aexit__.return_value = False

        with pytest.raises(asyncpg.UniqueViolationError, match="duplicate key"):
            async with safe_acquire(mock_pool):
                pass  # pragma: no cover


class TestSafeUserScopedConnection:
    """Tests for the safe_user_scoped_connection helper."""

    async def test_translates_connection_error(self) -> None:
        """user_scoped_connection raising PostgresConnectionError -> DatabaseUnavailable."""
        error = asyncpg.PostgresConnectionError("connection refused")
        mock_pool = AsyncMock(spec=asyncpg.Pool)

        with patch("gubbi.storage.connection.user_scoped_connection") as mock_usc:
            mock_usc.return_value.__aenter__.side_effect = error
            mock_usc.return_value.__aexit__.return_value = False

            with pytest.raises(DatabaseUnavailable, match="connection refused"):
                async with safe_user_scoped_connection(mock_pool, _SAMPLE_USER_ID):
                    pass  # pragma: no cover

    async def test_success_path(self) -> None:
        """Normal user_scoped_connection yields a connection without error."""
        mock_pool = AsyncMock(spec=asyncpg.Pool)
        mock_conn = AsyncMock()

        with patch("gubbi.storage.connection.user_scoped_connection") as mock_usc:
            mock_usc.return_value.__aenter__.return_value = mock_conn
            mock_usc.return_value.__aexit__.return_value = False

            yielded: object | None = None
            async with safe_user_scoped_connection(mock_pool, _SAMPLE_USER_ID) as conn:
                yielded = conn

            assert yielded is mock_conn
            mock_usc.assert_called_once_with(mock_pool, _SAMPLE_USER_ID, hnsw_ef_search=100)

    async def test_translates_cannot_connect_now(self) -> None:
        """user_scoped_connection raising CannotConnectNowError -> DatabaseUnavailable."""
        error = asyncpg.CannotConnectNowError("starting up")
        mock_pool = AsyncMock(spec=asyncpg.Pool)

        with patch("gubbi.storage.connection.user_scoped_connection") as mock_usc:
            mock_usc.return_value.__aenter__.side_effect = error
            mock_usc.return_value.__aexit__.return_value = False

            with pytest.raises(DatabaseUnavailable, match="starting up"):
                async with safe_user_scoped_connection(mock_pool, _SAMPLE_USER_ID):
                    pass  # pragma: no cover

    async def test_translates_os_error(self) -> None:
        """user_scoped_connection raising OSError -> DatabaseUnavailable."""
        error = OSError("network down")
        mock_pool = AsyncMock(spec=asyncpg.Pool)

        with patch("gubbi.storage.connection.user_scoped_connection") as mock_usc:
            mock_usc.return_value.__aenter__.side_effect = error
            mock_usc.return_value.__aexit__.return_value = False

            with pytest.raises(DatabaseUnavailable, match="network down"):
                async with safe_user_scoped_connection(mock_pool, _SAMPLE_USER_ID):
                    pass  # pragma: no cover

    async def test_does_not_translate_other_errors(self) -> None:
        """Non-connection exceptions propagate unchanged."""
        error = asyncpg.UniqueViolationError("duplicate key")
        mock_pool = AsyncMock(spec=asyncpg.Pool)

        with patch("gubbi.storage.connection.user_scoped_connection") as mock_usc:
            mock_usc.return_value.__aenter__.side_effect = error
            mock_usc.return_value.__aexit__.return_value = False

            with pytest.raises(asyncpg.UniqueViolationError, match="duplicate key"):
                async with safe_user_scoped_connection(mock_pool, _SAMPLE_USER_ID):
                    pass  # pragma: no cover
