"""Unit tests for journalctl.memory.setup — env configuration and init factory."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

from journalctl.memory.bootstrap import configure_env, init_service

_TEST_DB = "memory_test.db"  # relative path, no /tmp


# ──────────────────────────────────────────────────────────────────────────────
# configure_env
# ──────────────────────────────────────────────────────────────────────────────


class TestConfigureEnv:
    def setup_method(self) -> None:
        """Remove any pre-set MCP_ vars before each test."""
        for key in (
            "MCP_MEMORY_USE_ONNX",
            "MCP_MEMORY_STORAGE_BACKEND",
            "MCP_ENABLE_AUTO_SPLIT",
            "MCP_QUALITY_BOOST_ENABLED",
        ):
            os.environ.pop(key, None)

    def test_sets_all_defaults(self) -> None:
        configure_env()
        assert os.environ["MCP_MEMORY_USE_ONNX"] == "true"
        assert os.environ["MCP_MEMORY_STORAGE_BACKEND"] == "sqlite_vec"
        assert os.environ["MCP_ENABLE_AUTO_SPLIT"] == "false"
        assert os.environ["MCP_QUALITY_BOOST_ENABLED"] == "false"

    def test_does_not_override_existing_values(self) -> None:
        os.environ["MCP_MEMORY_USE_ONNX"] = "false"
        configure_env()
        assert os.environ["MCP_MEMORY_USE_ONNX"] == "false"  # unchanged

    def test_idempotent(self) -> None:
        configure_env()
        configure_env()  # second call must not raise or change values
        assert os.environ["MCP_MEMORY_USE_ONNX"] == "true"


# ──────────────────────────────────────────────────────────────────────────────
# init_service
# ──────────────────────────────────────────────────────────────────────────────


def _mock_settings(*, enabled: bool = True, db_path: str = _TEST_DB) -> MagicMock:
    settings = MagicMock()
    settings.memory_enabled = enabled
    settings.memory_db_path = db_path
    return settings


class TestInitService:
    async def test_returns_none_when_disabled(self) -> None:
        result = await init_service(_mock_settings(enabled=False))
        assert result is None

    async def test_returns_none_on_import_error(self) -> None:
        with patch.dict(
            "sys.modules",
            {
                "mcp_memory_service": None,
                "mcp_memory_service.services": None,
                "mcp_memory_service.services.memory_service": None,
            },
        ):
            result = await init_service(_mock_settings())
        assert result is None

    async def test_returns_none_on_storage_init_error(self) -> None:
        mock_storage = MagicMock()
        mock_storage.initialize = AsyncMock(side_effect=RuntimeError("db error"))
        mock_storage_cls = MagicMock(return_value=mock_storage)

        mock_mem_mod = MagicMock()
        mock_mem_mod.MemoryService = MagicMock()
        mock_storage_mod = MagicMock()
        mock_storage_mod.SqliteVecMemoryStorage = mock_storage_cls

        with patch.dict(
            "sys.modules",
            {
                "mcp_memory_service.services.memory_service": mock_mem_mod,
                "mcp_memory_service.storage.sqlite_vec": mock_storage_mod,
            },
        ):
            result = await init_service(_mock_settings())
        assert result is None

    async def test_returns_service_on_success(self) -> None:
        mock_storage = MagicMock()
        mock_storage.initialize = AsyncMock()
        mock_storage_cls = MagicMock(return_value=mock_storage)

        mock_service = MagicMock()
        mock_service_cls = MagicMock(return_value=mock_service)

        mock_mem_mod = MagicMock()
        mock_mem_mod.MemoryService = mock_service_cls
        mock_storage_mod = MagicMock()
        mock_storage_mod.SqliteVecMemoryStorage = mock_storage_cls

        with patch.dict(
            "sys.modules",
            {
                "mcp_memory_service.services.memory_service": mock_mem_mod,
                "mcp_memory_service.storage.sqlite_vec": mock_storage_mod,
            },
        ):
            result = await init_service(_mock_settings(db_path="custom.db"))

        assert result is mock_service
        mock_storage_cls.assert_called_once_with(db_path="custom.db")
        mock_storage.initialize.assert_awaited_once()
        mock_service_cls.assert_called_once_with(mock_storage)
