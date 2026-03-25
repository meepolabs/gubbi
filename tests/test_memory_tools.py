"""Unit tests for memory MCP tools.

Uses mocked MemoryService so tests run without onnxruntime/sqlite-vec installed.
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp.server.fastmcp import FastMCP

from journalctl.tools import memory

# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_memory_service() -> MagicMock:
    """Mock MemoryService with realistic return values."""
    service = MagicMock()

    service.store_memory = AsyncMock(
        return_value={
            "success": True,
            "memory": {
                "content": "test content",
                "content_hash": "abc123def456",
                "tags": [],
                "memory_type": None,
            },
        }
    )

    service.retrieve_memories = AsyncMock(
        return_value={
            "memories": [],
            "query": "test query",
            "count": 0,
        }
    )

    service.search_by_tag = AsyncMock(
        return_value={
            "memories": [],
            "tags": ["test"],
            "match_type": "ANY",
            "count": 0,
        }
    )

    service.list_memories = AsyncMock(
        return_value={
            "memories": [],
            "page": 1,
            "page_size": 10,
            "total": 0,
            "has_more": False,
        }
    )

    service.delete_memory = AsyncMock(
        return_value={
            "success": True,
            "content_hash": "abc123def456",
        }
    )

    service.health_check = AsyncMock(
        return_value={
            "healthy": True,
            "storage_type": "sqlite_vec",
            "total_memories": 0,
        }
    )

    return service


def _get_tools(memory_service: Any) -> dict[str, Any]:
    """Register memory tools and return a name→callable map."""
    mcp = FastMCP("test-memory")
    memory.register(mcp, memory_service)
    # FastMCP stores tools in _tool_manager
    return {name: tool.fn for name, tool in mcp._tool_manager._tools.items()}


# ──────────────────────────────────────────────────────────────────────────────
# memory_store
# ──────────────────────────────────────────────────────────────────────────────


class TestMemoryStore:
    async def test_store_content_only(self, mock_memory_service: MagicMock) -> None:
        tools = _get_tools(mock_memory_service)
        result = await tools["memory_store"](content="Remember this fact")
        assert result["success"] is True
        mock_memory_service.store_memory.assert_called_once_with(
            content="Remember this fact",
            tags=None,
            memory_type=None,
            metadata=None,
        )

    async def test_store_with_tags(self, mock_memory_service: MagicMock) -> None:
        tools = _get_tools(mock_memory_service)
        await tools["memory_store"](
            content="Prefers dark mode",
            tags=["preference", "ui"],
            memory_type="preference",
        )
        kwargs = mock_memory_service.store_memory.call_args.kwargs
        assert kwargs["tags"] == ["preference", "ui"]
        assert kwargs["memory_type"] == "preference"

    async def test_store_with_metadata(self, mock_memory_service: MagicMock) -> None:
        tools = _get_tools(mock_memory_service)
        await tools["memory_store"](
            content="Uses Poetry for Python",
            metadata={"source": "conversation", "confidence": "high"},
        )
        kwargs = mock_memory_service.store_memory.call_args.kwargs
        assert kwargs["metadata"] == {"source": "conversation", "confidence": "high"}

    async def test_store_returns_service_response(self, mock_memory_service: MagicMock) -> None:
        tools = _get_tools(mock_memory_service)
        result = await tools["memory_store"](content="test")
        assert "memory" in result
        assert result["memory"]["content_hash"] == "abc123def456"


# ──────────────────────────────────────────────────────────────────────────────
# memory_retrieve
# ──────────────────────────────────────────────────────────────────────────────


class TestMemoryRetrieve:
    async def test_retrieve_basic(self, mock_memory_service: MagicMock) -> None:
        tools = _get_tools(mock_memory_service)
        result = await tools["memory_retrieve"](query="what are my preferences")
        assert "memories" in result
        mock_memory_service.retrieve_memories.assert_called_once_with(
            query="what are my preferences",
            n_results=10,
            tags=None,
            memory_type=None,
        )

    async def test_retrieve_with_filters(self, mock_memory_service: MagicMock) -> None:
        tools = _get_tools(mock_memory_service)
        await tools["memory_retrieve"](
            query="database choices",
            n_results=5,
            tags=["decision"],
            memory_type="decision",
        )
        kwargs = mock_memory_service.retrieve_memories.call_args.kwargs
        assert kwargs["n_results"] == 5
        assert kwargs["tags"] == ["decision"]
        assert kwargs["memory_type"] == "decision"


# ──────────────────────────────────────────────────────────────────────────────
# memory_search_by_tag
# ──────────────────────────────────────────────────────────────────────────────


class TestMemorySearchByTag:
    async def test_search_single_tag(self, mock_memory_service: MagicMock) -> None:
        tools = _get_tools(mock_memory_service)
        result = await tools["memory_search_by_tag"](tags=["project-x"])
        assert "memories" in result
        mock_memory_service.search_by_tag.assert_called_once_with(
            tags=["project-x"],
            match_all=False,
        )

    async def test_search_match_all(self, mock_memory_service: MagicMock) -> None:
        tools = _get_tools(mock_memory_service)
        await tools["memory_search_by_tag"](tags=["project-x", "decision"], match_all=True)
        kwargs = mock_memory_service.search_by_tag.call_args.kwargs
        assert kwargs["match_all"] is True


# ──────────────────────────────────────────────────────────────────────────────
# memory_list
# ──────────────────────────────────────────────────────────────────────────────


class TestMemoryList:
    async def test_list_defaults(self, mock_memory_service: MagicMock) -> None:
        tools = _get_tools(mock_memory_service)
        result = await tools["memory_list"]()
        assert "memories" in result
        mock_memory_service.list_memories.assert_called_once_with(
            page=1,
            page_size=10,
            tag=None,
            memory_type=None,
        )

    async def test_list_with_pagination(self, mock_memory_service: MagicMock) -> None:
        tools = _get_tools(mock_memory_service)
        await tools["memory_list"](page=2, page_size=5, tag="ski")
        kwargs = mock_memory_service.list_memories.call_args.kwargs
        assert kwargs["page"] == 2
        assert kwargs["page_size"] == 5
        assert kwargs["tag"] == "ski"


# ──────────────────────────────────────────────────────────────────────────────
# memory_delete
# ──────────────────────────────────────────────────────────────────────────────


_VALID_SHA256 = "a" * 64  # Valid 64-char lowercase hex string


class TestMemoryDelete:
    async def test_delete_by_valid_hash(self, mock_memory_service: MagicMock) -> None:
        tools = _get_tools(mock_memory_service)
        result = await tools["memory_delete"](content_hash=_VALID_SHA256)
        assert result["success"] is True
        mock_memory_service.delete_memory.assert_called_once_with(content_hash=_VALID_SHA256)

    async def test_delete_invalid_hash_rejected(self, mock_memory_service: MagicMock) -> None:
        """Validation should reject short/invalid hashes without calling service."""
        tools = _get_tools(mock_memory_service)
        result = await tools["memory_delete"](content_hash="too-short")
        assert result["success"] is False
        assert "Invalid content_hash" in result["error"]
        mock_memory_service.delete_memory.assert_not_called()


# ──────────────────────────────────────────────────────────────────────────────
# memory_health
# ──────────────────────────────────────────────────────────────────────────────


class TestMemoryHealth:
    async def test_health_returns_status(self, mock_memory_service: MagicMock) -> None:
        tools = _get_tools(mock_memory_service)
        result = await tools["memory_health"]()
        assert result["healthy"] is True
        assert "total_memories" in result
        mock_memory_service.health_check.assert_called_once()


# ──────────────────────────────────────────────────────────────────────────────
# Registration
# ──────────────────────────────────────────────────────────────────────────────


class TestRegistration:
    def test_all_six_tools_registered(self, mock_memory_service: MagicMock) -> None:
        tools = _get_tools(mock_memory_service)
        expected = {
            "memory_store",
            "memory_retrieve",
            "memory_search_by_tag",
            "memory_list",
            "memory_delete",
            "memory_health",
        }
        assert expected.issubset(set(tools.keys()))

    def test_no_memory_tools_without_service(self) -> None:
        """import_tools with memory_service=None must not register memory tools."""
        from journalctl.config import get_settings
        from journalctl.import_tools import import_tools

        mcp = FastMCP("test-no-memory")
        import_tools(mcp, MagicMock(), MagicMock(), get_settings.__wrapped__(), memory_service=None)
        tool_names = set(mcp._tool_manager._tools.keys())
        assert not any(name.startswith("memory_") for name in tool_names)
