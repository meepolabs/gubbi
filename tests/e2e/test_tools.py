"""Integration tests for MCP tools — end-to-end through the tool layer."""

from pathlib import Path
from typing import Any

import pytest

# We test tools by calling the registered async functions directly.
# FastMCP registers them as coroutines, so we invoke them the same way.
from mcp.server.fastmcp import FastMCP

from journalctl.config import get_settings
from journalctl.storage.database import DatabaseStorage
from journalctl.storage.search_index import SearchIndex
from journalctl.tools.registry import register_tools


class _StubMemoryService:
    """Minimal stub for memory_service in tool tests."""

    async def store_memory(self, content: str, **kwargs: Any) -> dict[str, Any]:
        return {"content_hash": "a" * 64, "status": "stored"}

    async def retrieve_memories(self, query: str, **kwargs: Any) -> dict[str, Any]:
        return {"memories": []}

    async def search_by_tag(self, tags: Any, **kwargs: Any) -> dict[str, Any]:
        return {"memories": []}

    async def list_memories(self, **kwargs: Any) -> dict[str, Any]:
        return {"memories": [], "total": 0}

    async def delete_memory(self, content_hash: str, **kwargs: Any) -> dict[str, Any]:
        return {"status": "deleted"}

    async def health_check(self) -> dict[str, Any]:
        return {"status": "ok"}

    async def close(self) -> None:
        pass


@pytest.fixture
def mcp_server(
    storage: DatabaseStorage,
    index: SearchIndex,
) -> FastMCP:
    """Create an MCP server with all tools registered."""
    settings = get_settings()
    mcp = FastMCP("test-journalctl")
    register_tools(mcp, storage, index, settings, memory_service=_StubMemoryService())
    return mcp


@pytest.fixture
def tools(mcp_server: FastMCP) -> dict:
    """Extract tool functions from the MCP server for direct calling."""
    tool_map = {}
    for name, tool in mcp_server._tool_manager._tools.items():
        tool_map[name] = tool.fn
    return tool_map


class TestAppendAndRead:
    """journal_append_entry and journal_read_topic round-trip."""

    @pytest.mark.asyncio
    async def test_append_and_read(self, tools: dict) -> None:
        await tools["journal_create_topic"](topic="work/acme", title="Acme Corp Notes")
        result = await tools["journal_append_entry"](
            topic="work/acme",
            content="Got the offer today.",
            tags=["decision", "milestone"],
            date="2025-06-01",
        )
        assert result["status"] == "appended"
        assert result["entry_count"] == 1
        assert "entry_id" in result

        result = await tools["journal_read_topic"](topic="work/acme")
        assert result["total"] == 1
        assert "Got the offer today." in result["entries"][0]["content"]
        assert result["entries"][0]["id"] is not None

    @pytest.mark.asyncio
    async def test_append_with_reasoning(self, tools: dict) -> None:
        await tools["journal_create_topic"](topic="work/decision", title="Work Decisions")
        result = await tools["journal_append_entry"](
            topic="work/decision",
            content="Chose SQLite as canonical storage.",
            reasoning="Markdown has no stable IDs; SQLite enables relationships.",
            tags=["decision"],
        )
        assert result["status"] == "appended"

        read = await tools["journal_read_topic"](topic="work/decision")
        assert (
            read["entries"][0]["reasoning"]
            == "Markdown has no stable IDs; SQLite enables relationships."
        )

    @pytest.mark.asyncio
    async def test_read_recent_entries(self, tools: dict) -> None:
        await tools["journal_create_topic"](topic="test/recent", title="Test Recent")
        await tools["journal_append_entry"](
            topic="test/recent", content="Entry 1", date="2024-01-01"
        )
        await tools["journal_append_entry"](
            topic="test/recent", content="Entry 2", date="2024-06-01"
        )
        await tools["journal_append_entry"](
            topic="test/recent", content="Entry 3", date="2025-01-01"
        )

        result = await tools["journal_read_topic"](topic="test/recent", limit=2)
        assert len(result["entries"]) == 2
        assert result["total"] == 3


class TestSearch:
    """journal_search."""

    @pytest.mark.asyncio
    async def test_search_finds_entry(self, tools: dict) -> None:
        await tools["journal_create_topic"](topic="work/acme", title="Acme Corp Notes")
        await tools["journal_append_entry"](
            topic="work/acme",
            content="Promotion confirmed for Q3.",
            date="2025-06-01",
        )

        result = await tools["journal_search"](query="promotion Q3")
        assert result["total"] >= 1
        assert any("work/acme" in r["topic"] for r in result["results"])

    @pytest.mark.asyncio
    async def test_search_no_results(self, tools: dict) -> None:
        result = await tools["journal_search"](query="xyznonexistent123")
        assert result["total"] == 0

    @pytest.mark.asyncio
    async def test_search_rejects_invalid_date_from(self, tools: dict) -> None:
        result = await tools["journal_search"](query="anything", date_from="not-a-date")
        assert result.get("error_code") == "INVALID_DATE"

    @pytest.mark.asyncio
    async def test_search_rejects_invalid_date_to(self, tools: dict) -> None:
        # "2025-13-01" passes the YYYY-MM-DD regex but fails strptime (month 13 is invalid)
        result = await tools["journal_search"](query="anything", date_to="2025-13-01")
        assert result.get("error_code") == "INVALID_DATE"

    @pytest.mark.asyncio
    async def test_search_limit_is_capped(self, tools: dict) -> None:
        result = await tools["journal_search"](query="anything", limit=99999)
        # Should not raise; limit is silently capped
        assert isinstance(result["results"], list)


class TestConversationFlow:
    """journal_save_conversation end-to-end."""

    @pytest.mark.asyncio
    async def test_save_and_list(self, tools: dict) -> None:
        await tools["journal_create_topic"](topic="work/acme", title="Acme Corp Notes")
        result = await tools["journal_save_conversation"](
            topic="work/acme",
            title="Q3 Planning Session",
            messages=[
                {"role": "user", "content": "What should we focus on?"},
                {"role": "assistant", "content": "Focus on impact."},
            ],
            summary="Discussed Q3 planning priorities.",
            tags=["work"],
        )
        assert result["status"] == "saved"
        assert result["summary"]

        listed = await tools["journal_list_conversations"](topic_prefix="work")
        assert listed["total"] == 1

    @pytest.mark.asyncio
    async def test_save_returns_conversation_id(self, tools: dict) -> None:
        await tools["journal_create_topic"](topic="hobbies/running", title="Running")
        result = await tools["journal_save_conversation"](
            topic="hobbies/running",
            title="Training Plan",
            messages=[
                {"role": "user", "content": "How should I train?"},
                {"role": "assistant", "content": "Start with intervals."},
            ],
            summary="Training plan discussion.",
        )
        assert "conversation_id" in result
        assert isinstance(result["conversation_id"], int)

    @pytest.mark.asyncio
    async def test_resave_updates(self, tools: dict) -> None:
        await tools["journal_create_topic"](topic="test/resave", title="Test Resave")
        msgs_v1 = [
            {"role": "user", "content": "V1 question"},
            {"role": "assistant", "content": "V1 answer"},
        ]
        msgs_v2 = [
            {"role": "user", "content": "V1 question"},
            {"role": "assistant", "content": "V1 answer"},
            {"role": "user", "content": "V2 followup"},
            {"role": "assistant", "content": "V2 answer"},
        ]

        r1 = await tools["journal_save_conversation"](
            "test/resave", "Same Chat", msgs_v1, summary="V1 chat."
        )
        assert r1["status"] == "saved"

        r2 = await tools["journal_save_conversation"](
            "test/resave", "Same Chat", msgs_v2, summary="V2 chat."
        )
        assert r2["status"] == "updated"

        conv_id = r2["conversation_id"]
        read = await tools["journal_read_conversation"](conversation_id=conv_id)
        assert read["metadata"]["message_count"] == 4


class TestTopicManagement:
    """journal_list_topics and journal_create_topic."""

    @pytest.mark.asyncio
    async def test_create_and_list(self, tools: dict) -> None:
        await tools["journal_create_topic"](
            topic="projects/alpha",
            title="Project Alpha",
            description="First project.",
            tags=["projects"],
        )
        await tools["journal_create_topic"](
            topic="projects/beta",
            title="Project Beta",
            description="Second project.",
        )

        result = await tools["journal_list_topics"](topic_prefix="projects")
        assert result["total"] == 2


class TestTimeline:
    """journal_timeline and journal_briefing."""

    @pytest.mark.asyncio
    async def test_timeline_this_week(self, tools: dict) -> None:
        await tools["journal_create_topic"](topic="test/timeline", title="Test Timeline")
        await tools["journal_append_entry"](topic="test/timeline", content="Today's entry.")

        result = await tools["journal_timeline"](period="this-week")
        assert result["count"] >= 1

    @pytest.mark.asyncio
    async def test_briefing(self, tools: dict, tmp_journal: Path) -> None:
        await tools["journal_create_topic"](topic="work/acme", title="Acme Corp Notes")
        await tools["journal_append_entry"](topic="work/acme", content="Working on the project.")

        profile_path = tmp_journal / "knowledge" / "user-profile.md"
        profile_path.write_text(
            "# User Profile\n\nSoftware engineer.",
            encoding="utf-8",
        )

        result = await tools["journal_briefing"]()
        assert "Software engineer" in result["user_profile"]
        assert result["topic_count"] >= 1
        assert "stats" in result


class TestReindex:
    """journal_reindex."""

    @pytest.mark.asyncio
    async def test_reindex(self, tools: dict) -> None:
        await tools["journal_create_topic"](topic="test/reindex", title="Test Reindex")
        await tools["journal_append_entry"](topic="test/reindex", content="Indexed entry.")

        result = await tools["journal_reindex"]()
        assert result["status"] == "rebuilt"
        assert result["documents_indexed"] >= 1
