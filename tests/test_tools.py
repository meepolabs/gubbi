"""Integration tests for MCP tools — end-to-end through the tool layer."""

from pathlib import Path

import pytest

# We test tools by calling the registered async functions directly.
# FastMCP registers them as coroutines, so we invoke them the same way.
from mcp.server.fastmcp import FastMCP

from journalctl.config import get_settings
from journalctl.import_tools import import_tools
from journalctl.storage.index import SearchIndex
from journalctl.storage.markdown import MarkdownStorage


@pytest.fixture
def mcp_server(
    storage: MarkdownStorage,
    index: SearchIndex,
) -> FastMCP:
    """Create an MCP server with all tools registered."""
    settings = get_settings()
    mcp = FastMCP("test-journalctl")
    import_tools(mcp, storage, index, settings)
    return mcp


@pytest.fixture
def tools(mcp_server: FastMCP) -> dict:
    """Extract tool functions from the MCP server for direct calling."""
    tool_map = {}
    for name, tool in mcp_server._tool_manager._tools.items():
        tool_map[name] = tool.fn
    return tool_map


class TestAppendAndRead:
    """journal_append and journal_read round-trip."""

    @pytest.mark.asyncio
    async def test_append_and_read(self, tools: dict) -> None:
        result = await tools["journal_append"](
            topic="work/acme",
            content="Got the offer today.",
            tags=["decision", "milestone"],
            date="2025-06-01",
        )
        assert result["status"] == "appended"
        assert result["entry_count"] == 1

        result = await tools["journal_read"](topic="work/acme")
        assert "Got the offer today." in result["content"]

    @pytest.mark.asyncio
    async def test_read_recent_entries(self, tools: dict) -> None:
        await tools["journal_append"](topic="test/recent", content="Entry 1", date="2024-01-01")
        await tools["journal_append"](topic="test/recent", content="Entry 2", date="2024-06-01")
        await tools["journal_append"](topic="test/recent", content="Entry 3", date="2025-01-01")

        result = await tools["journal_read"](topic="test/recent", n=2)
        assert result["showing"] == 2
        assert result["total_entries"] == 3


class TestSearch:
    """journal_search."""

    @pytest.mark.asyncio
    async def test_search_finds_entry(self, tools: dict) -> None:
        await tools["journal_append"](
            topic="work/acme",
            content="Promotion confirmed for Q3.",
            date="2025-06-01",
        )

        result = await tools["journal_search"](query="promotion Q3")
        assert result["count"] >= 1
        assert any("work/acme" in r["topic"] for r in result["results"])

    @pytest.mark.asyncio
    async def test_search_no_results(self, tools: dict) -> None:
        result = await tools["journal_search"](query="xyznonexistent123")
        assert result["count"] == 0


class TestConversationFlow:
    """journal_save_conversation end-to-end."""

    @pytest.mark.asyncio
    async def test_save_and_list(self, tools: dict) -> None:
        result = await tools["journal_save_conversation"](
            topic="work/acme",
            title="Q3 Planning Session",
            messages=[
                {"role": "user", "content": "What should we focus on?"},
                {"role": "assistant", "content": "Focus on impact."},
            ],
            tags=["work"],
        )
        assert result["status"] == "saved"
        assert "Q3 Planning Session" in result["summary"]

        listed = await tools["journal_list_conversations"](topic="work")
        assert listed["count"] == 1

    @pytest.mark.asyncio
    async def test_save_creates_topic_summary(self, tools: dict) -> None:
        await tools["journal_save_conversation"](
            topic="hobbies/running",
            title="Training Plan",
            messages=[
                {"role": "user", "content": "How should I train?"},
                {"role": "assistant", "content": "Start with intervals."},
            ],
        )

        result = await tools["journal_read"](topic="hobbies/running")
        assert "conversation-summary" in result["content"]
        assert "[[conversations/hobbies/running/training-plan]]" in result["content"]

    @pytest.mark.asyncio
    async def test_resave_updates(self, tools: dict) -> None:
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

        r1 = await tools["journal_save_conversation"]("test/resave", "Same Chat", msgs_v1)
        assert r1["status"] == "saved"

        r2 = await tools["journal_save_conversation"]("test/resave", "Same Chat", msgs_v2)
        assert r2["status"] == "updated"

        read = await tools["journal_read_conversation"](topic="test/resave", title="Same Chat")
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

        result = await tools["journal_list_topics"](prefix="projects")
        assert result["total"] == 2


class TestTimeline:
    """journal_timeline and journal_briefing."""

    @pytest.mark.asyncio
    async def test_timeline_this_week(self, tools: dict) -> None:
        await tools["journal_append"](topic="test/timeline", content="Today's entry.")

        result = await tools["journal_timeline"](period="this-week")
        assert result["count"] >= 1

    @pytest.mark.asyncio
    async def test_briefing(self, tools: dict, tmp_journal: Path) -> None:
        await tools["journal_append"](topic="work/acme", content="Working on the project.")

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
        await tools["journal_append"](topic="test/reindex", content="Indexed entry.")

        result = await tools["journal_reindex"]()
        assert result["status"] == "rebuilt"
        assert result["documents_indexed"] >= 1
