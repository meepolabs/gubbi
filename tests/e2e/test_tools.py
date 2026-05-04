"""Integration tests for MCP tools — end-to-end through the tool layer."""

from pathlib import Path
from typing import Any

import asyncpg
import pytest
import structlog
from mcp.server.fastmcp import FastMCP

from gubbi.config import get_settings
from gubbi.core.context import AppContext
from gubbi.tools.constants import LIST_SUMMARY_PREVIEW_CHARS
from gubbi.tools.registry import register_tools


class _StubEmbeddingService:
    """Minimal stub for EmbeddingService in tool tests.

    encode() returns a zero vector; store/search do nothing.
    This keeps tool tests fast and free of ONNX model downloads.
    """

    def encode(self, text: str) -> list[float]:
        return [0.0] * 384

    async def store(self, conn: Any, entry_id: int, text: str) -> None:
        pass

    async def search(
        self,
        conn: Any,
        text: str,
        limit: int = 10,
        topic_prefix: str | None = None,
    ) -> list[dict]:
        return []

    async def search_by_vector(
        self,
        conn: Any,
        embedding: list[float],
        limit: int = 10,
        topic_prefix: str | None = None,
        date_from: Any = None,
        date_to: Any = None,
    ) -> list[dict]:
        return []


@pytest.fixture
def mcp_server(clean_pool: asyncpg.Pool, tmp_journal: Path) -> FastMCP:
    """Create an MCP server with all tools registered against a test pool."""
    settings = get_settings()
    app_ctx = AppContext(
        pool=clean_pool,
        embedding_service=_StubEmbeddingService(),  # type: ignore[arg-type]
        settings=settings,
        logger=structlog.get_logger("test"),
    )
    mcp = FastMCP("test-gubbi")
    register_tools(mcp, app_ctx)
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

    async def test_append_and_read(self, tools: dict) -> None:
        await tools["journal_create_topic"](topic="work/acme", title="Acme Corp Notes")
        result = await tools["journal_append_entry"](
            topic="work/acme",
            content="Got the offer today.",
            tags=["decision", "milestone"],
            date="2025-06-01",
        )
        assert result["status"] == "appended"
        assert "entry_id" in result

        result = await tools["journal_read_topic"](topic="work/acme")
        assert result["total"] == 1
        assert "Got the offer today." in result["entries"][0]["content"]
        assert result["entries"][0]["id"] is not None

    async def test_append_with_reasoning(self, tools: dict) -> None:
        await tools["journal_create_topic"](topic="work/decision", title="Work Decisions")
        result = await tools["journal_append_entry"](
            topic="work/decision",
            content="Chose PostgreSQL as canonical storage.",
            reasoning="SQLite had no async driver; PostgreSQL enables concurrency.",
            tags=["decision"],
        )
        assert result["status"] == "appended"

        read = await tools["journal_read_topic"](topic="work/decision")
        assert (
            read["entries"][0]["reasoning"]
            == "SQLite had no async driver; PostgreSQL enables concurrency."
        )

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

    async def test_search_no_results(self, tools: dict) -> None:
        result = await tools["journal_search"](query="xyznonexistent123")
        assert result["total"] == 0

    async def test_search_rejects_invalid_date_from(self, tools: dict) -> None:
        result = await tools["journal_search"](query="anything", date_from="not-a-date")
        assert result.get("error_code") == "INVALID_DATE"

    async def test_search_rejects_invalid_date_to(self, tools: dict) -> None:
        result = await tools["journal_search"](query="anything", date_to="2025-13-01")
        assert result.get("error_code") == "INVALID_DATE"

    async def test_search_limit_is_capped(self, tools: dict) -> None:
        result = await tools["journal_search"](query="anything", limit=99999)
        assert isinstance(result["results"], list)


class TestConversationFlow:
    """journal_save_conversation end-to-end."""

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

    async def test_resave_updates(self, tools: dict) -> None:
        await tools["journal_create_topic"](topic="test/resave", title="Test Resave")
        msgs_v1 = [
            {"role": "user", "content": "V1 question"},
            {"role": "assistant", "content": "V1 answer"},
        ]
        msgs_v2 = msgs_v1 + [
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

    async def test_resave_same_count_different_content(self, tools: dict) -> None:
        """Regression: resaving same topic+title with same message count but
        different content must reflect the new messages, not stale ones."""
        await tools["journal_create_topic"](topic="test/stale-resave", title="Stale Resave Test")
        msgs_v1 = [
            {"role": "user", "content": "V1 question"},
            {"role": "assistant", "content": "V1 answer"},
        ]
        msgs_v2 = [
            {"role": "user", "content": "V2 edited question"},
            {"role": "assistant", "content": "V2 edited answer"},
        ]

        r1 = await tools["journal_save_conversation"](
            "test/stale-resave", "Same Title", msgs_v1, summary="V1 summary."
        )
        assert r1["status"] == "saved"

        r2 = await tools["journal_save_conversation"](
            "test/stale-resave", "Same Title", msgs_v2, summary="V2 summary."
        )
        assert r2["status"] == "updated"
        conv_id = r2["conversation_id"]

        read = await tools["journal_read_conversation"](conversation_id=conv_id)
        content = read["content"]

        # Must contain V2 content
        assert "V2 edited question" in content
        assert "V2 edited answer" in content
        # Must NOT contain stale V1 content
        assert "V1 question" not in content
        assert "V1 answer" not in content

    async def test_list_truncates_long_summary(self, tools: dict) -> None:
        """journal_list_conversations truncates summaries exceeding LIST_SUMMARY_PREVIEW_CHARS."""
        long_summary = "Long summary. " * 100  # ~1500 chars, well over LIST_SUMMARY_PREVIEW_CHARS
        assert len(long_summary) > LIST_SUMMARY_PREVIEW_CHARS
        await tools["journal_create_topic"](topic="test/truncation", title="Truncation Test")
        result = await tools["journal_save_conversation"](
            topic="test/truncation",
            title="Long Chat",
            messages=[{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}],
            summary=long_summary,
        )
        assert result["status"] == "saved"

        listed = await tools["journal_list_conversations"](topic_prefix="test/truncation")
        assert listed["total"] >= 1
        conv = listed["conversations"][0]
        assert len(conv["summary"]) == LIST_SUMMARY_PREVIEW_CHARS
        assert conv["summary_truncated"] is True

    async def test_list_short_summary_not_truncated(self, tools: dict) -> None:
        """Short summaries are returned as-is with summary_truncated=False."""
        await tools["journal_create_topic"](topic="test/short-summary", title="Short Summary Test")
        short_summary = "Brief."
        await tools["journal_save_conversation"](
            topic="test/short-summary",
            title="Short Chat",
            messages=[{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hey"}],
            summary=short_summary,
        )
        listed = await tools["journal_list_conversations"](topic_prefix="test/short-summary")
        assert listed["total"] >= 1
        conv = listed["conversations"][0]
        assert conv["summary"] == short_summary
        assert conv["summary_truncated"] is False


class TestTopicManagement:
    """journal_list_topics and journal_create_topic."""

    async def test_create_and_list(self, tools: dict) -> None:
        await tools["journal_create_topic"](
            topic="projects/alpha",
            title="Project Alpha",
            description="First project.",
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

    async def test_timeline_this_week(self, tools: dict) -> None:
        await tools["journal_create_topic"](topic="test/timeline", title="Test Timeline")
        await tools["journal_append_entry"](topic="test/timeline", content="Today's entry.")

        result = await tools["journal_timeline"](period="this-week")
        assert result["count"] >= 1

    async def test_briefing(self, tools: dict, tmp_journal: Path) -> None:
        await tools["journal_create_topic"](topic="work/acme", title="Acme Corp Notes")
        await tools["journal_append_entry"](topic="work/acme", content="Working on the project.")

        profile_path = tmp_journal / "knowledge" / "user-profile.md"
        profile_path.write_text("# User Profile\n\nSoftware engineer.", encoding="utf-8")

        result = await tools["journal_briefing"]()
        assert "Software engineer" in result["user_profile"]
        assert result["user_profile_status"] == "configured"
        assert result["topic_count"] >= 1
        assert "stats" in result

    async def test_briefing_missing_profile(self, tools: dict, tmp_journal: Path) -> None:
        # knowledge/user-profile.md does NOT exist
        result = await tools["journal_briefing"]()
        assert result["user_profile"] is None
        assert result["user_profile_status"] == "missing"

    async def test_briefing_configured_profile(self, tools: dict, tmp_journal: Path) -> None:
        profile_path = tmp_journal / "knowledge" / "user-profile.md"
        profile_path.write_text("# User\n\nName: Ada.", encoding="utf-8")
        result = await tools["journal_briefing"]()
        assert result["user_profile"] == "# User\n\nName: Ada."
        assert result["user_profile_status"] == "configured"

    async def test_briefing_empty_profile(self, tools: dict, tmp_journal: Path) -> None:
        # File exists but is empty -- distinct from missing
        profile_path = tmp_journal / "knowledge" / "user-profile.md"
        profile_path.write_text("", encoding="utf-8")
        result = await tools["journal_briefing"]()
        assert result["user_profile"] == ""
        assert result["user_profile_status"] == "empty"
