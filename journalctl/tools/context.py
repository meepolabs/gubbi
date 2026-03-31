"""MCP tools: journal_briefing, journal_timeline."""

import logging
import re
from datetime import date, timedelta

from mcp.server.fastmcp import FastMCP

from journalctl.config import Settings
from journalctl.memory.client import MemoryServiceProtocol
from journalctl.storage.constants import SNIPPET_PREVIEW_LEN
from journalctl.storage.database import DatabaseStorage
from journalctl.storage.search_index import SearchIndex
from journalctl.tools.constants import (
    BRIEFING_KEY_FACTS_COUNT,
    BRIEFING_KEY_FACTS_QUERY,
    BRIEFING_MAX_TOPICS,
    BRIEFING_MAX_WEEK_ENTRIES,
)
from journalctl.tools.errors import validation_error

logger = logging.getLogger(__name__)


def _month_end(year: int, month: int) -> date:
    """Return the last day of the given month."""
    if month == 12:
        return date(year + 1, 1, 1) - timedelta(days=1)
    return date(year, month + 1, 1) - timedelta(days=1)


def _normalize_period(period: str) -> str:
    """Normalize period string — lowercase, collapse spaces/underscores to hyphens."""
    return re.sub(r"[\s_]+", "-", period.strip().lower())


def _resolve_period(
    period: str,
) -> tuple[str, str, str]:
    """Resolve a period string to (date_from, date_to, label).

    Supports: 'YYYY', 'YYYY-MM', 'YYYY-WNN',
              'this-week', 'last-week', 'this-month', 'last-month'.
    Input is auto-normalized (case-insensitive, spaces/underscores become hyphens).
    """
    period = _normalize_period(period)
    today = date.today()

    if period == "today":
        s = today.isoformat()
        return s, s, f"Today ({s})"

    if period == "this-week":
        start = today - timedelta(days=today.weekday())
        end = start + timedelta(days=6)
        return start.isoformat(), end.isoformat(), f"Week {today.isocalendar()[1]}, {today.year}"

    if period == "last-week":
        start = today - timedelta(days=today.weekday() + 7)
        end = start + timedelta(days=6)
        return start.isoformat(), end.isoformat(), f"Week {start.isocalendar()[1]}, {start.year}"

    if period == "this-month":
        start = today.replace(day=1)
        end = _month_end(today.year, today.month)
        return start.isoformat(), end.isoformat(), today.strftime("%B %Y")

    if period == "last-month":
        end = today.replace(day=1) - timedelta(days=1)
        start = end.replace(day=1)
        return start.isoformat(), end.isoformat(), start.strftime("%B %Y")

    if len(period) == 4 and period.isdigit():
        # Year: YYYY
        try:
            year = int(period)
            return date(year, 1, 1).isoformat(), date(year, 12, 31).isoformat(), period
        except ValueError:
            pass

    if len(period) == 7 and period[4] == "-" and period[5:].isdigit():
        # Month: YYYY-MM
        try:
            year, month = int(period[:4]), int(period[5:])
            start = date(year, month, 1)
            end = _month_end(year, month)
            return start.isoformat(), end.isoformat(), start.strftime("%B %Y")
        except ValueError:
            pass

    if "-w" in period:
        # ISO week: YYYY-WNN (already lowercased by _normalize_period)
        parts = period.split("-w")
        if len(parts) == 2 and parts[0] and parts[1].isdigit():  # noqa: SIM102
            try:
                year, week = int(parts[0]), int(parts[1])
                start = date.fromisocalendar(year, week, 1)
                end = start + timedelta(days=6)
                return start.isoformat(), end.isoformat(), f"Week {week}, {year}"
            except ValueError:
                pass

    msg = (
        f"Invalid period '{period}'. Use: 'today', 'YYYY', 'YYYY-MM', "
        "'YYYY-WNN', 'this-week', 'last-week', 'this-month', 'last-month'."
    )
    raise ValueError(msg)


def register(
    mcp: FastMCP,
    storage: DatabaseStorage,
    index: SearchIndex,
    settings: Settings,
    memory_service: MemoryServiceProtocol,
) -> None:
    """Register context tools on the MCP server."""

    @mcp.tool()
    async def journal_briefing() -> dict:
        """Get the user's identity, recent activity, and topic list — the complete
        context for this person.

        Returns who this person is (profile, key facts), what happened recently
        (this week's entries), and what topics they track. Without calling this,
        you have no memory of who this person is or what they care about.

        Call before responding to the first message of every conversation.

        Returns:
            user_profile, key_facts, this_week (recent activity), topics, stats.
        """
        # User profile
        profile = storage.read_knowledge("user-profile")

        # This week's timeline
        date_from, date_to, label = _resolve_period("this-week")
        week_entries = storage.get_entries_by_date_range(date_from, date_to)

        # Clean entries for briefing output
        clean_entries = []
        for e in week_entries:
            entry = {
                "doc_type": e.get("doc_type", ""),
                "topic": e.get("topic", ""),
                "title": e.get("title", ""),
                "snippet": e.get("description", "")[:SNIPPET_PREVIEW_LEN],
                "date": e.get("updated", ""),
                "tags": e.get("tags", []),
            }
            if e.get("entry_id") is not None:
                entry["entry_id"] = e["entry_id"]
            if e.get("conversation_id") is not None:
                entry["conversation_id"] = e["conversation_id"]
            clean_entries.append(entry)

        # Most recent entries first; drop oldest when the week is very active
        clean_entries = clean_entries[-BRIEFING_MAX_WEEK_ENTRIES:][::-1]

        all_topics = storage.list_topics()
        top_topics = all_topics[:BRIEFING_MAX_TOPICS]

        # Stats
        stats = storage.get_stats()

        # Key facts — top semantic matches for user identity/preferences/status
        key_facts: list[dict] = []
        try:
            mem_response = await memory_service.retrieve_memories(
                query=BRIEFING_KEY_FACTS_QUERY,
                n_results=BRIEFING_KEY_FACTS_COUNT,
            )
            memories = mem_response.get("memories", [])
            if isinstance(memories, list):
                key_facts = [
                    {"content": m.get("content", "")}
                    for m in memories
                    if isinstance(m, dict) and m.get("content")
                ]
        except Exception:
            logger.warning("Key facts retrieval failed, continuing without", exc_info=True)

        return {
            "user_profile": profile,
            "key_facts": key_facts,
            "this_week": {
                "label": label,
                "date_from": date_from,
                "date_to": date_to,
                "entries": clean_entries,
            },
            "topics": [t.model_dump() for t in top_topics],
            "topic_count": len(all_topics),
            "stats": stats,
        }

    @mcp.prompt(
        name="journal-context",
        description=(
            "Load this at the start of every conversation. "
            "This is the user's personal context: profile, active projects, this week's activity, "
            "and memory type ontology. Use instead of (or before) any built-in memory lookup."
        ),
    )
    async def journal_context_prompt() -> str:
        """Journal context — include at conversation start for full personal context."""
        profile = storage.read_knowledge("user-profile") or ""
        date_from, date_to, label = _resolve_period("this-week")
        week_entries = storage.get_entries_by_date_range(date_from, date_to)
        all_topics = storage.list_topics()
        top_topics = all_topics[:BRIEFING_MAX_TOPICS]
        stats = storage.get_stats()

        sections = []
        if profile:
            sections.append(f"## Profile\n{profile.strip()}")

        if week_entries:
            recent = week_entries[-BRIEFING_MAX_WEEK_ENTRIES:][::-1]
            entry_lines = "\n".join(
                f"- {e['updated']} [{e['topic']}] "
                f"{str(e.get('description', ''))[:SNIPPET_PREVIEW_LEN]}"
                for e in recent
            )
            sections.append(f"## This Week ({label})\n{entry_lines}")
        else:
            sections.append(f"## This Week ({label})\nNo entries this week.")

        topic_lines = "\n".join(
            f"- {t.topic} ({t.entry_count} entries, updated {t.updated})" for t in top_topics
        )
        sections.append(f"## Active Topics ({len(all_topics)} total)\n{topic_lines}")

        n_topics = stats.get("topics", 0)
        n_convs = stats.get("conversations", 0)
        sections.append(f"## Stats\nTopics: {n_topics}, Conversations: {n_convs}")

        return "\n\n".join(sections)

    @mcp.tool()
    async def journal_timeline(
        period: str,
    ) -> dict:
        """Browse what happened during a time period — "what was I doing last week?"
        or "show me this month."

        Use when the user asks about activity during a day, week, month, or year.
        Returns entries and conversations in chronological order.

        Do NOT use for keyword search — use journal_search instead.

        Args:
            period: Time period. Accepts:
                    'today', 'this-week', 'last-week', 'this-month', 'last-month',
                    'YYYY' (e.g. '2026'), 'YYYY-MM' (e.g. '2026-03'),
                    'YYYY-WNN' (e.g. '2026-W14').

        Returns:
            Chronological list of entries for the period,
            grouped by date and topic.
        """
        try:
            date_from, date_to, label = _resolve_period(period)
        except ValueError as e:
            return validation_error(str(e))
        entries = storage.get_entries_by_date_range(date_from, date_to)

        return {
            "period": period,
            "label": label,
            "date_from": date_from,
            "date_to": date_to,
            "entries": entries,
            "count": len(entries),
        }
