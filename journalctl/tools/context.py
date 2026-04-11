"""MCP tools: journal_briefing, journal_timeline."""

import logging
import re
from datetime import date, timedelta

from mcp.server.fastmcp import FastMCP

from journalctl.core.context import AppContext
from journalctl.core.validation import local_today
from journalctl.storage import knowledge
from journalctl.storage.constants import SNIPPET_PREVIEW_LEN
from journalctl.storage.repositories import entries as entry_repo
from journalctl.storage.repositories import topics as topic_repo
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


def _resolve_period(period: str, today: date) -> tuple[str, str, str]:
    """Resolve a period string to (date_from, date_to, label).

    Supports: 'YYYY', 'YYYY-MM', 'YYYY-WNN',
              'this-week', 'last-week', 'this-month', 'last-month'.
    Input is auto-normalized (case-insensitive, spaces/underscores become hyphens).

    today must be supplied by the caller (computed from the configured timezone)
    so that period boundaries reflect the user's local date rather than server UTC.
    """
    period = _normalize_period(period)

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


def register(mcp: FastMCP, app_ctx: AppContext) -> None:
    """Register context tools on the MCP server."""

    @mcp.tool()
    async def journal_briefing() -> dict:
        """Get the user's identity, recent activity, and topic list — the complete
        context for this person.

        Call this FIRST in every new conversation before responding.
        Without calling this, you have no memory of who this person is or what they care about.

        Returns who this person is (profile, key facts), what happened recently
        (this week's entries), and what topics they track.

        Returns:
            user_profile (text), key_facts (semantic matches for identity/status),
            this_week (label, date_from, date_to, entries list),
            topics (list of topic objects), topic_count,
            stats (total counts: topics, entries, conversations).
        """

        # User profile
        profile = knowledge.read(app_ctx.settings.journal_root, "user-profile")

        # This week's timeline
        _today = date.fromisoformat(local_today(app_ctx.settings.timezone))
        date_from, date_to, label = _resolve_period("this-week", today=_today)

        # Encode before acquiring a DB connection — keeps the pool free during inference
        import asyncio  # noqa: PLC0415

        key_facts_embedding: list[float] | None = None
        try:
            key_facts_embedding = await asyncio.to_thread(
                app_ctx.embedding_service.encode, BRIEFING_KEY_FACTS_QUERY
            )
        except Exception:
            logger.warning("Key facts encoding failed, continuing without", exc_info=True)

        key_facts: list[dict] = []
        async with app_ctx.pool.acquire() as conn:
            week_entries = await entry_repo.get_by_date_range(
                conn, date_from, date_to, limit=BRIEFING_MAX_WEEK_ENTRIES
            )
            all_topics, topic_count = await topic_repo.list_all(conn, limit=BRIEFING_MAX_TOPICS)
            stats = await entry_repo.get_stats(conn)

            if key_facts_embedding is not None:
                try:
                    raw_facts = await app_ctx.embedding_service.search_by_vector(
                        conn,
                        key_facts_embedding,
                        limit=BRIEFING_KEY_FACTS_COUNT,
                    )
                    key_facts = [{"content": r["content"]} for r in raw_facts]
                except Exception:
                    logger.warning("Key facts retrieval failed, continuing without", exc_info=True)

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

        # SQL returned DESC order (limit was set); reverse to most-recent-first
        clean_entries = list(reversed(clean_entries))

        return {
            "user_profile": profile,
            "key_facts": key_facts,
            "this_week": {
                "label": label,
                "date_from": date_from,
                "date_to": date_to,
                "entries": clean_entries,
            },
            "topics": [t.model_dump() for t in all_topics],
            "topic_count": topic_count,
            "stats": stats,
        }

    @mcp.tool()
    async def journal_timeline(period: str) -> dict:
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
            period, label (human-readable), date_from, date_to,
            entries (flat chronological list), count.
        """
        try:
            _today = date.fromisoformat(local_today(app_ctx.settings.timezone))
            date_from, date_to, label = _resolve_period(period, today=_today)
        except ValueError as e:
            return validation_error(str(e))
        async with app_ctx.pool.acquire() as conn:
            entries = await entry_repo.get_by_date_range(conn, date_from, date_to)
        return {
            "period": period,
            "label": label,
            "date_from": date_from,
            "date_to": date_to,
            "entries": entries,
            "count": len(entries),
        }
