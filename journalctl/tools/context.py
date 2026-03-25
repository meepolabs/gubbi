"""MCP tools: journal_briefing, journal_timeline."""

from datetime import date, timedelta

from mcp.server.fastmcp import FastMCP

from journalctl.config import Settings
from journalctl.storage.index import SearchIndex
from journalctl.storage.markdown import MarkdownStorage


def _resolve_period(
    period: str,
) -> tuple[str, str, str]:
    """Resolve a period string to (date_from, date_to, label).

    Supports: 'YYYY', 'YYYY-MM', 'YYYY-WNN',
              'this-week', 'last-week', 'this-month', 'last-month'.
    """
    today = date.today()

    if period == "this-week":
        # ISO week: Monday = 0
        start = today - timedelta(days=today.weekday())
        end = start + timedelta(days=6)
        label = f"Week {today.isocalendar()[1]}, {today.year}"
    elif period == "last-week":
        start = today - timedelta(days=today.weekday() + 7)
        end = start + timedelta(days=6)
        label = f"Week {start.isocalendar()[1]}, {start.year}"
    elif period == "this-month":
        start = today.replace(day=1)
        # Last day of month
        if today.month == 12:
            end = today.replace(year=today.year + 1, month=1, day=1) - timedelta(days=1)
        else:
            end = today.replace(month=today.month + 1, day=1) - timedelta(days=1)
        label = f"{today.strftime('%B %Y')}"
    elif period == "last-month":
        first_this_month = today.replace(day=1)
        end = first_this_month - timedelta(days=1)
        start = end.replace(day=1)
        label = f"{start.strftime('%B %Y')}"
    elif len(period) == 4 and period.isdigit():
        # Year: YYYY
        start = date(int(period), 1, 1)
        end = date(int(period), 12, 31)
        label = period
    elif len(period) == 7 and period[4] == "-" and period[5:].isdigit():
        # Month: YYYY-MM
        year, month = int(period[:4]), int(period[5:])
        start = date(year, month, 1)
        if month == 12:
            end = date(year + 1, 1, 1) - timedelta(days=1)
        else:
            end = date(year, month + 1, 1) - timedelta(days=1)
        label = f"{start.strftime('%B %Y')}"
    elif "-W" in period:
        # ISO week: YYYY-WNN
        parts = period.split("-W")
        year, week = int(parts[0]), int(parts[1])
        start = date.fromisocalendar(year, week, 1)
        end = start + timedelta(days=6)
        label = f"Week {week}, {year}"
    else:
        msg = (
            f"Invalid period '{period}'. Use: 'YYYY', 'YYYY-MM', "
            "'YYYY-WNN', 'this-week', 'last-week', 'this-month', "
            "'last-month'."
        )
        raise ValueError(msg)

    return start.isoformat(), end.isoformat(), label


def register(
    mcp: FastMCP,
    storage: MarkdownStorage,
    index: SearchIndex,
    settings: Settings,
) -> None:
    """Register context tools on the MCP server."""

    @mcp.tool()
    async def journal_briefing() -> dict:
        """ALWAYS call this tool automatically at the start of EVERY conversation,
        before responding to the user's first message — even for casual greetings.

        This is the user's primary lifelong memory and personal context store.
        It replaces Claude's built-in memory. Without calling this, you have no
        context about who this person is, what they're working on, or what
        matters to them.

        Do not skip. Do not wait. Call it first, every time.

        Returns:
            user_profile, this_week (timeline), topics (top 20), stats (counts).
        """
        # User profile
        profile = storage.read_knowledge("user-profile")

        # This week's timeline
        date_from, date_to, label = _resolve_period("this-week")
        week_entries = index.get_entries_by_date_range(date_from, date_to)

        # Top 20 recently active topics
        all_topics = storage.list_topics()
        top_topics = all_topics[:20]

        # Stats
        stats = index.get_stats()

        return {
            "user_profile": profile,
            "this_week": {
                "label": label,
                "date_from": date_from,
                "date_to": date_to,
                "entries": week_entries,
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
        week_entries = index.get_entries_by_date_range(date_from, date_to)
        all_topics = storage.list_topics()
        top_topics = all_topics[:20]
        stats = index.get_stats()

        sections = []
        if profile:
            sections.append(f"## Profile\n{profile.strip()}")

        if week_entries:
            entry_lines = "\n".join(
                f"- {e['date']} [{e['topic']}] {str(e.get('content', ''))[:120]}"
                for e in week_entries[:15]
            )
            sections.append(f"## This Week ({label})\n{entry_lines}")
        else:
            sections.append(f"## This Week ({label})\nNo entries this week.")

        topic_lines = "\n".join(
            f"- {t.topic} ({t.entry_count} entries, updated {t.updated})" for t in top_topics
        )
        sections.append(f"## Active Topics ({len(all_topics)} total)\n{topic_lines}")

        sections.append(f"## Stats\nTotal entries: {stats.get('total_entries', 0)}")

        return "\n\n".join(sections)

    @mcp.tool()
    async def journal_timeline(
        period: str,
    ) -> dict:
        """Browse what happened during a specific time period — "what was I doing last week?"

        Use when the user asks about activity during a week, month, or year.
        Returns all journal entries for the period in chronological order.
        For keyword-specific lookups, use journal_search instead.

        Args:
            period: Time period to view. Accepts:
                    'YYYY' (year), 'YYYY-MM' (month),
                    'YYYY-WNN' (ISO week), 'this-week',
                    'last-week', 'this-month', 'last-month'.

        Returns:
            Chronological list of entries for the period,
            grouped by date and topic.
        """
        date_from, date_to, label = _resolve_period(period)
        entries = index.get_entries_by_date_range(date_from, date_to)

        return {
            "period": period,
            "label": label,
            "date_from": date_from,
            "date_to": date_to,
            "entries": entries,
            "count": len(entries),
        }
