"""MCP tools: journal_briefing, journal_timeline."""

from datetime import date, timedelta

from mcp.server.fastmcp import FastMCP

from journalctl.config import Settings
from journalctl.storage.index import SearchIndex
from journalctl.storage.markdown import MarkdownStorage


def _resolve_period(
    period: str,
    tz_name: str,
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
        """Context loader — call at conversation start.

        Returns user profile, this week's activity, top 20
        recently-active topics, and system stats. Gives Claude
        enough context to be helpful immediately.

        Returns:
            user_profile, this_week (timeline), topics (top 20),
            stats (counts).
        """
        # User profile
        profile = storage.read_knowledge("user-profile")

        # This week's timeline
        date_from, date_to, label = _resolve_period(
            "this-week",
            settings.timezone,
        )
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

    @mcp.tool()
    async def journal_timeline(
        period: str,
    ) -> dict:
        """View journal activity for a time period.

        Queries the FTS5 index for all entries within the date
        range and returns them chronologically.

        Args:
            period: Time period to view. Accepts:
                    'YYYY' (year), 'YYYY-MM' (month),
                    'YYYY-WNN' (ISO week), 'this-week',
                    'last-week', 'this-month', 'last-month'.

        Returns:
            Chronological list of entries for the period,
            grouped by date and topic.
        """
        date_from, date_to, label = _resolve_period(
            period,
            settings.timezone,
        )
        entries = index.get_entries_by_date_range(date_from, date_to)

        return {
            "period": period,
            "label": label,
            "date_from": date_from,
            "date_to": date_to,
            "entries": entries,
            "count": len(entries),
        }
