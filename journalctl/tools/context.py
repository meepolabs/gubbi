"""MCP tools: journal_briefing, journal_timeline."""

import asyncio
import logging
import re
from datetime import date, timedelta
from typing import Any

import asyncpg
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from journalctl.core.cipher_guard import require_cipher
from journalctl.core.context import AppContext
from journalctl.core.crypto import DecryptionError
from journalctl.core.db_context import user_scoped_connection
from journalctl.core.scope import require_scope
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
    MAX_SEARCH_CONTENT_CHARS,
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

    @mcp.tool(
        title="Journal Briefing",
        annotations=ToolAnnotations(
            readOnlyHint=True,
        ),
    )
    @require_scope("journal:read")
    async def journal_briefing() -> dict:
        """Get the user's identity, recent activity, and topic list — the complete
        context for this person.

        Call this FIRST in every new conversation before responding.
        Without calling this, you have no memory of who this person is or what they care about.

        Returns who this person is (profile, key facts), what happened recently
        (this week's entries), and what topics they track.

        Returns:
            user_profile (str | None): None when knowledge/user-profile.md is missing,
                "" when present but empty, populated str when configured.
            user_profile_status (str): one of "configured", "missing", "empty".
            key_facts (list | None): None when no embeddings exist for this user at all,
                [] when embeddings exist but query returned no matches,
                list of dicts when results found.
            key_facts_status (str): one of "configured", "missing", "empty".
            this_week (label, date_from, date_to, entries list),
            topics (list of topic objects), topic_count,
            stats (total counts: topics, entries, conversations).
        """

        # User profile — tri-state: missing / empty / configured
        profile = knowledge.read(app_ctx.settings.data_dir, "user-profile")
        if profile is None:
            user_profile_status = "missing"
            user_profile = None
        elif profile == "":
            user_profile_status = "empty"
            user_profile = ""
        else:
            user_profile_status = "configured"
            user_profile = profile

        # This week's timeline
        _today = date.fromisoformat(local_today(app_ctx.settings.timezone))
        date_from, date_to, label = _resolve_period("this-week", today=_today)

        # Encode before acquiring a DB connection — keeps the pool free during inference
        key_facts_embedding: list[float] | None = None
        try:
            key_facts_embedding = await asyncio.to_thread(
                app_ctx.embedding_service.encode, BRIEFING_KEY_FACTS_QUERY
            )
        except Exception:
            logger.warning("Key facts encoding failed, continuing without", exc_info=True)

        cipher = require_cipher(app_ctx)
        key_facts: list[dict] | None = None
        key_facts_status: str = "missing"

        async with user_scoped_connection(app_ctx.pool) as conn:
            week_entries = await entry_repo.get_by_date_range(
                conn, cipher, date_from, date_to, limit=BRIEFING_MAX_WEEK_ENTRIES, ascending=False
            )
            all_topics, topic_count = await topic_repo.list_all(conn, limit=BRIEFING_MAX_TOPICS)
            stats = await entry_repo.get_stats(conn)

            has_embeddings: bool = await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM entry_embeddings LIMIT 1)"
            )

            raw_facts: list[dict[str, Any]] = []
            if key_facts_embedding is not None:
                try:
                    raw_facts = await app_ctx.embedding_service.search_by_vector(
                        conn,
                        key_facts_embedding,
                        limit=BRIEFING_KEY_FACTS_COUNT,
                    )
                except DecryptionError as exc:
                    logger.warning(
                        "Key facts batch decryption failed (%s)",
                        type(exc).__name__,
                        exc_info=True,
                    )
                except asyncpg.PostgresError:
                    logger.exception("Key facts batch query failed")
                except Exception:
                    logger.exception("Key facts retrieval failed unexpectedly")
                    raise

            if raw_facts:
                # Batch-collect entry IDs for a single round-trip.
                fact_entry_ids: list[int] = sorted(
                    {int(row["entry_id"]) for row in raw_facts if row.get("entry_id") is not None}
                )

                facts_list: list[dict] = []
                decrypted_entries: dict[int, tuple[str, str | None]] = {}
                try:
                    decrypted_entries = await entry_repo.get_texts(conn, cipher, fact_entry_ids)
                except asyncpg.PostgresError:
                    logger.exception(
                        "Key facts entry batch query failed for %d entries",
                        len(fact_entry_ids),
                    )

                for row in raw_facts:
                    entry_id = row.get("entry_id")
                    if entry_id is None:
                        continue
                    eid = int(entry_id)
                    if eid in decrypted_entries:
                        content, _reasoning = decrypted_entries[eid]
                        facts_list.append({"content": content[:MAX_SEARCH_CONTENT_CHARS]})

                if facts_list:
                    key_facts = facts_list
                    key_facts_status = "configured"
                elif has_embeddings:
                    key_facts = []
                    key_facts_status = "empty"
                else:
                    key_facts = None
                    key_facts_status = "missing"
            else:
                # encoding failed or no facts found
                key_facts = [] if has_embeddings else None
                key_facts_status = "empty" if has_embeddings else "missing"

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

        return {
            "user_profile": user_profile,
            "user_profile_status": user_profile_status,
            "key_facts": key_facts,
            "key_facts_status": key_facts_status,
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

    @mcp.tool(
        title="Journal Timeline",
        annotations=ToolAnnotations(
            readOnlyHint=True,
        ),
    )
    @require_scope("journal:read")
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
        cipher = require_cipher(app_ctx)
        async with user_scoped_connection(app_ctx.pool) as conn:
            entries = await entry_repo.get_by_date_range(conn, cipher, date_from, date_to)
        return {
            "period": period,
            "label": label,
            "date_from": date_from,
            "date_to": date_to,
            "entries": entries,
            "count": len(entries),
        }
