"""Repository module for onboarding setup-signal probes.

Two cheap EXISTS checks that derive booleans from the user's journal data.
Both rely on RLS (the ``app.current_user_id`` GUC set by
``user_scoped_connection``) to scope rows to the current user -- no explicit
user_id predicate is needed.

These are existence probes only: no decryption, no row hydration. They back
the setup-state composer in cloud-api, which must not query journal tables
directly (data-plane boundary).

Public surface
--------------
SetupSignals      -- dataclass returned by get_setup_signals
get_setup_signals -- single round trip: (has_entries, has_synced_conversations)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import asyncpg

__all__: list[str] = [
    "SYNCED_CONVERSATION_SOURCES",
    "SetupSignals",
    "get_setup_signals",
]

# Source labels written by the conversation ingest endpoint's request body and
# the extraction pipeline. The stored ``conversations.source`` column, however,
# carries the platform name (``chatgpt`` / ``claude``) for ingested rows -- the
# same value a manually-saved conversation can carry -- so ``source`` alone
# cannot tell a synced conversation from a hand-saved one. ``platform_id`` is
# the reliable discriminator: it is populated only on the ingest (sync) path.
# The source set is kept as a defensive secondary signal in case future writers
# persist these labels onto the column directly.
SYNCED_CONVERSATION_SOURCES: tuple[str, ...] = (
    "extension_chatgpt",
    "extension_claude",
    "zip_upload",
)


@dataclass(frozen=True)
class SetupSignals:
    """Onboarding-derived booleans for the current RLS-scoped user.

    has_entries               : at least one non-soft-deleted entry exists.
    has_synced_conversations  : at least one conversation arrived via the
                                extension/zip ingest (sync) path.
    """

    has_entries: bool
    has_synced_conversations: bool


async def get_setup_signals(conn: asyncpg.Connection) -> SetupSignals:
    """Return the two onboarding signals in a single round trip.

    RLS scopes both EXISTS probes to the current user; no user_id predicate is
    needed. A synced conversation is one carrying a ``platform_id`` (set only by
    the ingest path) or one of the known ingest source labels.
    """
    row = await conn.fetchrow(
        """
        SELECT
            EXISTS (
                SELECT 1 FROM entries WHERE deleted_at IS NULL
            ) AS has_entries,
            EXISTS (
                SELECT 1 FROM conversations
                WHERE platform_id IS NOT NULL
                   OR source = ANY($1::text[])
            ) AS has_synced_conversations
        """,
        list(SYNCED_CONVERSATION_SOURCES),
    )
    if row is None:
        return SetupSignals(has_entries=False, has_synced_conversations=False)
    return SetupSignals(
        has_entries=bool(row["has_entries"]),
        has_synced_conversations=bool(row["has_synced_conversations"]),
    )
