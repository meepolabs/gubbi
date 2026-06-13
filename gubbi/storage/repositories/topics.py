"""Topic repository -- all SQL for topics table."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC
from datetime import datetime as datetime_cls
from typing import Any

import asyncpg
from gubbi_common.audit.actions import Action
from gubbi_common.audit.sql import record_audit_async as record_audit
from gubbi_common.audit.targets import TargetKind

from gubbi.models.journal import TopicMeta
from gubbi.storage.exceptions import TopicNotFoundError
from gubbi.storage.repositories.base import _add_param, _escape_like
from gubbi.validation import validate_title, validate_topic

__all__: list[str] = [
    "MoveCounts",
    "TopicAlreadyExists",
    "TopicMergeConflict",
    "count",
    "create",
    "delete_with_reassign",
    "get",
    "get_id",
    "list_all",
    "merge",
    "rename",
]


class TopicAlreadyExists(ValueError):
    """Raised by ``create`` when a topic path already exists for the user.

    Subclasses ``ValueError`` so existing ``except ValueError`` callers
    keep working; new callers should catch this typed class instead of
    string-matching the message.
    """


class TopicMergeConflict(ValueError):
    """Raised when moving entries/conversations would violate a uniqueness rule.

    Conversations carry a ``UNIQUE (topic_id, slug)`` constraint, so moving a
    conversation into a destination topic that already holds one with the same
    slug collides. Surface a typed error the router maps to a 409 rather than
    leaking a raw integrity error.
    """


@dataclass(frozen=True)
class MoveCounts:
    """How many rows a reassignment moved, by kind.

    ``entries`` counts ALL entries pointing at the source topic (including
    soft-deleted ones): every row holds a topic_id foreign key under
    ``ON DELETE RESTRICT``, so all of them must be reassigned before the source
    topic can be removed.
    """

    entries: int
    conversations: int


def _row_to_topic_meta(row: asyncpg.Record) -> TopicMeta:
    return TopicMeta(
        id=row["id"],
        topic=row["path"],
        title=row["title"],
        description=row["description"] or "",
        created=row["created_at"].date().isoformat(),
        updated=row["updated_at"].date().isoformat(),
        entry_count=row["entry_count"],
    )


async def get_id(conn: asyncpg.Connection, topic: str) -> int:
    """Return topic_id. Raises TopicNotFoundError if missing."""
    topic = validate_topic(topic)
    row = await conn.fetchrow("SELECT id FROM topics WHERE path = $1", topic)
    if row:
        return int(row["id"])
    msg = f"Topic '{topic}' not found -- create it first with journal_create_topic"
    raise TopicNotFoundError(msg)


async def get(conn: asyncpg.Connection, topic: str) -> TopicMeta | None:
    """Get a single topic by path."""
    topic = validate_topic(topic)
    row = await conn.fetchrow(
        """
        SELECT t.id, t.path, t.title, t.description,
               t.created_at, t.updated_at,
               COUNT(e.id) AS entry_count
        FROM topics t
        LEFT JOIN entries e ON e.topic_id = t.id AND e.deleted_at IS NULL
        WHERE t.path = $1
        GROUP BY t.id
        """,
        topic,
    )
    return _row_to_topic_meta(row) if row else None


async def create(
    conn: asyncpg.Connection,
    topic: str,
    title: str,
    description: str = "",
    created_at: datetime_cls | None = None,
) -> int:
    """Create a new topic. Returns topic_id. Raises TopicAlreadyExists if duplicate."""
    topic = validate_topic(topic)
    now = datetime_cls.now(UTC)
    created = created_at or now
    try:
        row = await conn.fetchrow(
            """
            INSERT INTO topics (path, title, description, user_id, created_at, updated_at)
            VALUES (
                $1, $2, $3,
                (SELECT NULLIF(current_setting('app.current_user_id', true), '')::uuid),
                $4, $5
            )
            RETURNING id
            """,
            topic,
            title,
            description,
            created,
            now,
        )
        if row is None:
            raise RuntimeError("INSERT topics failed: no row returned")
        return int(row["id"])
    except asyncpg.UniqueViolationError as e:
        msg = f"Topic '{topic}' already exists"
        raise TopicAlreadyExists(msg) from e


async def list_all(
    conn: asyncpg.Connection,
    topic_prefix: str | None = None,
    limit: int | None = None,
    offset: int = 0,
) -> tuple[list[TopicMeta], int]:
    """List topics sorted by most recently updated. Returns (topics, total_count).

    total_count reflects the full filtered set before LIMIT -- use for pagination.
    """
    params: list[Any] = []
    where = ""
    if topic_prefix:
        topic_prefix = validate_topic(topic_prefix)
        where = (
            f"WHERE t.path LIKE {_add_param(params, _escape_like(topic_prefix) + '%')} ESCAPE '!'"
        )

    pagination = ""
    if limit is not None:
        pagination = f"LIMIT {_add_param(params, limit)} OFFSET {_add_param(params, offset)}"

    sql = f"""
        SELECT t.id, t.path, t.title, t.description,
               t.created_at, t.updated_at,
               COUNT(e.id) AS entry_count,
               COUNT(*) OVER() AS total_count
        FROM topics t
        LEFT JOIN entries e ON e.topic_id = t.id AND e.deleted_at IS NULL
        {where}
        GROUP BY t.id
        ORDER BY t.updated_at DESC, t.id DESC
        {pagination}
    """
    rows = await conn.fetch(sql, *params)
    if rows:
        total = int(rows[0]["total_count"])
    else:
        # Empty page (e.g. offset past the end): COUNT(*) OVER() yields no row,
        # so fall back to the dedicated COUNT over the same prefix filter.
        total = await count(conn, topic_prefix=topic_prefix)
    return [_row_to_topic_meta(r) for r in rows], total


async def count(
    conn: asyncpg.Connection,
    topic_prefix: str | None = None,
) -> int:
    """Return total topic count, optionally filtered by prefix."""
    params: list[Any] = []
    where = ""
    if topic_prefix:
        topic_prefix = validate_topic(topic_prefix)
        where = (
            f"WHERE t.path LIKE {_add_param(params, _escape_like(topic_prefix) + '%')} ESCAPE '!'"
        )
    sql = f"SELECT COUNT(*) FROM topics t {where}"  # noqa: S608 -- safe: topic_prefix is validated by validate_topic(); all user values go through _add_param()
    return int(await conn.fetchval(sql, *params) or 0)


# -- Mutations -----------------------------------------------------------------
#
# Every mutation below runs as a single transaction on the user-scoped (RLS)
# connection: all rows it touches belong to the authenticated user, so RLS
# permits the cross-row moves without escalating to an admin connection. Each
# mutation also writes one audit row in the SAME transaction -- the RLS insert
# policy requires actor_type='user' and actor_id = the current user, which the
# user-scoped connection already binds, so the audit write stays in-band.


async def _require_path(conn: asyncpg.Connection, topic_id: int) -> str:
    """Return a topic's path, or raise ``TopicNotFoundError``.

    Resolves a topic id to its path under RLS. Another user's topic is
    invisible, so it raises the same not-found as a truly absent id (no
    cross-tenant existence signal).
    """
    row = await conn.fetchrow("SELECT path FROM topics WHERE id = $1", topic_id)
    if row is None:
        msg = f"Topic id {topic_id} not found"
        raise TopicNotFoundError(msg)
    return str(row["path"])


async def _move_counts(conn: asyncpg.Connection, topic_id: int) -> MoveCounts:
    """Count entries (incl. soft-deleted) and conversations under a topic."""
    entries = await conn.fetchval(
        "SELECT COUNT(*) FROM entries WHERE topic_id = $1",
        topic_id,
    )
    conversations = await conn.fetchval(
        "SELECT COUNT(*) FROM conversations WHERE topic_id = $1",
        topic_id,
    )
    return MoveCounts(entries=int(entries or 0), conversations=int(conversations or 0))


async def _reassign_rows(
    conn: asyncpg.Connection,
    source_topic_id: int,
    dest_topic_id: int,
) -> MoveCounts:
    """Move all entries + conversations from source to destination.

    Keeps the entry/conversation reassignment SQL in the topics module by
    design -- it is a topic-lifecycle operation, not an entry/conversation one.
    A ``UNIQUE (topic_id, slug)`` collision on conversations surfaces as
    ``TopicMergeConflict``.
    """
    moved_conversations = 0
    try:
        conv_result = await conn.execute(
            "UPDATE conversations SET topic_id = $1, updated_at = now() WHERE topic_id = $2",
            dest_topic_id,
            source_topic_id,
        )
    except asyncpg.UniqueViolationError as exc:
        msg = "Destination topic already holds a conversation with the same slug"
        raise TopicMergeConflict(msg) from exc
    moved_conversations = _affected_rows(conv_result)

    entry_result = await conn.execute(
        "UPDATE entries SET topic_id = $1, updated_at = now() WHERE topic_id = $2",
        dest_topic_id,
        source_topic_id,
    )
    return MoveCounts(entries=_affected_rows(entry_result), conversations=moved_conversations)


def _affected_rows(command_tag: str) -> int:
    """Parse the row count from an asyncpg ``UPDATE n`` command tag."""
    parts = command_tag.split()
    return int(parts[-1]) if parts else 0


async def _audit(
    conn: asyncpg.Connection,
    *,
    user_id: str,
    action: Action,
    topic_id: int,
    metadata: dict[str, Any],
) -> None:
    """Write one audit row for a topic mutation in the caller's transaction."""
    await record_audit(
        conn,
        actor_type="user",
        actor_id=user_id,
        action=action,
        target_kind=TargetKind.TOPIC,
        target_id=str(topic_id),
        metadata=metadata,
    )


async def rename(
    conn: asyncpg.Connection,
    topic_id: int,
    new_path: str,
    new_title: str | None = None,
    *,
    actor_id: str,
) -> TopicMeta:
    """Rename a topic's path (and optionally its title). Returns updated meta.

    Raises ``TopicNotFoundError`` when the id does not resolve for the user and
    ``TopicAlreadyExists`` when ``new_path`` collides with another of the user's
    topics (the per-user ``UNIQUE (user_id, path)`` constraint).
    """
    new_path = validate_topic(new_path)
    old_path = await _require_path(conn, topic_id)
    title_clause = ""
    params: list[Any] = [new_path]
    if new_title is not None:
        new_title = validate_title(new_title)
        title_clause = f", title = {_add_param(params, new_title)}"
    id_param = _add_param(params, topic_id)

    sql = (
        f"UPDATE topics SET path = $1{title_clause}, updated_at = now() "  # noqa: S608 -- params are bound; title_clause uses _add_param()
        f"WHERE id = {id_param} "
        "RETURNING id, path, title, description, created_at, updated_at"
    )
    try:
        row = await conn.fetchrow(sql, *params)
    except asyncpg.UniqueViolationError as exc:
        msg = f"Topic '{new_path}' already exists"
        raise TopicAlreadyExists(msg) from exc
    if row is None:
        msg = f"Topic id {topic_id} not found"
        raise TopicNotFoundError(msg)

    await _audit(
        conn,
        user_id=actor_id,
        action=Action.TOPIC_RENAMED,
        topic_id=topic_id,
        metadata={"old_path": old_path, "new_path": new_path},
    )
    return TopicMeta(
        id=row["id"],
        topic=row["path"],
        title=row["title"],
        description=row["description"] or "",
        created=row["created_at"].date().isoformat(),
        updated=row["updated_at"].date().isoformat(),
        entry_count=0,
    )


async def delete_with_reassign(
    conn: asyncpg.Connection,
    topic_id: int,
    move_entries_to: int | None,
    *,
    actor_id: str,
) -> MoveCounts:
    """Reassign the topic's entries + conversations, then delete the topic.

    ``move_entries_to`` is REQUIRED when the topic holds any entries or
    conversations; the router inspects the returned/raised counts to decide.
    When the topic is empty it is deleted directly. Raises ``TopicNotFoundError``
    for an unresolved source or destination, ``ValueError`` when a destination
    is needed but absent, and ``TopicMergeConflict`` on a slug collision.
    """
    source_path = await _require_path(conn, topic_id)
    counts = await _move_counts(conn, topic_id)

    if counts.entries == 0 and counts.conversations == 0:
        await conn.execute("DELETE FROM topics WHERE id = $1", topic_id)
        await _audit(
            conn,
            user_id=actor_id,
            action=Action.TOPIC_DELETED,
            topic_id=topic_id,
            metadata={"source_path": source_path, "entries_moved": 0, "conversations_moved": 0},
        )
        return MoveCounts(entries=0, conversations=0)

    if move_entries_to is None:
        msg = f"Topic has {counts.entries} entries that need a destination"
        raise ValueError(msg)
    if move_entries_to == topic_id:
        msg = "Cannot reassign a topic's entries to itself"
        raise ValueError(msg)

    dest_path = await _require_path(conn, move_entries_to)
    moved = await _reassign_rows(conn, topic_id, move_entries_to)
    await conn.execute("DELETE FROM topics WHERE id = $1", topic_id)
    await conn.execute(
        "UPDATE topics SET updated_at = now() WHERE id = $1",
        move_entries_to,
    )

    await record_audit(
        conn,
        actor_type="user",
        actor_id=actor_id,
        action=Action.ENTRY_MOVED,
        target_kind=TargetKind.TOPIC,
        target_id=str(move_entries_to),
        metadata={
            "source_path": source_path,
            "dest_path": dest_path,
            "entries_moved": moved.entries,
            "conversations_moved": moved.conversations,
        },
    )
    await _audit(
        conn,
        user_id=actor_id,
        action=Action.TOPIC_DELETED,
        topic_id=topic_id,
        metadata={
            "source_path": source_path,
            "dest_path": dest_path,
            "entries_moved": moved.entries,
            "conversations_moved": moved.conversations,
        },
    )
    return moved


async def merge(
    conn: asyncpg.Connection,
    source_topic_id: int,
    into_topic_id: int,
    *,
    actor_id: str,
) -> MoveCounts:
    """Move all entries + conversations from source into destination; drop source.

    Raises ``ValueError`` when source == destination, ``TopicNotFoundError``
    when either id does not resolve for the user, and ``TopicMergeConflict`` on
    a conversation slug collision.
    """
    if source_topic_id == into_topic_id:
        msg = "Cannot merge a topic into itself"
        raise ValueError(msg)

    source_path = await _require_path(conn, source_topic_id)
    dest_path = await _require_path(conn, into_topic_id)

    moved = await _reassign_rows(conn, source_topic_id, into_topic_id)
    await conn.execute("DELETE FROM topics WHERE id = $1", source_topic_id)
    await conn.execute(
        "UPDATE topics SET updated_at = now() WHERE id = $1",
        into_topic_id,
    )

    await _audit(
        conn,
        user_id=actor_id,
        action=Action.TOPIC_MERGED,
        topic_id=into_topic_id,
        metadata={
            "source_path": source_path,
            "dest_path": dest_path,
            "entries_moved": moved.entries,
            "conversations_moved": moved.conversations,
        },
    )
    return moved
