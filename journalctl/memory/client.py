"""MemoryServiceProtocol — structural interface for mcp-memory-service.

Defined here so it can be imported anywhere without pulling in the optional
mcp_memory_service package at module load time.
"""

import logging
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class MemoryServiceProtocol(Protocol):
    """Structural interface for mcp-memory-service MemoryService."""

    async def store_memory(self, content: str, **kwargs: Any) -> dict[str, Any]: ...
    async def retrieve_memories(self, query: str, **kwargs: Any) -> dict[str, Any]: ...
    async def search_by_tag(self, tags: Any, **kwargs: Any) -> dict[str, Any]: ...
    async def list_memories(self, **kwargs: Any) -> dict[str, Any]: ...
    async def delete_memory(self, content_hash: str) -> dict[str, Any]: ...
    async def health_check(self) -> dict[str, Any]: ...
    async def close(self) -> None: ...


def _get_memory_conn(memory_service: MemoryServiceProtocol) -> Any:
    """Get the underlying sqlite3 connection from the memory service, or None."""
    return getattr(getattr(memory_service, "storage", None), "conn", None)


async def hard_delete_memory(
    memory_service: MemoryServiceProtocol,
    content_hash: str,
) -> None:
    """Delete a memory and purge the soft-delete tombstone.

    mcp-memory-service delete_memory() only soft-deletes (sets deleted_at),
    leaving the row in the memories table.  A subsequent store_memory() with
    the same content_hash then fails on the UNIQUE constraint
    (upstream issue doobidoo/mcp-memory-service#644).

    This helper calls delete_memory() first (removes the embedding vector),
    then hard-deletes the tombstone row via the underlying storage connection.
    """
    await memory_service.delete_memory(content_hash=content_hash)

    conn = _get_memory_conn(memory_service)
    if conn is not None:
        conn.execute(
            "DELETE FROM memories WHERE content_hash = ? AND deleted_at IS NOT NULL",
            (content_hash,),
        )
        conn.commit()


def hard_delete_by_entry_id(
    memory_service: MemoryServiceProtocol,
    entry_id: int,
) -> int:
    """Hard-delete ALL memories whose metadata contains a given entry_id.

    Removes both the embedding vectors and the memory rows. This cleans up
    stale embeddings left behind when entry content was updated (the old
    content produced a different hash, so hard_delete_memory() can't find it).

    Returns the number of rows deleted.
    """
    conn = _get_memory_conn(memory_service)
    if conn is None:
        return 0

    # Find all memory row ids matching this entry_id in metadata JSON.
    rows = conn.execute(
        "SELECT id, content_hash FROM memories" " WHERE json_extract(metadata, '$.entry_id') = ?",
        (entry_id,),
    ).fetchall()

    if not rows:
        return 0

    ids = [r[0] for r in rows]
    placeholders = ",".join("?" * len(ids))

    # Remove embeddings first (foreign-key-like dependency on rowid).
    # placeholders is built from "?" * len(ids) — safe from injection.
    conn.execute(f"DELETE FROM memory_embeddings WHERE rowid IN ({placeholders})", ids)  # noqa: S608
    conn.execute(f"DELETE FROM memories WHERE id IN ({placeholders})", ids)  # noqa: S608
    conn.commit()
    return len(ids)
