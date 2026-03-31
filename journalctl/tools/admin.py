"""MCP tool: journal_reindex."""

import asyncio
import hashlib
import json
import logging
import sqlite3

from mcp.server.fastmcp import FastMCP

from journalctl.memory.client import MemoryServiceProtocol
from journalctl.storage.database import DatabaseStorage
from journalctl.storage.search_index import SearchIndex
from journalctl.tools.constants import REINDEX_BATCH_SIZE

logger = logging.getLogger(__name__)


def register(
    mcp: FastMCP,
    index: SearchIndex,
    storage: DatabaseStorage,
    memory_service: MemoryServiceProtocol,
) -> None:
    """Register admin tools on the MCP server."""

    _reindex_lock = asyncio.Lock()

    @mcp.tool()
    async def journal_reindex() -> dict:
        """Repair the search index — use when journal_search returns wrong or missing results.

        Only needed if search results seem stale or incomplete.
        Rarely needed during normal use.

        Returns:
            Number of documents indexed, embeddings generated, and duration.
        """
        if _reindex_lock.locked():
            return {
                "status": "already_running",
                "message": "A reindex is already in progress.",
            }

        async with _reindex_lock:
            result = index.rebuild_from_db(storage)

            # Rebuild semantic embeddings for entries whose indexed_at is stale or absent.
            # Processes in batches using an id-cursor so we advance even when individual
            # embeddings fail (preventing infinite retries within a run).
            # The indexed_at watermark makes this resumable across invocations.
            embeddings_generated = 0
            last_id = 0
            semantic_status = "ok"

            try:
                while True:
                    batch = storage.get_unindexed_entries(last_id, REINDEX_BATCH_SIZE)

                    if not batch:
                        break

                    for r in batch:
                        try:
                            # Remove stale embedding before storing updated one.
                            # Hash must match mcp-memory-service's generate_content_hash():
                            # sha256 of content.strip().lower().encode('utf-8').
                            old_content = storage.get_entry_content(r["id"])
                            if old_content:
                                try:
                                    old_hash = hashlib.sha256(
                                        old_content.strip().lower().encode("utf-8")
                                    ).hexdigest()
                                    await memory_service.delete_memory(content_hash=old_hash)
                                except Exception:  # noqa: S110
                                    pass

                            first_line = (r["content"] or "").split("\n", 1)[0][:80]
                            await memory_service.store_memory(
                                content=r["content"],
                                tags=json.loads(r["tags"] or "[]"),
                                metadata={
                                    "entry_id": r["id"],
                                    "source": "journal_entry",
                                    "topic": r["topic"],
                                    "date": r["date"],
                                    "title": first_line,
                                },
                            )
                            storage.mark_entry_indexed(r["id"])
                            embeddings_generated += 1
                        except Exception as e:
                            logger.warning(
                                "Failed to embed entry %s during reindex: %s",
                                r["id"],
                                e,
                                exc_info=True,
                            )

                    last_id = batch[-1]["id"]  # advance cursor past all processed entries

            except (sqlite3.Error, json.JSONDecodeError, OSError):
                logger.warning("Semantic reindex failed", exc_info=True)
                semantic_status = "failed"

            return {
                "status": "rebuilt",
                "semantic_status": semantic_status,
                "documents_indexed": result["documents_indexed"],
                "embeddings_generated": embeddings_generated,
                "duration_seconds": result["duration_seconds"],
            }
