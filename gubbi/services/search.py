"""Hybrid journal-search pipeline shared by the MCP tool and the web REST API.

The pipeline is: encode the query, run FTS + semantic (pgvector) backends,
merge with FTS-first dedup, batch-decrypt (hydrate) the surviving rows, then
sort by rank and slice to ``limit``. The semantic backend is floored at
``SEMANTIC_MIN_SIMILARITY`` so a query matching nothing returns nothing rather
than the nearest rows in the corpus. Both the ``journal_search`` MCP tool and
``GET /api/v1/search`` call :func:`run_journal_search` so the ranking,
degradation, and decryption semantics are identical across surfaces.

Each caller owns its own input validation and ``limit`` cap (the tool clamps to
20, the web surface to 50) and passes the already-validated, already-acquired
user-scoped ``conn`` plus the resolved ``cipher``. The service returns the
payload dict ``{"results": [...], "total": N, "query": str}``; callers adapt
that to their own response contract.

Decryption-failed rows surface the ``[decryption failed]`` (space) sentinel and
a per-result ``decryption_failed: true`` flag -- the same sentinel the web
decryption helper uses -- so the contract is uniform regardless of caller.
"""

from __future__ import annotations

import asyncio
from datetime import date as date_cls
from typing import TYPE_CHECKING, Any

import asyncpg
import structlog

from gubbi.models.search import SearchResult
from gubbi.storage.repositories import conversations as conv_repo
from gubbi.storage.repositories import entries as entry_repo
from gubbi.storage.repositories import search as search_repo
from gubbi.tools.constants import MAX_SEARCH_CONTENT_CHARS

if TYPE_CHECKING:
    from gubbi.app_context import AppContext
    from gubbi.crypto.cipher import ContentCipher

__all__: list[str] = [
    "DECRYPTION_FAILED_SENTINEL",
    "REPO_DECRYPTION_FAILED_SENTINEL",
    "SEMANTIC_MIN_SIMILARITY",
    "encode_query",
    "run_journal_search",
]

logger = structlog.get_logger(__name__)

# Cosine-similarity floor for the semantic backend. Without it the k-NN is
# unconditional -- cosine distance totally orders every stored row, so a query
# resembling nothing in the corpus still returns the ``limit`` nearest entries,
# and the caller (often an AI client) treats them as retrieved fact.
#
# Tuned for all-MiniLM-L6-v2 mean-pooled + L2-normalised embeddings, where
# cosine similarity equals the dot product (see EmbeddingService.encode).
# Measured against this corpus shape: unrelated-token queries land at 0.00-0.16,
# a partial topic-slug match at ~0.28, and genuine matches (exact term, or a
# paraphrase like "running plan for a race" vs "marathon training schedule") at
# 0.47-0.80. 0.35 sits in the empty band between the two clusters, so it drops
# noise without costing paraphrase recall -- which is the whole point of having
# a semantic backend alongside FTS.
SEMANTIC_MIN_SIMILARITY: float = 0.35

# The repository decryption path stores this hyphen sentinel as the entry
# "text" when a row cannot be decrypted.
REPO_DECRYPTION_FAILED_SENTINEL: str = "[decryption-failed]"

# Surfaced to clients in place of plaintext, paired with the per-result
# ``decryption_failed`` flag. Matches the web decryption helper's sentinel so
# the contract is uniform across the tool and REST surfaces.
DECRYPTION_FAILED_SENTINEL: str = "[decryption failed]"


def _truncate_text(value: str) -> str:
    return value[:MAX_SEARCH_CONTENT_CHARS]


def _truncate_title_summary(title: str, summary: str) -> tuple[str, str]:
    budget = MAX_SEARCH_CONTENT_CHARS
    if len(title) + len(summary) <= budget:
        return title, summary
    if len(title) >= budget:
        return title[:budget], ""
    remaining = budget - len(title)
    return title, summary[:remaining]


async def _run_dual_search(
    conn: asyncpg.Connection,
    app_ctx: AppContext,
    query: str,
    query_embedding: list[float] | None,
    topic_prefix: str | None,
    date_from: str | None,
    date_to: str | None,
    df: date_cls | None,
    dt: date_cls | None,
    limit: int,
) -> list[SearchResult]:
    """Run FTS + semantic search backends and merge with dedup."""
    fts_results: list[SearchResult] = await search_repo.fts_search(
        conn, query, topic_prefix, date_from, date_to, limit
    )

    semantic_results: list[SearchResult] = []
    if query_embedding is not None:
        try:
            raw = await app_ctx.embedding_service.search_by_vector(
                conn,
                query_embedding,
                limit=limit,
                topic_prefix=topic_prefix,
                date_from=df,
                date_to=dt,
                min_similarity=SEMANTIC_MIN_SIMILARITY,
            )
            semantic_results = [
                SearchResult(
                    source_key=f"entry:{r.get('entry_id')}",
                    doc_type="entry",
                    topic=str(r.get("topic", "")),
                    rank=-float(r.get("similarity", 0.0)),
                    date=str(r.get("date", "")),
                    entry_id=r.get("entry_id"),
                    conversation_id=None,
                )
                for r in raw
                if r.get("entry_id") is not None
            ]
        except asyncpg.PostgresError:
            await logger.warning("Semantic search failed, using FTS only", exc_info=True)
        except Exception:
            await logger.exception("Semantic search failed unexpectedly")
            raise

    # Merge with seen_keys dedup -- FTS first, semantic second preserves order.
    seen_keys: set[str] = set()
    merged: list[SearchResult] = []
    for result in fts_results:
        if result.source_key not in seen_keys:
            seen_keys.add(result.source_key)
            merged.append(result)
    for result in semantic_results:
        if result.source_key not in seen_keys:
            seen_keys.add(result.source_key)
            merged.append(result)

    return merged


async def _hydrate_results(
    conn: asyncpg.Connection,
    cipher: ContentCipher,
    merged: list[SearchResult],
) -> list[SearchResult]:
    """Decrypt+truncation hydration for merged search results."""
    # Batch-collect unique IDs for a single round-trip per entity type.
    entry_ids: set[int] = set()
    conv_ids: set[int] = set()
    for result in merged:
        if result.doc_type == "entry" and result.entry_id is not None:
            entry_ids.add(result.entry_id)
        elif result.doc_type == "conversation" and result.conversation_id is not None:
            conv_ids.add(result.conversation_id)

    # Batched fetches -- one query per entity type instead of N.
    entry_id_list = list(entry_ids)
    conv_id_list = list(conv_ids)

    decrypted_entries: dict[int, tuple[str, str | None]] = {}
    entries_batch_failed = False
    try:
        decrypted_entries = await entry_repo.get_texts(conn, cipher, entry_id_list)
    except asyncpg.PostgresError:
        entries_batch_failed = True
        await logger.exception(
            "Entry batch query failed, surfacing entries as decryption-failed",
            entry_count=len(entry_id_list),
        )

    decrypted_convs: dict[int, tuple[str, str]] = {}
    convs_batch_failed = False
    try:
        decrypted_convs = await conv_repo.get_titles_summaries(conn, cipher, conv_id_list)
    except asyncpg.PostgresError:
        convs_batch_failed = True
        await logger.exception(
            "Conversation batch query failed, surfacing conversations as decryption-failed",
            conv_count=len(conv_id_list),
        )

    hydrated: list[SearchResult] = []
    for result in merged:
        if result.doc_type == "entry" and result.entry_id is not None:
            hydrated.append(
                _hydrate_entry(result, decrypted_entries, batch_failed=entries_batch_failed)
            )
        elif result.doc_type == "conversation" and result.conversation_id is not None:
            hydrated.append(
                _hydrate_conversation(result, decrypted_convs, batch_failed=convs_batch_failed)
            )

    return hydrated


def _hydrate_entry(
    result: SearchResult,
    decrypted_entries: dict[int, tuple[str, str | None]],
    *,
    batch_failed: bool,
) -> SearchResult:
    """Map one entry hit to its hydrated form.

    On a batch-fetch failure (or a per-row decryption sentinel) the hit is kept
    with the client-facing sentinel and ``decryption_failed=True`` -- never
    dropped, so the result count stays honest.
    """
    assert result.entry_id is not None  # noqa: S101 - narrowed by caller
    decrypted = decrypted_entries.get(result.entry_id)
    if batch_failed or decrypted is None:
        return result.model_copy(
            update={"content": DECRYPTION_FAILED_SENTINEL, "decryption_failed": True}
        )
    content, _reasoning = decrypted
    decryption_failed = content == REPO_DECRYPTION_FAILED_SENTINEL
    return result.model_copy(
        update={
            "content": (
                DECRYPTION_FAILED_SENTINEL if decryption_failed else _truncate_text(content)
            ),
            "decryption_failed": decryption_failed,
        }
    )


def _hydrate_conversation(
    result: SearchResult,
    decrypted_convs: dict[int, tuple[str, str]],
    *,
    batch_failed: bool,
) -> SearchResult:
    """Map one conversation hit to its hydrated form.

    On a batch-fetch failure the hit is kept with the sentinel and
    ``decryption_failed=True`` rather than dropped.
    """
    assert result.conversation_id is not None  # noqa: S101 - narrowed by caller
    decrypted = decrypted_convs.get(result.conversation_id)
    if batch_failed or decrypted is None:
        return result.model_copy(
            update={
                "title": DECRYPTION_FAILED_SENTINEL,
                "summary": DECRYPTION_FAILED_SENTINEL,
                "decryption_failed": True,
            }
        )
    title, summary = decrypted
    truncated_title, truncated_summary = _truncate_title_summary(title, summary)
    return result.model_copy(update={"title": truncated_title, "summary": truncated_summary})


def _build_payload(hydrated: list[SearchResult], query: str, limit: int) -> dict[str, Any]:
    """Sort, slice, and shape hydrated results into the response dict."""
    sorted_results = sorted(hydrated, key=lambda x: x.rank)[:limit]

    payload: list[dict[str, Any]] = []
    for result in sorted_results:
        if result.doc_type == "entry":
            payload.append(
                {
                    "doc_type": "entry",
                    "topic": result.topic,
                    "date": result.date,
                    "entry_id": result.entry_id,
                    "conversation_id": None,
                    "content": result.content or "",
                    "decryption_failed": result.decryption_failed,
                }
            )
        elif result.doc_type == "conversation":
            payload.append(
                {
                    "doc_type": "conversation",
                    "topic": result.topic,
                    "date": result.date,
                    "entry_id": None,
                    "conversation_id": result.conversation_id,
                    "title": result.title or "",
                    "summary": result.summary or "",
                    "decryption_failed": result.decryption_failed,
                }
            )

    return {
        "results": payload,
        "total": len(payload),
        "query": query,
    }


async def encode_query(app_ctx: AppContext, query: str) -> list[float] | None:
    """Encode a query string to an embedding vector off the event loop.

    Runs the CPU-bound encode in a worker thread. Returns ``None`` (and logs a
    warning) when encoding fails so the caller degrades to FTS-only search.

    Callers should invoke this BEFORE acquiring a user-scoped DB connection so
    the encode does not run while holding a transaction.
    """
    try:
        return await asyncio.to_thread(app_ctx.embedding_service.encode, query)
    except Exception:
        await logger.warning("Query encoding failed, semantic search disabled", exc_info=True)
        return None


# Sentinel distinguishing "caller did not precompute the embedding" from "caller
# precomputed it as None because encoding failed". When the embedding is not
# provided, ``run_journal_search`` encodes it itself (back-compat path).
_ENCODE_INTERNALLY: object = object()


async def run_journal_search(
    conn: asyncpg.Connection,
    cipher: ContentCipher,
    app_ctx: AppContext,
    *,
    query: str,
    topic_prefix: str | None,
    date_from: str | None,
    date_to: str | None,
    limit: int,
    query_embedding: list[float] | None | object = _ENCODE_INTERNALLY,
) -> dict[str, Any]:
    """Run the hybrid journal-search pipeline and return the payload dict.

    Runs the FTS + semantic backends, merges FTS-first with dedup, hydrates
    (batch-decrypts) the surviving rows, then sorts by rank and slices to
    ``limit``.

    ``query_embedding`` should be precomputed by the caller via
    :func:`encode_query` BEFORE acquiring ``conn`` so the CPU-bound encode does
    not run while holding a transaction. When omitted, the embedding is encoded
    internally (back-compat); semantic search degrades to FTS-only if encoding
    failed (embedding is ``None``).

    Inputs must already be validated by the caller (``query`` length,
    ``topic_prefix`` / date syntax) and ``conn`` must be a user-scoped
    connection for the authenticated user. The caller also owns the ``limit``
    cap (the tool clamps to 20, the web surface to 50).

    Returns:
        ``{"results": [...], "total": N, "query": query}`` -- the same shape the
        ``journal_search`` tool returns. Entry results carry ``content`` +
        ``decryption_failed``; conversation results carry ``title`` +
        ``summary``.
    """
    embedding: list[float] | None
    if query_embedding is _ENCODE_INTERNALLY:
        embedding = await encode_query(app_ctx, query)
    else:
        embedding = query_embedding  # type: ignore[assignment]

    df = date_cls.fromisoformat(date_from) if date_from else None
    dt = date_cls.fromisoformat(date_to) if date_to else None

    merged = await _run_dual_search(
        conn,
        app_ctx,
        query,
        embedding,
        topic_prefix,
        date_from,
        date_to,
        df,
        dt,
        limit,
    )
    hydrated = await _hydrate_results(conn, cipher, merged)

    return _build_payload(hydrated, query, limit)
