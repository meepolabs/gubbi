"""Search read endpoint: hybrid FTS + semantic journal search.

``GET /api/v1/search`` -- rank-merged search across entries and conversations.

Reuses the shared :func:`gubbi.services.search.run_journal_search` pipeline (the
same one the ``journal_search`` MCP tool calls), so ranking, semantic-degrade,
and decryption semantics are identical. The web surface raises the result cap to
50 (the tool clamps to 20) by passing its own ``limit`` to the service.

Results carry decrypted content, so the response is ``private, no-store``. A row
that fails to decrypt surfaces the ``[decryption failed]`` sentinel and a
per-result ``decryption_failed: true`` flag rather than failing the response.
There is no ``offset`` in v1: the result set is rank-merged; "load more" raises
``limit`` up to the cap.
"""

from __future__ import annotations

from typing import Annotated, Literal
from uuid import UUID  # noqa: TC003

import structlog
from fastapi import APIRouter, Depends, Query, Request, Response
from gubbi_common.telemetry import bound_logger
from pydantic import BaseModel, Field

from gubbi.api.v1.auth import require_scope
from gubbi.api.v1.web.responses import private_no_store_response
from gubbi.app_state import require_app_ctx
from gubbi.crypto.guard import require_cipher
from gubbi.services.search import encode_query, run_journal_search
from gubbi.storage.connection import safe_user_scoped_connection
from gubbi.validation import validate_date, validate_topic

__all__: list[str] = [
    "DEFAULT_SEARCH_LIMIT",
    "MAX_QUERY_LEN",
    "MAX_SEARCH_LIMIT",
    "ConversationSearchResult",
    "EntrySearchResult",
    "SearchResponse",
    "router",
    "search",
]

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/search", tags=["search"])

# Web-surface search contract (raised above the tool-layer cap of 20 per spec).
DEFAULT_SEARCH_LIMIT: int = 10
MAX_SEARCH_LIMIT: int = 50

# Query input guard -- mirrors the tool-layer cap so both surfaces reject the
# same oversized query, here as a 422 before the handler runs.
MAX_QUERY_LEN: int = 2000

QueryParam = Annotated[
    str,
    Query(min_length=1, max_length=MAX_QUERY_LEN, description="Search query (max 2000 chars)."),
]
LimitParam = Annotated[
    int, Query(ge=1, le=MAX_SEARCH_LIMIT, description="Maximum results (max 50).")
]


class EntrySearchResult(BaseModel):
    """A search hit backed by a journal entry."""

    doc_type: Literal["entry"] = "entry"
    topic: str
    date: str
    entry_id: int | None
    conversation_id: None = None
    content: str
    decryption_failed: bool = False


class ConversationSearchResult(BaseModel):
    """A search hit backed by a saved conversation."""

    doc_type: Literal["conversation"] = "conversation"
    topic: str
    date: str
    entry_id: None = None
    conversation_id: int | None
    title: str
    summary: str


class SearchResponse(BaseModel):
    """``GET /api/v1/search`` body: ``{results, total, query}``."""

    results: list[
        Annotated[
            EntrySearchResult | ConversationSearchResult,
            Field(discriminator="doc_type"),
        ]
    ]
    total: int
    query: str


@router.get("", response_model=SearchResponse)
async def search(
    request: Request,
    auth: Annotated[tuple[UUID, frozenset[str]], Depends(require_scope("journal:read"))],
    q: QueryParam,
    topic_prefix: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    limit: LimitParam = DEFAULT_SEARCH_LIMIT,
) -> Response:
    """GET /api/v1/search.

    Hybrid search across the authenticated user's entries and conversations,
    rank-merged (FTS first, semantic second, dedup). ``total`` is the size of
    the returned page (no separate unbounded count; the set is capped at
    ``limit``). Invalid ``topic_prefix`` / date syntax maps to an empty result
    set rather than a 422 -- a malformed filter cannot match anything.

    Cache-Control is ``private, no-store``: results carry decrypted content.
    """
    user_id, _scopes = auth
    app_ctx = require_app_ctx(request)
    log = bound_logger(request)

    normalized_prefix = topic_prefix.rstrip("/") if topic_prefix else None
    if normalized_prefix:
        try:
            normalized_prefix = validate_topic(normalized_prefix)
        except ValueError:
            return private_no_store_response(SearchResponse(results=[], total=0, query=q))
    else:
        normalized_prefix = None

    for value in (date_from, date_to):
        if value:
            try:
                validate_date(value)
            except ValueError:
                return private_no_store_response(SearchResponse(results=[], total=0, query=q))

    cipher = require_cipher(app_ctx)

    # Encode the query before acquiring the connection so the CPU-bound encode
    # does not run while holding a user-scoped transaction.
    query_embedding = await encode_query(app_ctx, q)

    async with safe_user_scoped_connection(app_ctx.pool, user_id=user_id) as conn:
        result = await run_journal_search(
            conn,
            cipher,
            app_ctx,
            query=q,
            topic_prefix=normalized_prefix,
            date_from=date_from,
            date_to=date_to,
            limit=limit,
            query_embedding=query_embedding,
        )

    await log.info("web_search", result_count=result["total"])
    body = SearchResponse.model_validate(result)
    return private_no_store_response(body)
