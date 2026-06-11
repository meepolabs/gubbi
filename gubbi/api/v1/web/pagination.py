"""Pagination query-param helpers for the web REST endpoints.

The web surface uses higher per-page caps than the MCP tool layer (the spec
raises them explicitly per endpoint -- e.g. topics cap 200 vs the tool-layer
20), so each router declares its own ``limit`` cap while sharing the uniform
``offset`` param here. Over-max ``limit`` is rejected with HTTP 422 by
Pydantic's ``Query`` constraint -- no manual clamping, no silent truncation.

``limit`` varies per resource, so each router inlines its own annotated alias
(a module-level constant keeps the cap a single named value)::

    from fastapi import Query

    DEFAULT_LIMIT = 50
    MAX_LIMIT = 200
    LimitParam = Annotated[int, Query(ge=1, le=MAX_LIMIT)]

    @router.get("")
    async def list_x(limit: LimitParam = DEFAULT_LIMIT, offset: OffsetQuery = 0): ...

``offset`` is uniform, so import :data:`OffsetQuery` directly.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Query

__all__: list[str] = [
    "OffsetQuery",
]

# ``offset`` is uniform across resources: non-negative, default 0.
OffsetQuery = Annotated[int, Query(ge=0, description="Number of items to skip.")]
