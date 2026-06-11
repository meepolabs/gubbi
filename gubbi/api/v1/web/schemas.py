"""Shared Pydantic response patterns for the web REST endpoints.

Two reusable shapes that every resource router composes:

* :class:`PaginatedList` -- generic ``(items, total, limit, offset)`` envelope.
  Per-resource responses subclass it with a concrete item type and rename the
  ``items`` field to the resource noun via an alias (e.g. ``topics``) so the
  JSON matches the spec while the pagination metadata stays uniform.
* :class:`DecryptableItem` -- mixin contributing the ``decryption_failed``
  flag that any item carrying decrypted content must expose. Pair it with
  :func:`gubbi.api.v1.web.decryption.decrypt_field` so a single bad row yields
  the sentinel + ``decryption_failed: true`` instead of a 500.
"""

from __future__ import annotations

from typing import Generic, TypeVar

from pydantic import BaseModel

__all__: list[str] = [
    "DecryptableItem",
    "PaginatedList",
]

ItemT = TypeVar("ItemT", bound=BaseModel)


class PaginatedList(BaseModel, Generic[ItemT]):
    """Generic offset-paginated list envelope.

    ``total`` is the full filtered count before ``LIMIT`` (the repo functions
    return it via ``COUNT(*) OVER()``); ``limit`` / ``offset`` echo the request
    so the client can page without re-deriving them.

    Per-resource responses subclass this and alias ``items`` to the resource
    noun the spec uses::

        class TopicListResponse(PaginatedList[TopicItem]):
            items: list[TopicItem] = Field(serialization_alias="topics")

    Serializing with ``by_alias=True`` then emits ``{"topics": [...],
    "total": ..., "limit": ..., "offset": ...}``.
    """

    items: list[ItemT]
    total: int
    limit: int
    offset: int


class DecryptableItem(BaseModel):
    """Mixin for list/detail items that carry decrypted content.

    ``decryption_failed`` is ``False`` on the happy path and ``True`` when the
    content field could not be decrypted -- in which case the content field
    holds the ``[decryption failed]`` sentinel rather than plaintext. One bad
    row never fails the whole response.
    """

    decryption_failed: bool = False
