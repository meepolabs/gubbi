"""Error mapping for the web REST endpoints.

Maps storage-layer "not found" conditions to FastAPI's default error envelope
(``{"detail": "<code>"}``) -- no custom envelope. Centralized so every router
raises the same ``detail`` code for the same condition.
"""

from __future__ import annotations

from fastapi import HTTPException

__all__: list[str] = [
    "TOPIC_NOT_FOUND",
    "topic_not_found",
]

# Stable machine-readable detail codes. Clients match on these, not on prose.
TOPIC_NOT_FOUND: str = "topic_not_found"


def topic_not_found() -> HTTPException:
    """Return a 404 ``HTTPException`` with ``{"detail": "topic_not_found"}``.

    Raised when a topic path or id does not resolve for the authenticated user.
    Under RLS another user's topic is indistinguishable from a missing one --
    which is the intended behavior (no cross-tenant existence signal).
    """
    return HTTPException(status_code=404, detail=TOPIC_NOT_FOUND)
