"""Request-scoped authentication context for MCP requests.

- ``current_user_id``: the authenticated user's UUID (used by
  ``user_scoped_connection`` for RLS enforcement).
- ``current_token_scopes``: the set of OAuth scopes granted to the
  token (used by the ``@require_scope`` decorator).
"""

from __future__ import annotations

from contextvars import ContextVar
from uuid import UUID

current_user_id: ContextVar[UUID | None] = ContextVar("current_user_id", default=None)

# Token scopes as a frozenset, set by BearerAuthMiddleware after
# successful introspection.  None means the token has not been
# introspected (e.g. API-key auth path).
current_token_scopes: ContextVar[frozenset[str] | None] = ContextVar(
    "current_token_scopes", default=None
)


class AuthenticationError(RuntimeError):
    """Raised when code requires an authenticated user but none is set."""


def get_current_user_id() -> UUID:
    """Return the current user's ID, raising AuthenticationError if unset.

    Reads the ``current_user_id`` ContextVar.  Callers that need the ID
    *before* acquiring a database connection should use this helper rather
    than reading the ContextVar directly, because the context may not have
    been initialised yet (e.g. outside a request pipeline).
    """
    value = current_user_id.get()
    if value is None:
        raise AuthenticationError("No authenticated user in request context")
    if not isinstance(value, UUID):
        raise AuthenticationError(f"current_user_id is not a UUID: {type(value).__qualname__}")
    return value
