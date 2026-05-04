"""OAuth scope checking infrastructure for MCP tools.

Provides:
- Data-driven scope mapping (v1: ``journal`` grants both read and write)
- ``@require_scope`` decorator for per-tool scope enforcement
- Human-readable scope descriptions for consent UI
- Insufficient-scope error builder (OpenAI ``tools/call`` format)

Future-split: when Hydra issues ``journal:read`` and ``journal:write``
separately, update the mapping -- the decorator and middleware do not
change.

The ``mcp.types`` import is a deliberate cross-layer coupling: the
insufficient-scope error builder must return the MCP transport shape
(``CallToolResult`` with ``_meta`` headers per the MCP spec) so
``@require_scope`` can short-circuit to a valid tool result without the
caller assembling the envelope.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Collection
from typing import Any

from mcp.types import CallToolResult, TextContent

from gubbi.core.auth_context import current_token_scopes

# ---------------------------------------------------------------------------
# Scope descriptions for consent UI
# ---------------------------------------------------------------------------

SCOPE_DESCRIPTIONS: dict[str, str] = {
    "journal": "Access your journal: read, write, search, and manage topics and entries.",
    "openid": "Identify you via OpenID Connect.",
    "email": "Access your email address.",
    "offline_access": "Request a refresh token for long-lived access.",
}

# ---------------------------------------------------------------------------
# Data-driven scope mapping
# ---------------------------------------------------------------------------
# Maps actual Hydra-issued scopes to the "permissions" they grant.
#
# v1: Hydra issues the single ``journal`` scope. The mapping treats it as
# granting both ``journal:read`` and ``journal:write`` so the per-tool
# @require_scope("journal:read") / @require_scope("journal:write")
# decorators work immediately.
#
# Future (split): Hydra issues ``journal:read`` and ``journal:write``. The
# mapping is updated so each scope grants only its own permission. The
# decorator code and middleware do not change.

SCOPE_GRANTS: dict[str, frozenset[str]] = {
    "journal": frozenset({"journal:read", "journal:write"}),
    "journal:read": frozenset({"journal:read"}),
    "journal:write": frozenset({"journal:write"}),
}

# ---------------------------------------------------------------------------
# Precomputed permission resolution (compiled once at import)
# ---------------------------------------------------------------------------

# Merges every grant set into one membership -- short-circuits mapped match
# when the required scope is ungrantable by any known scope.
_GRANTS_UNION: frozenset[str] = frozenset(
    perm for grants in SCOPE_GRANTS.values() for perm in grants
)

# Inverse index: permission -> set-of-scopes that can grant it.  Replaces
# the O(n) per-call loop over token scopes with a single intersection test.
_GRANT_INVERSE: dict[str, frozenset[str]] = {
    perm: frozenset(scope for scope, grants in SCOPE_GRANTS.items() if perm in grants)
    for perm in _GRANTS_UNION
}


# ---------------------------------------------------------------------------
# Scope checker
# ---------------------------------------------------------------------------


def check_scope(token_scopes: Collection[str], required_scope: str) -> bool:
    """Return True if the token's scopes grant the required permission.

    Uses the ``SCOPE_GRANTS`` mapping so the check is data-driven, not
    hardcoded. Adding a new scope only requires updating the mapping.

    The check is two-stage:
    1. Direct match -- if ``required_scope`` is directly in ``token_scopes``.
    2. Mapped match -- if any token scope grants ``required_scope`` via
       ``SCOPE_GRANTS``.

    In v1 the Hydra-issued ``journal`` scope directly matches the
    middleware's ``required_scope="journal"`` (v1).  The per-tool
    decorator checks ``"journal:read"`` / ``"journal:write"`` which are
    granted via the mapping (stage 2).

    Parameters
    ----------
    token_scopes :
        The set of scopes present on the validated token (from Hydra
        introspection or self-host OAuth).
    required_scope :
        The scope the tool or endpoint requires (e.g. ``"journal:read"``).

    Returns
    -------
    bool
        True if the token's scopes grant the required permission.

    Examples
    --------
    >>> check_scope({"journal"}, "journal:read")
    True
    >>> check_scope({"openid", "email"}, "journal:read")
    False
    >>> check_scope({"journal"}, "journal")
    True
    """
    # Stage 1: direct match (v1: "journal" token scope matches "journal"
    # middleware required_scope)
    if required_scope in token_scopes:
        return True

    # Stage 2: mapped permission -- use the precomputed inverse index to avoid
    # iterating over every token scope.  _GRANT_INVERSE[perm] is O(1), so the
    # intersection test is O(k) where k is the size of the grant set for this
    # specific required_scope (tiny, typically 1).
    return bool(_GRANTS_UNION & {required_scope}) and bool(
        _GRANT_INVERSE.get(required_scope, frozenset()) & set(token_scopes)
    )


# ---------------------------------------------------------------------------
# Insufficient-scope error builder (OpenAI-required format)
# ---------------------------------------------------------------------------


def insufficient_scope_response(
    required_scope: str,
    detail: str = "",
) -> CallToolResult:
    """Build an MCP ``tools/call`` error result for insufficient scope.

    The ``_meta`` field carries the ``www_authenticate`` value per the MCP
    spec. When present, ChatGPT triggers its tool-level OAuth re-consent UI.
    Anthropic ignores it (harmless).
    """
    description = detail or f"Token scope does not include {required_scope}"
    return CallToolResult(
        isError=True,
        content=[
            TextContent(
                type="text",
                text=f"insufficient_scope: {description}",
            ),
        ],
        _meta={
            "mcp/www_authenticate": (
                f'Bearer error="insufficient_scope", error_description="{description}"'
            ),
        },
    )


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------


def require_scope(scope: str) -> Callable[..., Callable[..., Any]]:
    """Decorator that enforces a scope check before the tool handler runs.

    Reads ``current_token_scopes`` (set by ``BearerAuthMiddleware`` after
    token validation).  If the check fails, returns an MCP error result with
    ``isError=True`` and the OpenAI-required ``_meta`` header.

    In v1 the middleware already rejects tokens without ``journal`` at the
    HTTP layer, so this decorator is defense-in-depth.  After the
    read/write split it becomes the primary enforcement point.

    Usage
    -----
    .. code-block:: python

        @mcp.tool()
        @require_scope("journal:read")
        async def journal_search(...):
            ...
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            token_scopes: Collection[str] = current_token_scopes.get() or frozenset()
            if not check_scope(token_scopes, scope):
                return insufficient_scope_response(scope)
            return await fn(*args, **kwargs)

        return wrapper

    return decorator
