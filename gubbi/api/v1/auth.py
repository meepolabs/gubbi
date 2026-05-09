"""Shared FastAPI authentication dependencies for REST API routes."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from uuid import UUID

from fastapi import HTTPException, Request

from gubbi.auth.scope import SCOPE_GRANTS, check_scope
from gubbi.auth.strategies import AuthRejected, AuthResult, AuthStrategy

INVALID_TOKEN_MESSAGE: str = "Invalid or expired token"  # noqa: S105


def _has_scope(token_scopes: frozenset[str], required_scope: str) -> bool:
    """Return True if token_scopes satisfy required_scope."""
    if check_scope(set(token_scopes), required_scope):
        return True
    expanded = SCOPE_GRANTS.get(required_scope)
    return expanded is not None and expanded.issubset(token_scopes)


async def resolve_user_id(
    request: Request,
    scope: str | None = None,
) -> tuple[UUID, frozenset[str]]:
    """Authenticate request and return (user_id, granted_scopes)."""
    strategies: list[AuthStrategy] = request.app.state.auth_strategies  # type: ignore[attr-defined]
    result: AuthResult | None = None
    for strategy in strategies:
        try:
            result = await strategy.authenticate(request)
        except AuthRejected as exc:
            raise HTTPException(status_code=exc.status, detail=exc.detail) from None
        if result is not None:
            break
    if result is None:
        raise HTTPException(status_code=401, detail=INVALID_TOKEN_MESSAGE)
    if scope is not None and not _has_scope(result.scopes, scope):
        raise HTTPException(status_code=403, detail="insufficient_scope")
    return (result.user_id, result.scopes)


def require_scope(
    scope: str | None = None,
) -> Callable[[Request], Awaitable[tuple[UUID, frozenset[str]]]]:
    """FastAPI dependency factory that enforces a required OAuth scope.

    Usage::

        @router.get("/endpoint")
        async def endpoint(
            auth: Annotated[tuple[UUID, frozenset[str]], Depends(require_scope("journal:write"))],
        ):
            user_id, scopes = auth
    """

    async def _dep(request: Request) -> tuple[UUID, frozenset[str]]:
        return await resolve_user_id(request, scope=scope)

    return _dep
