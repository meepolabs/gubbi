"""@audited decorator for MCP tool handlers (M3 requirement).

Records an audit event after a write tool handler completes successfully.
Best-effort: audit write failures are logged and counted but never
propagated to the caller, so the user always gets their result.

Stacking order (outermost to innermost)::

    @mcp.tool()
    @require_scope("journal:write")
    @audited("entry.created", target_type="entry", app_ctx=app_ctx)
    async def journal_append_entry(...) -> dict:
        ...

The decorator runs the handler first, then — only on success (no ``"error"``
key in the returned dict) — opens its own database connection and appends
an audit row.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Awaitable, Callable
from typing import Any, ParamSpec, TypeVar

from gubbi_common.db.user_scoped import user_scoped_connection

from journalctl.audit import record_audit
from journalctl.core.auth_context import current_user_id
from journalctl.core.context import AppContext
from journalctl.telemetry.metrics import record_audit_persistence_failure

logger = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R", bound=dict[str, Any])

ACTION_ENTRY_CREATED = "entry.created"
ACTION_ENTRY_UPDATED = "entry.updated"
ACTION_ENTRY_DELETED = "entry.deleted"
ACTION_TOPIC_CREATED = "topic.created"
ACTION_CONVERSATION_SAVED = "conversation.saved"


def audited(
    action: str,
    target_type: str,
    app_ctx: AppContext,
) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]:
    """Decorate an MCP write-tool handler to record an audit event on success.

    Parameters
    ----------
    action :
        Audit action string (e.g. ``"entry.created"``).
    target_type :
        Type of the target resource (e.g. ``"entry"``, ``"topic"``).
    app_ctx :
        Application context (provides the database pool).
    """

    def decorator(fn: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
        @functools.wraps(fn)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            result = await fn(*args, **kwargs)

            # Only audit successful operations (no "error" key in result dict)
            if "error" not in result:
                user_id = current_user_id.get()
                if user_id is not None:
                    target_id = _extract_target_id(result)
                    try:
                        async with user_scoped_connection(app_ctx.pool, user_id=user_id) as conn:
                            await record_audit(
                                conn,
                                actor_type="user",
                                actor_id=str(user_id),
                                action=action,
                                target_type=target_type,
                                target_id=target_id,
                            )
                    except Exception:
                        logger.warning(
                            "Audit write failed for %s (action=%s, target_type=%s)",
                            fn.__name__,
                            action,
                            target_type,
                            exc_info=True,
                        )
                        record_audit_persistence_failure(action)

            return result

        return wrapper

    return decorator


def _extract_target_id(result: dict[str, Any]) -> str | None:
    """Extract a target ID from a tool result dict.

    Checks common keys in order: entry_id, conversation_id, topic.
    Returns the value as a string, or None if none are found.
    """
    for key in ("entry_id", "conversation_id", "topic"):
        value = result.get(key)
        if value is not None:
            return str(value)
    return None
