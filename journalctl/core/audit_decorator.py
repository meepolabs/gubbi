"""@audited decorator for MCP tool handlers (M3 requirement).

Records an audit event after a write tool handler completes successfully.
Best-effort at the decorator layer: ``record_audit`` propagates exceptions
on infra failure; this decorator catches them, logs, and increments a
metric -- the caller never sees an audit exception.  Direct (non-decorator)
callers of ``record_audit`` receive the exception and must handle it
themselves.

Stacking order (outermost to innermost)::

    @mcp.tool()
    @require_scope("journal:write")
    @audited("entry.created", target_type="entry", target_kind="entry", app_ctx=app_ctx)
    async def journal_append_entry(...) -> dict:
        ...

Contract (success heuristic)
-----------------------------
The decorator inspects the handler's return value to decide success/failure:

* If the return value is a ``mcp.types.CallToolResult``, the ``.isError``
  attribute decides success (``isError=False`` → success).
* Otherwise (a plain ``dict``), the heuristic is
  ``result.get("success", True)`` — a result is assumed successful unless
  it explicitly sets ``success=False``.

  *Rationale:* The old ``"error" not in result`` heuristic falsely treated
  ``CallToolResult(isError=True, ...)`` as success (the dict surrogate for
  MCP errors in this codebase uses an ``"error"`` key, but the
  ``CallToolResult`` envelope does not).  See H-3 audit-decorator-fix.

Actor context
--------------
The decorator reads ``current_user_id`` from the request context.  If the
ContextVar is unset (e.g. Arq worker path) the audit is silently skipped
and a warning is logged.  This is safe: extraction workers and other
non-MCP code paths call ``record_audit`` directly with an explicit
``actor_id``.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, ParamSpec, TypeVar

from gubbi_common.db.user_scoped import user_scoped_connection

from journalctl.audit import record_audit
from journalctl.core.auth_context import current_user_id
from journalctl.core.context import AppContext
from journalctl.telemetry.metrics import record_audit_persistence_failure

logger = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R", bound=Mapping[str, Any])

ACTION_ENTRY_CREATED = "entry.created"
ACTION_ENTRY_UPDATED = "entry.updated"
ACTION_ENTRY_DELETED = "entry.deleted"
ACTION_TOPIC_CREATED = "topic.created"
ACTION_CONVERSATION_SAVED = "conversation.saved"

_TARGET_KEYS: dict[str, tuple[str, str]] = {
    "entry": ("entry_id", "entry"),
    "conversation": ("conversation_id", "conversation"),
    "topic": ("topic", "topic"),
}


def audited(
    action: str,
    target_type: str,
    app_ctx: AppContext,
    *,
    target_kind: str | None = None,
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
    target_kind :
        Namespace discriminator for ``target_id`` (e.g. ``"entry"``,
        ``"conversation"``, ``"topic"``).  Required by ``record_audit``
        when ``target_id`` is derived from the handler result.
    """

    if target_type not in _TARGET_KEYS:
        logger.warning(
            "audited() target_type=%r has no _TARGET_KEYS entry; "
            "audit rows for action=%r will be written with target_id=None",
            target_type,
            action,
        )

    def decorator(fn: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
        @functools.wraps(fn)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            result = await fn(*args, **kwargs)

            # Determine success/failure.  CallToolResult uses .isError;
            # plain dicts use a "success" key (defaulting to True).
            is_success = _result_is_success(result)
            if is_success:
                user_id = current_user_id.get()
                if user_id is not None:
                    if isinstance(result, dict):
                        target_id, derived_kind = _extract_target_id(result, target_type)
                    else:
                        target_id, derived_kind = None, None
                    # caller-supplied target_kind takes precedence over
                    # the derived one (e.g. conversations returns
                    # conversation_id but callers may set
                    # target_kind="conversation").
                    effective_kind = target_kind or derived_kind
                    try:
                        async with user_scoped_connection(app_ctx.pool, user_id=user_id) as conn:
                            await record_audit(
                                conn,
                                actor_type="user",
                                actor_id=str(user_id),
                                action=action,
                                target_type=target_type,
                                target_id=target_id,
                                target_kind=effective_kind,
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


def _result_is_success(result: Any) -> bool:
    """Decide whether the handler result represents a success.

    * ``CallToolResult`` (or any object with ``.isError``) → ``not result.isError``
    * ``dict`` → ``result.get("success", True)``
    * anything else → ``True`` (assume success)
    """
    # Duck-type CallToolResult: any object with a boolean .isError attribute.
    if hasattr(result, "isError"):
        return not result.isError
    if isinstance(result, dict):
        return bool(result.get("success", True))
    return True


def _extract_target_id(
    result: dict[str, Any],
    target_type: str,
) -> tuple[str | None, str | None]:
    """Look up the target_id by target_type rather than positional walk.

    Returns (target_id, target_kind) for the named target_type if its
    key is present in the result dict; (None, None) otherwise.
    """
    mapping = _TARGET_KEYS.get(target_type)
    if mapping is None:
        return None, None
    result_key, kind = mapping
    value = result.get(result_key)
    if value is None:
        return None, None
    return str(value), kind
