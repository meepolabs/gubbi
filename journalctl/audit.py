"""Append-only audit log helper -- DEC-061 / TASK-02.20.

This module ships as a ready-to-call helper. Call-site wiring (Kratos
webhooks, key rotation scripts, admin flows, subscription lifecycle) lands
per-feature in the appropriate task. Import record_audit and the Action
constants; call at the point of the privileged action; pass an active
asyncpg connection.

Usage pattern::

    from journalctl.audit import record_audit, Action

    async def delete_user(conn, user_id, requesting_admin):
        await record_audit(
            conn,
            actor_type="admin",
            actor_id=requesting_admin,
            action=Action.IDENTITY_DELETED,
            target_type="user",
            target_id=user_id,
            reason="GDPR erasure request",
            metadata={"user_id": user_id},
        )
        # ... perform deletion

Caller owns transaction lifecycle. record_audit() executes a single INSERT
inside whatever transaction (or autocommit context) the caller has open.

The ``Action`` enum and the underlying SQL template now live in
:mod:`gubbi_common.audit` (the cross-repo single source of truth);
this module wraps the SQL with the journalctl-specific OTel span and
re-exports ``Action`` so existing imports keep working.

Security contract:
- Do NOT log secret values. Log which secret rotated, not its content.
- Do NOT log PII. Log entity IDs, not email addresses or journal content.
- Compensating entries (not edits) are the only correction for erroneous rows;
  the database trigger unconditionally blocks UPDATE and DELETE.
"""

from __future__ import annotations

import time
from typing import Any

import asyncpg
from gubbi_common.audit.actions import Action
from gubbi_common.audit.sql import record_audit_async
from opentelemetry import trace

from journalctl.telemetry.attrs import _NS_PER_MS, _TRACER_NAME, SpanNames, safe_set_attributes

__all__ = ["Action", "record_audit"]


async def record_audit(
    conn: asyncpg.Connection,
    actor_type: str,
    actor_id: str,
    action: str,
    target_type: str | None = None,
    target_id: str | None = None,
    reason: str | None = None,
    metadata: dict[str, Any] | None = None,
    ip_address: str | None = None,
    user_agent: str | None = None,
) -> None:
    """Insert one immutable row into audit_log, wrapped in an OTel span.

    Thin instrumentation layer over
    :func:`gubbi_common.audit.sql.record_audit_async`. See that function
    for parameter documentation; the only difference is that this
    wrapper records an ``audit.write`` OTel span with success / latency
    attributes.
    """
    span_name = SpanNames.AUDIT_WRITE
    start_ns = time.monotonic_ns()

    attrs: dict[str, Any] = {
        "event_type": action,
    }
    if target_id is not None:
        attrs["target_id"] = target_id
    if actor_type is not None:
        attrs["actor_type"] = actor_type

    with trace.get_tracer(_TRACER_NAME).start_as_current_span(span_name) as span:
        safe_set_attributes(span_name, span, attrs)

        audit_success = False
        try:
            await record_audit_async(
                conn,
                actor_type=actor_type,
                actor_id=actor_id,
                action=action,
                target_type=target_type,
                target_id=target_id,
                reason=reason,
                metadata=metadata,
                ip_address=ip_address,
                user_agent=user_agent,
            )
            audit_success = True
        except Exception as exc:
            from opentelemetry.trace import Status, StatusCode

            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR))
            raise
        finally:
            latency_ms = (time.monotonic_ns() - start_ns) / _NS_PER_MS
            safe_set_attributes(
                span_name,
                span,
                {
                    "success": audit_success,
                    "latency_ms": round(latency_ms, 2),
                },
            )
