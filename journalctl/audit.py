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

Security contract:
- Do NOT log secret values. Log which secret rotated, not its content.
- Do NOT log PII. Log entity IDs, not email addresses or journal content.
- Compensating entries (not edits) are the only correction for erroneous rows;
  the database trigger unconditionally blocks UPDATE and DELETE.
"""

from __future__ import annotations

import ipaddress
import json
import time
from typing import Any, Final

import asyncpg
from opentelemetry import trace

from journalctl.telemetry.attrs import _NS_PER_MS, _TRACER_NAME, SpanNames, safe_set_attributes

__all__ = ["Action", "record_audit"]

# ---------------------------------------------------------------------------
# Action constants
# ---------------------------------------------------------------------------
# These mirror the "events captured at minimum" list from TASK-02.20.
# Callers import from here to get IDE autocomplete and avoid raw string typos.


class Action:
    """Namespace for audit action string constants."""

    # Identity lifecycle events.  Values use the ``identity.*`` namespace so
    # downstream queries can filter ``action LIKE 'identity.%'`` and pick up
    # every identity-shaped event (created/updated/deleted/restored).
    # Migration 0015 rewrites legacy ``user.*`` rows from M2 to ``identity.*``.
    IDENTITY_CREATED: Final = "identity.created"
    IDENTITY_DELETED: Final = "identity.deleted"
    IDENTITY_RESTORED: Final = "identity.restored"

    TENANT_PROVISIONED: Final = "tenant.provisioned"
    TENANT_SUSPENDED: Final = "tenant.suspended"
    TENANT_REACTIVATED: Final = "tenant.reactivated"

    # auth events
    LOGIN_FAILED: Final = "login_failed"

    SUBSCRIPTION_CREATED: Final = "subscription.created"
    SUBSCRIPTION_CANCELED: Final = "subscription.canceled"
    SUBSCRIPTION_OVERRIDE: Final = "subscription.override"

    SECRET_ROTATED: Final = "secret.rotated"  # noqa: S105 -- action label, not a password
    ADMIN_QUERY_EXECUTED: Final = "admin.query_executed"
    ENCRYPTION_KEY_ROTATED: Final = "encryption.key_rotated"


# Values that the CHECK constraint on actor_type accepts.
# Must stay in sync with migration 0012's CHECK definition.
_VALID_ACTOR_TYPES: frozenset[str] = frozenset(
    {
        "user",
        "admin",
        "system",
        "hydra_subject",
    }
)

_INSERT_SQL = """
    INSERT INTO audit_log
        (actor_type, actor_id, action, target_type, target_id,
         reason, metadata, ip_address, user_agent)
    VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8::inet, $9)
"""


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
    """Insert one immutable row into audit_log.

    Parameters
    ----------
    conn:
        Active asyncpg connection. Caller owns transaction lifecycle.
    actor_type:
        Must be one of: 'user', 'admin', 'system', 'hydra_subject'.
        Raises ValueError on any other value.
    actor_id:
        Opaque identifier for the actor (UUID string, 'system:<worker-name>',
        'script:<name>', etc.).
    action:
        Event string. Use Action.* constants for the 13 documented events.
        Custom strings are accepted for extensibility.
    target_type:
        Optional entity kind ('user', 'tenant', 'subscription', 'secret', ...).
    target_id:
        Optional entity identifier.
    reason:
        Optional human-readable explanation.
    metadata:
        Optional dict of action-specific context. Must be JSON-serializable.
        Defaults to empty dict. Never include secret values or PII.
    ip_address:
        Optional originating IP. Passed as TEXT; PostgreSQL casts to INET.
    user_agent:
        Optional HTTP User-Agent string.

    Raises
    ------
    ValueError
        If actor_type is not one of the four accepted values.
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

        # Validation — runs before the DB INSERT and raises on failure.
        # The span records success=False when validation fails.
        audit_success = False
        try:
            if actor_type not in _VALID_ACTOR_TYPES:
                raise ValueError(
                    f"Invalid actor_type {actor_type!r}. "
                    f"Must be one of: {sorted(_VALID_ACTOR_TYPES)}"
                )

            if ip_address:
                try:
                    ipaddress.ip_address(ip_address)
                except ValueError:
                    raise ValueError(
                        f"audit.record_audit: invalid ip_address {ip_address!r}; "
                        "must be a valid IPv4 or IPv6 address"
                    ) from None

            resolved_metadata: dict[str, Any] = metadata if metadata is not None else {}
            metadata_json = json.dumps(resolved_metadata)

            await conn.execute(
                _INSERT_SQL,
                actor_type,
                actor_id,
                action,
                target_type,
                target_id,
                reason,
                metadata_json,
                ip_address,
                user_agent,
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
