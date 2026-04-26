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
            action=Action.USER_DELETED,
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
from typing import Any, Final

import asyncpg

# ---------------------------------------------------------------------------
# Action constants
# ---------------------------------------------------------------------------
# These mirror the "events captured at minimum" list from TASK-02.20.
# Callers import from here to get IDE autocomplete and avoid raw string typos.


class Action:
    """Namespace for audit action string constants."""

    USER_CREATED: Final = "user.created"
    USER_DELETED: Final = "user.deleted"
    USER_RESTORED: Final = "user.restored"

    TENANT_PROVISIONED: Final = "tenant.provisioned"
    TENANT_SUSPENDED: Final = "tenant.suspended"
    TENANT_REACTIVATED: Final = "tenant.reactivated"

    SUBSCRIPTION_CREATED: Final = "subscription.created"
    SUBSCRIPTION_CANCELED: Final = "subscription.canceled"
    SUBSCRIPTION_OVERRIDE: Final = "subscription.override"

    SECRET_ROTATED: Final = "secret.rotated"  # noqa: S105 -- action label, not a password
    ADMIN_QUERY_EXECUTED: Final = "admin.query_executed"
    ENCRYPTION_KEY_ROTATED: Final = "encryption.key_rotated"
    AUTH_FOUNDER_IMPERSONATION: Final = "auth.founder_impersonation"


# Values that the CHECK constraint on actor_type accepts.
_VALID_ACTOR_TYPES: frozenset[str] = frozenset(
    {
        "user",
        "admin",
        "system",
        "founder",
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
        Must be one of: 'user', 'admin', 'system', 'founder'.
        Raises ValueError on any other value.
    actor_id:
        Opaque identifier for the actor (UUID string, 'founder:<email>',
        'system:<worker-name>', etc.).
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
    if actor_type not in _VALID_ACTOR_TYPES:
        raise ValueError(
            f"Invalid actor_type {actor_type!r}. " f"Must be one of: {sorted(_VALID_ACTOR_TYPES)}"
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
