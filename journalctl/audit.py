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

The ``Action`` enum now lives in :mod:`gubbi_common.audit` (the cross-repo
single source of truth); this module re-exports ``Action`` so existing
imports keep working.  The INSERT SQL is defined locally (not delegated to
``gubbi_common.audit.sql.record_audit_async``) so the ``target_kind``
column added in migration 0020 is included in the insert.

See ``llm_context/audit_contract.md`` for actor_type taxonomy and when to
use ``record_audit`` vs. the ``@audited`` decorator.

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
from typing import Any

import asyncpg
from gubbi_common.audit.actions import Action
from gubbi_common.audit.sql import VALID_ACTOR_TYPES
from opentelemetry import trace

from journalctl.telemetry.attrs import _NS_PER_MS, _TRACER_NAME, SpanNames, safe_set_attributes

__all__ = ["Action", "record_audit"]

# 10-column insert including target_kind (migration 0020).  Defined here
# rather than in gubbi_common so the column is always present.
_AUDIT_INSERT_SQL: str = """
    INSERT INTO audit_log
        (actor_type, actor_id, action, target_type, target_id, target_kind,
         reason, metadata, ip_address, user_agent)
    VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9::inet, $10)
"""


async def record_audit(
    conn: asyncpg.Connection,
    actor_type: str,
    actor_id: str,
    action: str,
    target_type: str | None = None,
    target_id: str | None = None,
    target_kind: str | None = None,
    reason: str | None = None,
    metadata: dict[str, Any] | None = None,
    ip_address: str | None = None,
    user_agent: str | None = None,
) -> None:
    """Insert one immutable row into audit_log, wrapped in an OTel span.

    Parameters
    ----------
    conn :
        Active asyncpg connection.
    actor_type :
        One of ``user``, ``admin``, ``system``, ``hydra_subject``.
        Raises ``ValueError`` on any other value.
    actor_id :
        Opaque actor identifier (UUID string,
        ``system:<worker-name>``, ``script:<name>``, ...).
    action :
        Event string. Use values from ``gubbi_common.audit.actions.Action``.
    target_type :
        Optional entity kind (``user``, ``tenant``, ``subscription``,
        ``secret``, ...).
    target_id :
        Optional entity identifier.
    target_kind :
        Required when ``target_id`` is supplied.  Namespace discriminator
        that prevents cross-namespace collisions in the partial unique
        index ``audit_log_content_hash_uidx`` (e.g. ``"entry"``,
        ``"conversation"``, ``"topic"``).
    reason :
        Optional human-readable explanation.
    metadata :
        Optional JSON-serialisable dict. Defaults to empty dict. Never
        include secret values or PII; pass hashes when forensics need
        the link.
    ip_address :
        Optional originating IP. Validated against ``ipaddress`` and
        cast to ``inet`` server-side.
    user_agent :
        Optional HTTP User-Agent string.

    Raises
    ------
    ValueError
        If ``target_id`` is supplied without ``target_kind``, or if
        ``actor_type`` is not one of the four accepted values, or if
        ``ip_address`` is not a valid IPv4 / IPv6 address.
    """
    if target_id is not None and target_kind is None:
        raise ValueError("target_kind is required when target_id is supplied")

    if actor_type not in VALID_ACTOR_TYPES:
        raise ValueError(
            f"Invalid actor_type {actor_type!r}. " f"Must be one of: {sorted(VALID_ACTOR_TYPES)}"
        )

    if ip_address is not None:
        ip_address = ip_address.strip()
        if not ip_address:
            # Treat blank string as None (no IP logged).
            ip_address = None
        else:
            try:
                ipaddress.ip_address(ip_address)
            except ValueError:
                raise ValueError(
                    f"record_audit: invalid ip_address {ip_address!r}; "
                    "must be a valid IPv4 or IPv6 address"
                ) from None

    span_name = SpanNames.AUDIT_WRITE
    start_ns = time.monotonic_ns()

    attrs: dict[str, Any] = {
        "event_type": action,
    }
    if target_id is not None:
        attrs["target_id"] = target_id
    if target_kind is not None:
        attrs["target_kind"] = target_kind
    if actor_type is not None:
        attrs["actor_type"] = actor_type

    resolved_metadata: dict[str, Any] = metadata if metadata is not None else {}
    metadata_json = json.dumps(resolved_metadata)

    with trace.get_tracer(_TRACER_NAME).start_as_current_span(span_name) as span:
        safe_set_attributes(span_name, span, attrs)

        audit_success = False
        try:
            await conn.execute(
                _AUDIT_INSERT_SQL,
                actor_type,
                actor_id,
                action,
                target_type,
                target_id,
                target_kind,
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
