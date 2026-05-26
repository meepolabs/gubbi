"""Append-only audit log helper.

This module ships as a ready-to-call helper. Call-site wiring (Kratos
webhooks, key rotation scripts, admin flows, subscription lifecycle) lands
per-feature in the appropriate task. Import ``record_audit`` and the
``Action`` constants; call at the point of the privileged action; pass an
active asyncpg connection.

Usage pattern::

    from gubbi.audit import record_audit, Action

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

Caller owns transaction lifecycle. ``record_audit`` executes a single
INSERT inside whatever transaction (or autocommit context) the caller has
open.

As of gubbi-common 0.11.0 the canonical INSERT,
target_id / actor_id validation, banned-key metadata redaction, IP
normalisation, metadata size cap, and ``audit.write`` OTel span all live
in :func:`gubbi_common.audit.sql.record_audit_async`. This module re-exports
that helper as ``record_audit`` so existing imports keep working without
the duplicate local INSERT shape.

The ``Action`` enum lives in :mod:`gubbi_common.audit` (the cross-repo
single source of truth); this module re-exports ``Action`` so existing
imports keep working.

See the audit contract for actor_type taxonomy and when to
use ``record_audit`` vs. the ``@audited`` decorator.

Security contract:
- Do NOT log secret values. Log which secret rotated, not its content.
- Do NOT log PII. Log entity IDs, not email addresses or journal content.
- Compensating entries (not edits) are the only correction for erroneous rows;
  the database trigger unconditionally blocks UPDATE and DELETE.
"""

from __future__ import annotations

from gubbi_common.audit.actions import Action
from gubbi_common.audit.sql import record_audit_async as record_audit

__all__: list[str] = ["Action", "record_audit"]
