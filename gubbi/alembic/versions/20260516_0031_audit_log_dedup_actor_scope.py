"""Per-actor scoping on the ``audit_log`` content-hash dedup index.

Replaces the partial unique index ``audit_log_content_hash_uidx``
introduced by migration 0016
(``20260429_0016_audit_log_content_hash_dedup.py``) with one that
prepends ``actor_id`` to the keyed tuple. Pre-migration::

    UNIQUE (target_kind, target_id, action, (metadata->>'content_hash'))
        WHERE metadata ? 'content_hash'

Post-migration::

    UNIQUE (actor_id, target_kind, target_id, action,
            (metadata->>'content_hash'))
        WHERE metadata ? 'content_hash'

Background: the dedup index landed in 0016 to give the cloud-api
Kratos webhook handler an ``INSERT ... ON CONFLICT DO NOTHING`` path
for retry idempotency. Migration 0015
(``audit_log_user_to_identity``) reshaped ``actor_id`` into the
canonical actor column post-tenant-split, but 0016's unique tuple was
not updated to include it. The same oversight class as the
``topics.path`` global-uniqueness bug fixed by 0030
(``topics_user_path_unique``): a column was added to govern per-tenant
scoping, but a unique constraint introduced afterwards was not updated
to include it.

Concrete failure shape: webhook idempotency is the intended use of
this dedup -- Stripe / Kratos retries arrive with the same
``(target_kind, target_id, action, content_hash)`` tuple and the
second INSERT correctly DO-NOTHINGs. But if two distinct actors ever
produce the same tuple (e.g. a system actor and a user actor on the
same target with content that happens to hash identically, or two
scheduled-job actors hitting the same target), the second INSERT
silently fails -- a row that should have landed is dropped. For an
audit log marketed as forensic ground truth (immutable triggers,
REVOKE, BEFORE-DELETE guards from migration 0010), a swallowed insert
is a tampering vector.

Today's blast radius is narrow because ``content_hash`` is currently
set only by webhook callers whose ``actor_id`` is effectively a fixed
system value. A future caller that opts into the dedup with a
per-user ``actor_id`` would surface the bug. The fix is shipped now
rather than later because the migration is small and the alternative
is shipping the future caller and remembering to migrate at the same
time.

Webhook idempotency contract preserved: the cloud-api path that uses
the dedup writes with a constant ``actor_id`` per webhook source
(e.g. ``actor_id = '<kratos-system-actor-uuid>'``), so the same
retry from the same actor still keys the same tuple and still
DO-NOTHINGs on conflict. The cross-actor false-collision is closed.

Pre-flight: refuses to apply if any
``(actor_id, target_kind, target_id, action, content_hash)`` group
already has count > 1. Under the existing unique
``(target_kind, target_id, action, content_hash)`` no such group can
exist by construction (a stronger constraint forbids it), so the
pre-flight is cheap insurance against weird state from a
downgrade/upgrade cycle on a populated DB; the migration aborts
loudly rather than silently producing a half-migrated schema.

Lock duration: ``audit_log`` is small in practice (100 rows in the
testbench corpus; designed for ~1k events/day per gubbi alembic 0010
docstring -- well under a million in v1). Plain ``CREATE UNIQUE
INDEX`` (no ``CONCURRENTLY``) takes a ``ShareLock`` for the index
build; same trade-off as 0030 and 0025 (``messages_search_vector_gin``)
which deliberately avoid ``CONCURRENTLY`` for small / pre-launch
tables -- ``CONCURRENTLY`` brings transaction-management complexity
(``autocommit_block`` in ``gubbi.alembic._helpers``) that is only
justified for large production tables.

Downgrade: symmetric -- restore the original 0016 partial unique
keyed on ``(target_kind, target_id, action, content_hash)``. If the
post-upgrade table contains rows where two distinct actors share
``(target_kind, target_id, action, content_hash)`` (a state that is
reachable post-upgrade but unrepresentable in the pre-upgrade
schema), the downgrade FAILS at index creation. That is intentional:
silently dropping rows would be worse than raising. Operators
wanting to roll back must first reconcile the cross-actor
collisions manually -- there is no "right" choice between which
actor's row to keep, and the audit-log immutability contract forbids
the migration making that call.
"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "0031_audit_log_dedup_actor_scope"
down_revision = "0030_topics_user_path_unique"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Replace the dedup unique index with an ``actor_id``-scoped one."""
    conn = op.get_bind()
    duplicate_groups = conn.execute(
        sa.text(
            "SELECT count(*) FROM ("
            "  SELECT actor_id, target_kind, target_id, action,"
            "         metadata->>'content_hash' AS h"
            "  FROM audit_log"
            "  WHERE metadata ? 'content_hash'"
            "  GROUP BY 1, 2, 3, 4, 5 HAVING count(*) > 1"
            ") d"
        )
    ).scalar()
    if duplicate_groups:
        # ``duplicate_groups`` is a server-side integer count, not
        # user input; the SELECT shape inside the f-string is a
        # diagnostic snippet for the operator, not a query that gets
        # executed. The S608 noqa scopes the suppression to the
        # literal that ruff flags.
        raise RuntimeError(
            f"audit_log has {duplicate_groups} (actor_id, target_kind, "  # noqa: S608
            "target_id, action, content_hash) duplicate group(s); "
            "refusing to apply 0031. Investigate via "
            "`SELECT actor_id, target_kind, target_id, action, "
            "metadata->>'content_hash' AS h, count(*) FROM audit_log "
            "WHERE metadata ? 'content_hash' GROUP BY 1,2,3,4,5 "
            "HAVING count(*) > 1` and reconcile before re-running. "
            "Note: audit_log immutability triggers (gubbi alembic 0010) "
            "require SET session_replication_role = 'replica' under a "
            "superuser session for any DELETE."
        )
    op.execute("DROP INDEX IF EXISTS audit_log_content_hash_uidx")
    op.execute(
        "CREATE UNIQUE INDEX audit_log_content_hash_uidx "
        "ON audit_log (actor_id, target_kind, target_id, action, "
        "(metadata->>'content_hash')) "
        "WHERE metadata ? 'content_hash'"
    )


def downgrade() -> None:
    """Restore the original 0016 partial unique without ``actor_id``.

    NOTE: if the table contains rows where two distinct actors share
    ``(target_kind, target_id, action, content_hash)`` (a state that
    is reachable post-upgrade but unrepresentable pre-upgrade), the
    index creation FAILS here. That is intentional -- silently
    dropping rows would be worse than raising, and the audit-log
    immutability contract forbids the migration choosing which
    actor's row to keep. Investigate via
    ``SELECT target_kind, target_id, action, metadata->>'content_hash'
    AS h, count(*) FROM audit_log WHERE metadata ? 'content_hash'
    GROUP BY 1,2,3,4 HAVING count(*) > 1`` and reconcile manually
    before re-attempting the downgrade.
    """
    op.execute("DROP INDEX IF EXISTS audit_log_content_hash_uidx")
    op.execute(
        "CREATE UNIQUE INDEX audit_log_content_hash_uidx "
        "ON audit_log (target_kind, target_id, action, "
        "(metadata->>'content_hash')) "
        "WHERE metadata ? 'content_hash'"
    )
