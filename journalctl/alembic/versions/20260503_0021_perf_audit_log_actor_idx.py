"""Replace bare idx_audit_log_actor_id with composite (actor_id, occurred_at DESC).

The dominant query pattern on audit_log is:

    WHERE actor_id = $1 ORDER BY occurred_at DESC

A bare B-tree index on actor_id forces a forward scan followed by filtering.
A composite index on (actor_id, occurred_at DESC) supports an efficient
backward index scan that satisfies the sort without a separate step.

This migration uses CONCURRENTLY to avoid blocking concurrent audit inserts
during deployment.  The DROP runs before the CREATE -- this is safe because
the old index has no dependents (it was created alone).
"""

from typing import Any

from alembic import op

revision = "0021_perf_audit_log_actor_idx"
down_revision = "0020_audit_log_target_kind"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # CREATE INDEX CONCURRENTLY cannot run inside a transaction block.
    op.execute("COMMIT")

    conn = op.get_bind()
    raw: Any = conn.connection
    raw.autocommit = True
    try:
        # Drop the bare index concurrently to reduce deploy-time lock pressure.
        raw.execute("DROP INDEX CONCURRENTLY IF EXISTS idx_audit_log_actor_id")
        raw.execute(
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audit_log_actor_id "
            "ON audit_log (actor_id, occurred_at DESC)"
        )
    finally:
        raw.autocommit = False


def downgrade() -> None:
    # Drop the composite index first.
    op.execute("DROP INDEX IF EXISTS idx_audit_log_actor_id")
    # Re-create the original bare index (no CONCURRENTLY needed -- not a hot path).
    op.execute("CREATE INDEX idx_audit_log_actor_id ON audit_log(actor_id)")
