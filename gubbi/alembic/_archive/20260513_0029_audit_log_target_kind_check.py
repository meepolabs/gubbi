"""DB-level enforcement for the ``target_id requires target_kind`` invariant.

Adds CHECK constraint ``audit_log_target_kind_invariant`` to
``audit_log``:

    CHECK (target_id IS NULL OR target_kind IS NOT NULL)

Background: the Python boundary (``record_audit_async`` in gubbi-common
0.11.0) enforces ``target_id requires target_kind`` -- the dedup partial
unique index ``audit_log_content_hash_uidx`` keys on ``target_kind``, so a
row with a non-NULL ``target_id`` but a NULL ``target_kind`` would skip
the dedup namespace discriminator and risk false unique-violation
collisions across kinds (entry "42" vs. topic "42"). Belt-and-braces: the
DB enforces the invariant directly so a future writer that bypasses the
Python helper (or a buggy migration) cannot land a row in the
inconsistent shape. Locked by A3 Q3 (2026-05-13).

Migration shape: ``ADD CONSTRAINT ... NOT VALID`` followed by
``VALIDATE CONSTRAINT`` -- standard safe-rollout pattern. Since the
``audit_log`` table is effectively empty (no deploy since M3), the
VALIDATE pass is a no-op walk. The two-step is kept anyway for the
pattern's documentation value: if this migration is ever replayed against
a non-empty audit_log, the same two statements give us a non-blocking
``NOT VALID`` add followed by an explicit validation step we can defer.
"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "0029_audit_log_target_kind_check"
down_revision = "0028_audit_log_cross_attribution_guard"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add the target_kind invariant CHECK and validate it."""
    op.execute(
        "ALTER TABLE audit_log "
        "ADD CONSTRAINT audit_log_target_kind_invariant "
        "CHECK (target_id IS NULL OR target_kind IS NOT NULL) NOT VALID"
    )
    op.execute("ALTER TABLE audit_log VALIDATE CONSTRAINT audit_log_target_kind_invariant")


def downgrade() -> None:
    """Drop the target_kind invariant CHECK."""
    op.execute("ALTER TABLE audit_log DROP CONSTRAINT IF EXISTS audit_log_target_kind_invariant")
