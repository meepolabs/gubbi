"""Add target_kind column to audit_log for namespace-scoped dedup.

The partial unique index audit_log_content_hash_uidx (migration 0016) indexes
(target_id, action, metadata->>'content_hash') but target_id is heterogeneous
-- numeric for entries, path strings for topics, etc.  A cross-namespace
collision (e.g. entry id "42" vs. a topic path "42") would hit a false
unique-violation.

target_kind provides the namespace discriminator.  Going forward,
record_audit() requires target_kind whenever target_id is supplied.
Existing rows have NULL target_kind (nullable column, no backfill needed).

The upgrade rebuilds the dedup index with target_kind as the first column
to prevent cross-namespace collisions.  Both index creates use CONCURRENTLY
to avoid blocking concurrent audit writes during deployment; the ALTER TABLE
runs inside Alembic's transaction but the index builds run outside it.
"""

from alembic import op

revision = "0020_audit_log_target_kind"
down_revision = "0019_rls_users"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ------------------------------------------------------------------
    # Step 1: Add column (within the Alembic-managed transaction)
    # ------------------------------------------------------------------
    op.execute("ALTER TABLE audit_log ADD COLUMN target_kind TEXT")

    # CREATE INDEX CONCURRENTLY cannot run inside a transaction block.
    # Commit the ALTER TABLE so that subsequent statements can run
    # outside a transaction.  From here on every statement is its own
    # implicit transaction (autocommit on the psycopg connection).
    op.execute("COMMIT")

    conn = op.get_bind()
    raw = conn.connection  # psycopg.Connection  # type: ignore[attr-defined]
    raw.autocommit = True  # type: ignore[attr-defined]
    try:
        # ------------------------------------------------------------------
        # Step 2: Rebuild the dedup index with target_kind to prevent
        #         cross-namespace false unique-violations (H-3 follow-up).
        # ------------------------------------------------------------------
        raw.execute("DROP INDEX IF EXISTS audit_log_content_hash_uidx")
        raw.execute(
            "CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS audit_log_content_hash_uidx "
            "ON audit_log (target_kind, target_id, action, (metadata->>'content_hash')) "
            "WHERE metadata ? 'content_hash'"
        )

        # ------------------------------------------------------------------
        # Step 3: Lookup index for queries scoped by (target_kind, target_id).
        # ------------------------------------------------------------------
        raw.execute(
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audit_log_target_kind_target_id "
            "ON audit_log (target_kind, target_id)"
        )
    finally:
        raw.autocommit = False  # type: ignore[attr-defined]


def downgrade() -> None:
    # Downgrade runs inside the normal Alembic transaction -- no CONCURRENTLY
    # needed for index drops or re-creation of the old shape.
    op.execute("DROP INDEX IF EXISTS idx_audit_log_target_kind_target_id")
    op.execute("DROP INDEX IF EXISTS audit_log_content_hash_uidx")
    # Recreate the original unique index (without target_kind)
    op.execute(
        "CREATE UNIQUE INDEX audit_log_content_hash_uidx "
        "ON audit_log (target_id, action, (metadata->>'content_hash')) "
        "WHERE metadata ? 'content_hash'"
    )
    op.execute("ALTER TABLE audit_log DROP COLUMN IF EXISTS target_kind")
