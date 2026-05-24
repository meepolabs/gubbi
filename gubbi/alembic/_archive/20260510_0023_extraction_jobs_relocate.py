"""Relocate extraction_jobs table to gubbi as a user_id-keyed data-plane table.

Ownership transferred from gubbi-cloud (where it was tenant_id-keyed) to gubbi
(user_id-keyed, consistent with conversations/entries/topics/audit_log).

Upgrade:
  1. DROP TABLE IF EXISTS extraction_jobs CASCADE -- removes any remnant from
     gubbi-cloud migration 0003_billing_tables if the table was ever created
     in this database. CASCADE removes the RLS policy added by gubbi-cloud
     migration 0006_extend_rls if present.
  2. CREATE TABLE extraction_jobs with user_id FK, conversation_id INTEGER FK,
     status, source, progress counters, timestamps.
  3. Two indexes: composite (user_id, created_at DESC) for list queries;
     partial unique on (user_id, conversation_id, source) WHERE status NOT IN
     ('completed', 'failed') to enforce one in-flight job per conversation.
  4. RLS: ENABLE + FORCE, policy extraction_jobs_user_isolation, identical
     pattern to migration 0005_enable_rls tenant_isolation policies.
  5. GRANT SELECT, INSERT, UPDATE ON extraction_jobs TO journal_app.

Downgrade:
  Drops the table and all dependent objects (CASCADE handles policy + indexes).
"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "0023_extraction_jobs_relocate"
down_revision = "0022_audit_log_identity"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create extraction_jobs table with RLS under gubbi ownership."""

    # Step 1: Drop any remnant from gubbi-cloud.
    # CASCADE removes the gubbi-cloud RLS policy (0006_extend_rls) if present.
    op.execute(sa.text("DROP TABLE IF EXISTS extraction_jobs CASCADE"))

    # Step 2: Create the table with the final schema.
    op.execute(
        sa.text(
            """
            CREATE TABLE extraction_jobs (
                id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id         UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                source          TEXT NOT NULL,
                status          TEXT NOT NULL,
                topics_created  INTEGER NOT NULL DEFAULT 0,
                entries_created INTEGER NOT NULL DEFAULT 0,
                cents_spent     INTEGER NOT NULL DEFAULT 0,
                error_code      TEXT,
                created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
                started_at      TIMESTAMPTZ,
                completed_at    TIMESTAMPTZ
            )
            """
        )
    )

    # Step 3: Indexes.
    op.execute(
        sa.text(
            "CREATE INDEX idx_extraction_jobs_user_created"
            " ON extraction_jobs(user_id, created_at DESC)"
        )
    )
    op.execute(
        sa.text(
            "CREATE UNIQUE INDEX idx_extraction_jobs_active_per_conversation"
            " ON extraction_jobs(user_id, conversation_id, source)"
            " WHERE status NOT IN ('completed', 'failed')"
        )
    )

    # Step 4: RLS -- same InitPlan subquery pattern as migration 0005_enable_rls.
    op.execute(sa.text("ALTER TABLE extraction_jobs ENABLE ROW LEVEL SECURITY"))
    op.execute(sa.text("ALTER TABLE extraction_jobs FORCE ROW LEVEL SECURITY"))
    op.execute(sa.text("DROP POLICY IF EXISTS extraction_jobs_user_isolation ON extraction_jobs"))
    op.execute(
        sa.text(
            """
            CREATE POLICY extraction_jobs_user_isolation ON extraction_jobs
                USING (user_id = (SELECT current_setting('app.current_user_id', true)::uuid))
                WITH CHECK (user_id = (SELECT current_setting('app.current_user_id', true)::uuid))
            """
        )
    )

    # Step 5: Grants.
    op.execute(sa.text("GRANT SELECT, INSERT, UPDATE ON extraction_jobs TO journal_app"))


def downgrade() -> None:
    """Drop extraction_jobs; CASCADE removes policy and indexes."""
    op.execute(sa.text("DROP TABLE IF EXISTS extraction_jobs CASCADE"))
