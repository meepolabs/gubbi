"""Harden extraction_jobs RLS policy against empty-string current_setting.

Migration 0023 (extraction_jobs_relocate) created the
``extraction_jobs_user_isolation`` policy with the un-hardened pattern:

    USING (user_id = (SELECT current_setting('app.current_user_id', true)::uuid))
    WITH CHECK (user_id = (SELECT current_setting('app.current_user_id', true)::uuid))

This casts ``current_setting(..., true)`` directly to ``::uuid``. Once a
``SET LOCAL app.current_user_id = '<uuid>'`` has been issued on a pooled
connection and the transaction commits, the GUC is registered for the
session but its value is reset to the empty string ``""`` (the
missing-safe flag only short-circuits when the GUC has *never* been
touched on the connection). The next query on the same connection that
hits this policy raises ``invalid input syntax for type uuid: ""``
instead of the intended fail-closed default-deny.

Migration 0007 (rls_policy_null_coalesce) established the canonical fix
for the five tenant tables (topics, entries, conversations, messages,
entry_embeddings): wrap with ``NULLIF(..., '')`` so the empty-string
residue is treated as missing. The missing -> NULL -> ``user_id = NULL``
evaluates to UNKNOWN (never true), so the row is filtered out without
the cast ever firing on the empty string.

Migration 0023 regressed that pattern when it added the new
``extraction_jobs`` table. This migration ports the 0007 NULLIF wrap to
the ``extraction_jobs_user_isolation`` policy. The structure mirrors
migration 0007 line-for-line for the policy clauses.

Defensive reasoning: nothing post-M3 has been deployed, so there is no
running state to migrate around. We still ``DROP POLICY IF EXISTS``
before ``CREATE POLICY`` so the migration is safe to re-run on any
environment that already has the fixed shape.

The downgrade restores the pre-fix policy without NULLIF; only intended
for emergency rollback. Re-introducing the empty-string cast bug is
never desirable, so do not invoke this downgrade outside an incident.
"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "0027_extraction_jobs_rls_nullif"
down_revision = "0026_otel_ro_role"
branch_labels = None
depends_on = None


_POLICY_COMMENT = (
    "Default-deny user isolation for extraction_jobs. Matches rows where "
    "user_id equals the session variable app.current_user_id (set per-request "
    "by the app via SET LOCAL). NULLIF(..., '') treats an empty-string residue "
    "(left over after a previous SET LOCAL on the same pooled connection) as "
    "NULL, so an unscoped connection sees zero rows instead of raising on the "
    "cast. BYPASSRLS roles (journal_admin) skip this policy entirely and see "
    "every row."
)


def upgrade() -> None:
    """Replace extraction_jobs_user_isolation with a NULLIF-safe version."""
    op.execute("DROP POLICY IF EXISTS extraction_jobs_user_isolation ON extraction_jobs")
    op.execute(
        """
        CREATE POLICY extraction_jobs_user_isolation ON extraction_jobs
            FOR ALL TO journal_app
            USING (user_id = (
                SELECT NULLIF(current_setting('app.current_user_id', true), '')::uuid
            ))
            WITH CHECK (user_id = (
                SELECT NULLIF(current_setting('app.current_user_id', true), '')::uuid
            ))
        """
    )
    op.execute(
        f"COMMENT ON POLICY extraction_jobs_user_isolation ON extraction_jobs IS "
        f"$policy${_POLICY_COMMENT}$policy$"
    )


def downgrade() -> None:
    """Restore the pre-fix policy shape (no NULLIF wrap).

    Regression downgrade: only intended for emergency rollback. The
    restored shape carries the empty-string cast bug that 0027 fixes;
    pooled connections that previously bound app.current_user_id will
    raise ``invalid input syntax for type uuid: ""`` on subsequent
    queries against extraction_jobs.
    """
    op.execute("DROP POLICY IF EXISTS extraction_jobs_user_isolation ON extraction_jobs")
    op.execute(
        """
        CREATE POLICY extraction_jobs_user_isolation ON extraction_jobs
            FOR ALL TO journal_app
            USING (user_id = (SELECT current_setting('app.current_user_id', true)::uuid))
            WITH CHECK (user_id = (SELECT current_setting('app.current_user_id', true)::uuid))
        """
    )
