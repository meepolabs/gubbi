"""Add cross-attribution guards to audit_log.

Originally authored as 0020 alongside the live ``0020_audit_log_target_kind``
migration; left orphaned by the off-list fork at 0019 (both 0020s named
``down_revision = "0019_rls_users"``). Relocated to a sequential 0028 on
2026-05-13 to restore a single linear head. Lands sequentially after the
CRIT-2 NULLIF fix (``0027_extraction_jobs_rls_nullif``); chain reads
``... -> 0026_otel_ro_role -> 0027_extraction_jobs_rls_nullif
-> 0028_audit_log_cross_attribution_guard``. The two CRIT fixes are
orthogonal: 0027 hardens the ``extraction_jobs`` RLS policy, while 0028
adds RLS + trigger guards on ``audit_log``. Idempotency-hardened in place
(drop-then-create on the policy and trigger) so the migration is safe to
re-run.

G1: RLS WITH CHECK policy on journal_app INSERTs ensures actor_id matches
    the session-scoped app.current_user_id GUC, preventing the app pool
    from inserting rows on behalf of a different user.

G2: BEFORE INSERT trigger blocks journal_admin from inserting rows with
    actor_type='user', preventing admin-pool code from claiming user
    attribution.

Both guards ship in a single migration because they close related concerns
in the same security boundary.  journal_admin has BYPASSRLS so the RLS
policy does not affect webhook code; the trigger is the hammer for admin.

Design note: actor_type discipline is a caller-side convention, not a
DB-enforced invariant.  The G1 RLS WITH CHECK enforces ONLY that
``actor_id`` equals the session's ``app.current_user_id`` GUC.  It does
NOT constrain ``actor_type``.  journal_app callers can in principle write
any ``actor_type`` value as long as ``actor_id`` matches the user's UUID.
The existing convention is that journal_app code paths only ever pass
``actor_type='user'``; system audits go through journal_admin (BYPASSRLS)
with non-UUID actor_ids like ``'stripe-events-retention-sweep'``.  The
role boundary is data isolation (RLS by user_id); actor_type is a
code-review/lint-level concern, not a DB-level one.  Buggy code paths
that emit a wrong actor_type from journal_app would result in
misleading-but-recoverable audit metadata, not a data-leakage gap.

G2 (the BEFORE INSERT trigger) IS asymmetric and DOES police actor_type
for journal_admin: an admin session cannot insert ``actor_type='user'``
rows.  This is enforced because admin operations are categorically not
user operations and the boundary is meaningful at the role level.
"""

from alembic import op

revision = "0028_audit_log_cross_attribution_guard"
down_revision = "0027_extraction_jobs_rls_nullif"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # G1: RLS WITH CHECK on journal_app
    op.execute("ALTER TABLE audit_log ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE audit_log FORCE ROW LEVEL SECURITY")
    op.execute("DROP POLICY IF EXISTS audit_log_app_insert_self_only ON audit_log")
    op.execute(
        """
        CREATE POLICY audit_log_app_insert_self_only
            ON audit_log
            FOR INSERT
            TO journal_app
            WITH CHECK (
                actor_id = NULLIF(current_setting('app.current_user_id', true), '')
                AND actor_id <> ''
            )
        """
    )

    # G2: BEFORE INSERT trigger blocking actor_type='user' from journal_admin
    op.execute(
        """
        CREATE OR REPLACE FUNCTION audit_log_admin_no_user_actor() RETURNS trigger AS $func$
        BEGIN
            IF current_user = 'journal_admin' AND NEW.actor_type = 'user' THEN
                RAISE EXCEPTION
                    'journal_admin cannot insert audit_log row with actor_type=user;'
                    ' use app_pool/user_scoped_connection or set actor_type to system/admin/hydra_subject'
                    USING ERRCODE = 'insufficient_privilege';
            END IF;
            RETURN NEW;
        END;
        $func$ LANGUAGE plpgsql
        """  # noqa: E501
    )
    op.execute("DROP TRIGGER IF EXISTS trg_audit_log_admin_no_user_actor ON audit_log")
    op.execute(
        """
        CREATE TRIGGER trg_audit_log_admin_no_user_actor
            BEFORE INSERT ON audit_log
            FOR EACH ROW EXECUTE FUNCTION audit_log_admin_no_user_actor()
        """
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS trg_audit_log_admin_no_user_actor ON audit_log")
    op.execute("DROP FUNCTION IF EXISTS audit_log_admin_no_user_actor()")
    op.execute("DROP POLICY IF EXISTS audit_log_app_insert_self_only ON audit_log")
    op.execute("ALTER TABLE audit_log NO FORCE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE audit_log DISABLE ROW LEVEL SECURITY")
