"""Add cross-attribution guards to audit_log.

G1: RLS WITH CHECK policy on journal_app INSERTs ensures actor_id matches
    the session-scoped app.current_user_id GUC, preventing the app pool
    from inserting rows on behalf of a different user.

G2: BEFORE INSERT trigger blocks journal_admin from inserting rows with
    actor_type='user', preventing admin-pool code from claiming user
    attribution.

Both guards ship in a single migration because they close related concerns
in the same security boundary.  journal_admin has BYPASSRLS so the RLS
policy does not affect webhook code; the trigger is the hammer for admin.
"""

from alembic import op

revision = "0020_audit_log_cross_attribution_guard"
down_revision = "0019_rls_users"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # G1: RLS WITH CHECK on journal_app
    op.execute("ALTER TABLE audit_log ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE audit_log FORCE ROW LEVEL SECURITY")
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
                    ' use app_pool/user_scoped_connection or set actor_type to system/admin/founder'
                    USING ERRCODE = 'insufficient_privilege';
            END IF;
            RETURN NEW;
        END;
        $func$ LANGUAGE plpgsql
        """
    )
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
