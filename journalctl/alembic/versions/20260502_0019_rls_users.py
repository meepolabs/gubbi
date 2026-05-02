"""Add Row-Level Security to the users table.

users is created in 0003 but was intentionally omitted from _TENANT_TABLES in
0005/0007 because the standard tenant_isolation policy shape (user_id = GUC)
does not fit: users.id IS the user, so the policy checks id, not a user_id FK.

This migration closes m234-review C-7 with the correct policy shape.
"""

from alembic import op

revision = "0019_rls_users"
down_revision = "0018_add_conversations_processed_at"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE users ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE users FORCE ROW LEVEL SECURITY")

    op.execute("DROP POLICY IF EXISTS users_self_read ON users")
    op.execute(
        """
        CREATE POLICY users_self_read ON users
            FOR SELECT TO journal_app
            USING (
                id = (SELECT NULLIF(current_setting('app.current_user_id', true), '')::uuid)
                AND deleted_at IS NULL
            )
        """
    )

    op.execute("DROP POLICY IF EXISTS users_self_update ON users")
    op.execute(
        """
        CREATE POLICY users_self_update ON users
            FOR UPDATE TO journal_app
            USING      (id = (SELECT NULLIF(current_setting('app.current_user_id', true), '')::uuid))
            WITH CHECK (id = (SELECT NULLIF(current_setting('app.current_user_id', true), '')::uuid))
        """  # noqa: E501
    )

    op.execute("REVOKE INSERT, DELETE ON users FROM journal_app")


def downgrade() -> None:
    op.execute("GRANT INSERT, DELETE ON users TO journal_app")
    op.execute("DROP POLICY IF EXISTS users_self_update ON users")
    op.execute("DROP POLICY IF EXISTS users_self_read ON users")
    op.execute("ALTER TABLE users NO FORCE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE users DISABLE ROW LEVEL SECURITY")
