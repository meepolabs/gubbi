"""Grant journal_admin admin option on journal_app.

PG16 tightened CREATEROLE semantics: a role with CREATEROLE can no longer
alter another role's password unless it has ADMIN OPTION on that role or
created it. Migration 0002 creates journal_app and journal_admin as peers
(neither creates the other), so journal_admin cannot rotate journal_app's
password without the superuser path.

After this migration, journal_admin has ADMIN OPTION on journal_app and
can run `ALTER ROLE journal_app WITH PASSWORD ...` without needing the
superuser. journal_admin rotating its own password was already allowed
(self-password changes do not require ADMIN OPTION).

Idempotent: GRANT ... WITH ADMIN OPTION is a no-op if already granted.
Downgrade revokes only the admin option; the membership grant itself is
not added here so there is no membership to revoke.
"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "0009_grant_admin_journal_app"
down_revision = "0008_drop_plaintext_columns"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # PG16+ rejects re-granting ADMIN OPTION back to the original grantor.
    # If the bootstrap already set up this grant (e.g. testbench init.sh as
    # superuser), skip the no-op the docstring promised was safe.
    op.execute(
        """
        DO $do$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_auth_members am
                JOIN pg_roles r1 ON am.member = r1.oid
                JOIN pg_roles r2 ON am.roleid = r2.oid
                WHERE r1.rolname = 'journal_admin'
                  AND r2.rolname = 'journal_app'
                  AND am.admin_option = true
            ) THEN
                GRANT journal_app TO journal_admin WITH ADMIN OPTION;
            END IF;
        END
        $do$;
        """
    )


def downgrade() -> None:
    # Match the docstring contract: this migration only ever ADDS the
    # admin option on an existing or pre-seeded grant; it does not add the
    # underlying membership. Revoking the membership here would orphan
    # any pre-seeded grant (e.g. the testbench init.sh `GRANT journal_app
    # TO journal_admin WITH ADMIN OPTION`) and leave journal_admin unable
    # to rotate journal_app's password from a downgrade'd state.
    op.execute("REVOKE ADMIN OPTION FOR journal_app FROM journal_admin")
