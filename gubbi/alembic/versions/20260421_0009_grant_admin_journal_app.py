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
    op.execute("GRANT journal_app TO journal_admin WITH ADMIN OPTION")


def downgrade() -> None:
    op.execute("REVOKE ADMIN OPTION FOR journal_app FROM journal_admin")
    op.execute("REVOKE journal_app FROM journal_admin")
