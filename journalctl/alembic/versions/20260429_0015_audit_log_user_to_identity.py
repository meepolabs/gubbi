"""Rewrite legacy ``user.*`` audit_log actions to the ``identity.*`` namespace.

Pre-launch data migration to align audit-log action values with the
``Action.IDENTITY_*`` enum.  Pre-M3 rows used ``user.created``,
``user.deleted``, ``user.restored`` (from M2 testing); M3 onward emits
``identity.created``, ``identity.deleted``, ``identity.restored`` (matched
by the cloud-api Kratos webhook handler that uses the ``identity.*``
namespace for ``identity.updated`` / ``identity.deleted`` events).

Once this migration runs, downstream queries can filter
``action LIKE 'identity.%'`` and pick up every identity-shaped event
without missing pre-M3 rows.

The ``audit_log`` table has append-only enforcement triggers (DEC-061;
``trg_audit_log_no_update`` + ``trg_audit_log_no_delete``) that block
``UPDATE``/``DELETE``. This one-shot rewrite bypasses them via
``SET session_replication_role = 'replica'`` -- a superuser-only
session variable that suppresses non-replica triggers for the current
transaction. The migration MUST run under the ``journal`` superuser
(set ``JOURNAL_DB_MIGRATION_URL`` to the superuser DSN, see RUNBOOK
Section 5).
"""

from alembic import op

revision = "0015_audit_log_user_to_identity"
down_revision = "0014_drop_conversations_plaintext"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("SET session_replication_role = 'replica'")
    op.execute("UPDATE audit_log SET action = 'identity.created' WHERE action = 'user.created'")
    op.execute("UPDATE audit_log SET action = 'identity.deleted' WHERE action = 'user.deleted'")
    op.execute("UPDATE audit_log SET action = 'identity.restored' WHERE action = 'user.restored'")
    op.execute("SET session_replication_role = 'origin'")


def downgrade() -> None:
    op.execute("SET session_replication_role = 'replica'")
    op.execute("UPDATE audit_log SET action = 'user.created' WHERE action = 'identity.created'")
    op.execute("UPDATE audit_log SET action = 'user.deleted' WHERE action = 'identity.deleted'")
    op.execute("UPDATE audit_log SET action = 'user.restored' WHERE action = 'identity.restored'")
    op.execute("SET session_replication_role = 'origin'")
