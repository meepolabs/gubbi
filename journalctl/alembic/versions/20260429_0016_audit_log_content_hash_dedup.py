"""Partial unique index on audit_log to close the content-hash dedup race.

The cloud-api Kratos webhook handler dedups ``identity.updated`` re-deliveries
on ``(target_id, action, metadata->>'content_hash')`` to avoid duplicate audit
rows for re-deliveries of the same diff.  An application-layer
``SELECT``-then-``INSERT`` inside a transaction narrows the race window but
does not close it for concurrent connections; this partial unique index
closes it fully and lets cloud-api use ``INSERT ... ON CONFLICT DO NOTHING``.

Partial because most audit rows do not carry a ``content_hash`` (only
``identity.updated`` does in M3); a non-partial unique index would over-
constrain unrelated event types and reject legitimate duplicates of, for
example, ``user.created`` JIT-provision audits that share ``target_id`` /
``action`` with the Kratos webhook row but no hash.
"""

from alembic import op

revision = "0016_audit_log_content_hash_dedup"
down_revision = "0015_audit_log_user_to_identity"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE UNIQUE INDEX audit_log_content_hash_uidx
            ON audit_log (target_id, action, (metadata->>'content_hash'))
            WHERE metadata ? 'content_hash'
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS audit_log_content_hash_uidx")
