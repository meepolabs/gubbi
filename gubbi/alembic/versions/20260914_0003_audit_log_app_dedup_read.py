"""Grant journal_app the narrow read capability the deduped audit INSERT needs.

``AUDIT_INSERT_DEDUPED_SQL`` infers its conflict target from the partial
unique index ``audit_log_content_hash_uidx``, so PostgreSQL reads
``actor_id``, ``target_kind``, ``target_id``, ``action`` and ``metadata``
while evaluating the ON CONFLICT clause. Without a read capability on those
columns every user-attributed deduped write fails with insufficient_privilege.

Executed against PostgreSQL 17, both halves below are required and neither
works alone: the column grant alone still trips the row-level security check
because the SELECT the inference clause performs has no permissive policy,
and a policy alone still trips the column ACL. Table-wide SELECT stays
denied, so ``actor_type``, ``occurred_at``, ``reason``, ``ip_address`` and
``user_agent`` remain unreadable through the app role.

The SELECT policy mirrors the ``audit_log_app_insert_self_only`` WITH CHECK
predicate exactly, so an author's read surface is precisely the set of rows
it is permitted to write. Metadata redaction in the canonical writer remains
the boundary that keeps sensitive values out of that surface.
"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "0003_audit_log_app_dedup_read"
down_revision = "0002_add_onboarding_completed_at"
branch_labels = None
depends_on = None

_GRANT_CONFLICT_TARGET_COLUMNS = (
    "GRANT SELECT (actor_id, target_kind, target_id, action, metadata) "
    "ON TABLE public.audit_log TO journal_app"
)
_REVOKE_CONFLICT_TARGET_COLUMNS = (
    "REVOKE SELECT (actor_id, target_kind, target_id, action, metadata) "
    "ON TABLE public.audit_log FROM journal_app"
)

_DROP_SELECT_POLICY = "DROP POLICY IF EXISTS audit_log_app_select_self_only ON public.audit_log"

_CREATE_SELECT_POLICY = """
CREATE POLICY audit_log_app_select_self_only ON public.audit_log
    FOR SELECT TO journal_app
    USING (
        actor_id = (SELECT NULLIF(current_setting('app.current_user_id', true), ''))
        AND actor_id <> ''
        AND actor_type = 'user'
    )
"""


def upgrade() -> None:
    """Add the column grant plus the self-only SELECT policy on audit_log."""
    op.execute(_GRANT_CONFLICT_TARGET_COLUMNS)
    op.execute(_DROP_SELECT_POLICY)
    op.execute(_CREATE_SELECT_POLICY)


def downgrade() -> None:
    """Remove the SELECT policy and the column grant."""
    op.execute(_DROP_SELECT_POLICY)
    op.execute(_REVOKE_CONFLICT_TARGET_COLUMNS)
