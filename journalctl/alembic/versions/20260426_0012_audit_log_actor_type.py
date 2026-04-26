"""Align audit_log.actor_type CHECK with the Python whitelist.

Migration 0010 created the CHECK with ``('user', 'admin', 'system', 'founder')``.
TASK-AUTH-01 (Mode 3 hardening) added ``hydra_subject`` to the Python
``_VALID_ACTOR_TYPES`` whitelist but never widened the SQL CHECK -- so
the JIT email-collision audit write in ``middleware/auth.py`` silently
failed with ``CheckViolationError`` (swallowed by the surrounding
try/except). This migration corrects the drift.

It also drops ``founder``: a single-tenant-era value with zero call
sites in the current codebase. Mode 3 has no founder concept; if
support tooling ever needs an impersonation actor type later, add it
back in a new migration with explicit intent.

Final accepted values: ``user``, ``admin``, ``system``, ``hydra_subject``.
"""

from alembic import op

revision = "0012_audit_log_actor_type"
down_revision = "0011_cleanup_xml_spill"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE audit_log DROP CONSTRAINT IF EXISTS audit_log_actor_type_check")
    op.execute(
        "ALTER TABLE audit_log ADD CONSTRAINT audit_log_actor_type_check "
        "CHECK (actor_type IN ('user', 'admin', 'system', 'hydra_subject'))"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE audit_log DROP CONSTRAINT IF EXISTS audit_log_actor_type_check")
    op.execute(
        "ALTER TABLE audit_log ADD CONSTRAINT audit_log_actor_type_check "
        "CHECK (actor_type IN ('user', 'admin', 'system', 'founder'))"
    )
