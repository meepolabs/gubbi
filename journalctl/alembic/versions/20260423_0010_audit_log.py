"""Append-only audit_log table for privileged actions (DEC-061).

Implements DEC-061 and TASK-02.20. Introduces a tamper-resistant audit_log
table capturing every privileged action: admin role use, key rotation, manual
DB writes, user deletions, subscription overrides, tenant provisioning, etc.

Immutability contract:
- Trigger-based: the audit_log_immutable() function fires BEFORE UPDATE and
  BEFORE DELETE, raising an exception unconditionally. This fires even under
  journal_admin (BYPASSRLS, superuser-adjacent) because triggers are not
  bypassed by BYPASSRLS or superuser unless the trigger itself is dropped.
- journal_app receives INSERT only -- no SELECT, no UPDATE, no DELETE.
- journal_admin receives SELECT + INSERT -- no UPDATE, no DELETE.
  The triggers enforce this at the DB layer even if grant logic is ever
  accidentally widened.

The default privileges set in migration 0002 grant SELECT/INSERT/UPDATE/DELETE
to journal_app and ALL PRIVILEGES to journal_admin on all tables. We must
explicitly REVOKE the extra privileges on audit_log after creation so that
the permission model is narrow-by-design rather than relying on the trigger
alone. Defense in depth: trigger blocks the SQL path, REVOKE blocks the
privilege-check path.

Retention: rows are never deleted. The 7-year retention target is met by the
daily pg_dump backup path (already in place). No cron or partition scheme is
required at v1 scale (~35 MB/year uncompressed at 1k events/day).

Downgrade order matters: triggers -> function -> indexes -> table.
Dropping the table while triggers exist raises an error; dropping triggers
before the function is safe because PG holds a dependency reference.
"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "0010_audit_log"
down_revision = "0009_grant_admin_journal_app"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create audit_log table, indexes, immutability triggers, and permissions."""
    op.execute(
        """
        CREATE TABLE audit_log (
            id          BIGSERIAL PRIMARY KEY,
            occurred_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            actor_type  TEXT NOT NULL
                            CHECK (actor_type IN ('user', 'admin', 'system', 'founder')),
            actor_id    TEXT NOT NULL,
            action      TEXT NOT NULL,
            target_type TEXT,
            target_id   TEXT,
            reason      TEXT,
            metadata    JSONB NOT NULL DEFAULT '{}'::jsonb,
            ip_address  INET,
            user_agent  TEXT
        )
        """
    )

    op.execute("CREATE INDEX idx_audit_log_occurred_at ON audit_log(occurred_at DESC)")
    op.execute("CREATE INDEX idx_audit_log_actor_id ON audit_log(actor_id)")
    op.execute("CREATE INDEX idx_audit_log_action ON audit_log(action)")
    op.execute("CREATE INDEX idx_audit_log_target ON audit_log(target_type, target_id)")

    # Immutability trigger function -- fires for both UPDATE and DELETE.
    # Using $$ quoting keeps the body portable; no substitution occurs inside $$.
    op.execute(
        """
        CREATE OR REPLACE FUNCTION audit_log_immutable() RETURNS trigger AS $$
        BEGIN
            RAISE EXCEPTION 'audit_log rows are append-only';
        END;
        $$ LANGUAGE plpgsql
        """
    )

    op.execute(
        """
        CREATE TRIGGER trg_audit_log_no_update
            BEFORE UPDATE ON audit_log
            FOR EACH ROW EXECUTE FUNCTION audit_log_immutable()
        """
    )
    op.execute(
        """
        CREATE TRIGGER trg_audit_log_no_delete
            BEFORE DELETE ON audit_log
            FOR EACH ROW EXECUTE FUNCTION audit_log_immutable()
        """
    )

    # Migration 0002 default-privileges grant SELECT/INSERT/UPDATE/DELETE to
    # journal_app and ALL PRIVILEGES to journal_admin on every new table.
    # Revoke the extras so that the permission model is narrow even without the
    # trigger. journal_app gets INSERT only; journal_admin gets SELECT + INSERT.
    op.execute("REVOKE ALL ON audit_log FROM journal_app")
    op.execute("GRANT INSERT ON audit_log TO journal_app")

    op.execute("REVOKE ALL ON audit_log FROM journal_admin")
    op.execute("GRANT SELECT, INSERT ON audit_log TO journal_admin")

    # audit_log uses BIGSERIAL; journal_app does not need the sequence directly
    # (the DEFAULT handles it). No GRANT on the sequence is needed.


def downgrade() -> None:
    """Drop triggers, function, and table in dependency order."""
    op.execute("DROP TRIGGER IF EXISTS trg_audit_log_no_update ON audit_log")
    op.execute("DROP TRIGGER IF EXISTS trg_audit_log_no_delete ON audit_log")
    op.execute("DROP FUNCTION IF EXISTS audit_log_immutable()")
    op.execute("DROP INDEX IF EXISTS idx_audit_log_occurred_at")
    op.execute("DROP INDEX IF EXISTS idx_audit_log_actor_id")
    op.execute("DROP INDEX IF EXISTS idx_audit_log_action")
    op.execute("DROP INDEX IF EXISTS idx_audit_log_target")
    op.execute("DROP TABLE IF EXISTS audit_log")
