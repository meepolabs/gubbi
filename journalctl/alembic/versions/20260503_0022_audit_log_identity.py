"""Shift audit_log.id from BIGSERIAL to GENERATED ALWAYS AS IDENTITY.

BIGSERIAL secretly creates an owned sequence and sets a default via nextval().
PostgreSQL does not offer a one-step serial-to-identity conversion, so this
migration rewrites the column definition in-place:

Upgrade:
  1. Drop the DEFAULT expression so the old sequence (audit_log_id_seq) is
     unlinked from the column.
  2. Compute next id via coalesce(max(id),0)+1 to pick up whatever data
     already exists.
  3. ALTER COLUMN ... SET GENERATED ALWAYS AS IDENTITY RESTART <next>.
  4. Drop the now-unreferenced BIGSERIAL sequence.

Downgrade:
  1. ALTER COLUMN drop identity (the column becomes a plain BIGINT).
  2. Re-create the audit_log_id_seq and make it the default, resetting so
     nextval picks up past the existing max id.
  3. Set DEFAULT to keep the same contract old code relied on.

This is metadata-only DDL: no row movement, brief ACCESS EXCLUSIVE lock only
on the ALTER COLUMN step (PostgreSQL rewrites the column typmod in-place).
"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "0022_audit_log_identity"
down_revision = "0021_perf_audit_log_actor_idx"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()

    # Step 1: Drop the default so the column no longer references audit_log_id_seq.
    op.execute("ALTER TABLE audit_log ALTER COLUMN id DROP DEFAULT")

    # Steps 2-4: Compute next id and re-add as IDENTITY.
    # We must use a PL/pgSQL DO block because RESTART cannot take a subquery.
    do_sql = (
        "DO $$ DECLARE nxt BIGINT; BEGIN SELECT COALESCE(MAX(id), 0) + 1 INTO nxt "  # noqa: E501, S608
        "FROM audit_log; EXECUTE format("
        "'ALTER TABLE audit_log ALTER COLUMN id SET GENERATED ALWAYS AS IDENTITY "
        "(RESTART WITH %s)', nxt); DROP SEQUENCE IF EXISTS audit_log_id_seq; END $$;"  # noqa: E501, S608
    )
    conn.connection.execute(do_sql)


def downgrade() -> None:
    conn = op.get_bind()

    # Step 1: Drop the IDENTITY constraint.
    op.execute("ALTER TABLE audit_log ALTER COLUMN id DROP IDENTITY IF EXISTS")

    # Steps 2-3: Recreate sequence as BIGSERIAL-style and wire it back as default.
    # Use PL/pgSQL so the RESTART value is self-computed from existing data.
    downgrade_sql = (
        "DO $$ DECLARE nxt BIGINT; BEGIN CREATE SEQUENCE audit_log_id_seq OWNED BY audit_log.id; "  # noqa: E501, S608
        "SELECT COALESCE(MAX(id), 0) + 1 INTO nxt FROM audit_log; "
        "ALTER SEQUENCE audit_log_id_seq RESTART WITH nxt; "
        """EXECUTE format('ALTER TABLE audit_log ALTER COLUMN id SET DEFAULT nextval(''audit_log_id_seq''));"""  # noqa: E501, S608
        " END $$;"
    )
    conn.connection.execute(downgrade_sql)
