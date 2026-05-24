"""Create journal_app and journal_admin Postgres roles.

Provision two database roles: journal_app (runtime, RLS-enforced) and
journal_admin (BYPASSRLS, migration/deploy role). No passwords are set
here — authentication is handled by DSN-provisioned users via Doppler.
Idempotent: checks pg_roles before creating, skips existing roles.
"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "0002_create_db_roles"
down_revision = "0001_baseline"
branch_labels = None
depends_on = None


def _quoted_db_name() -> str:
    """Return the current database name wrapped in safe double quotes.

    Done in Python (not SQL format()) to avoid driver-level `%` escaping
    ambiguity between SQLAlchemy text() + psycopg paramstyle.
    """
    bind = op.get_bind()
    name = bind.execute(sa.text("SELECT current_database()")).scalar_one()
    escaped = str(name).replace('"', '""')
    return f'"{escaped}"'


def upgrade() -> None:
    """Create journal_app and journal_admin roles with privilege grants."""
    db = _quoted_db_name()

    # Create roles if they don't exist (idempotent for dev re-runs)
    op.execute(
        sa.text(
            """
            DO $$
            BEGIN
                IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'journal_app') THEN
                    CREATE ROLE journal_app LOGIN;
                END IF;
                IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'journal_admin') THEN
                    CREATE ROLE journal_admin LOGIN BYPASSRLS CREATEROLE;
                END IF;
            END $$;
            """
        )
    )

    # journal_app — runtime. No BYPASSRLS. Read/write on tables + sequences.
    op.execute(f"GRANT CONNECT ON DATABASE {db} TO journal_app")
    op.execute("GRANT USAGE ON SCHEMA public TO journal_app")
    op.execute("GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO journal_app")
    op.execute("GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO journal_app")

    # Default privileges for future tables/sequences created by the migrator
    op.execute(
        "ALTER DEFAULT PRIVILEGES IN SCHEMA public "
        "GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO journal_app"
    )
    op.execute(
        "ALTER DEFAULT PRIVILEGES IN SCHEMA public "
        "GRANT USAGE, SELECT ON SEQUENCES TO journal_app"
    )

    # journal_admin — all privileges, BYPASSRLS attribute already set at creation
    op.execute(f"GRANT ALL PRIVILEGES ON DATABASE {db} TO journal_admin")
    op.execute("GRANT ALL PRIVILEGES ON SCHEMA public TO journal_admin")
    op.execute("GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO journal_admin")
    op.execute("GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO journal_admin")

    # Default privileges for tables/sequences created AFTER this migration.
    # Without these, tables added by later migrations (e.g. users in 0003) are
    # created owned by the migration-running role and journal_admin cannot
    # access them until an explicit grant runs. That breaks 02.14 backfill
    # (runs as journal_admin) and every admin-pool test fixture.
    op.execute(
        "ALTER DEFAULT PRIVILEGES IN SCHEMA public "
        "GRANT ALL PRIVILEGES ON TABLES TO journal_admin"
    )
    op.execute(
        "ALTER DEFAULT PRIVILEGES IN SCHEMA public "
        "GRANT ALL PRIVILEGES ON SEQUENCES TO journal_admin"
    )


def downgrade() -> None:
    """Revoke privileges and drop journal_app and journal_admin roles."""
    db = _quoted_db_name()

    # Revoke default privileges first (they don't cascade on DROP ROLE)
    op.execute(
        "ALTER DEFAULT PRIVILEGES IN SCHEMA public "
        "REVOKE SELECT, INSERT, UPDATE, DELETE ON TABLES FROM journal_app"
    )
    op.execute(
        "ALTER DEFAULT PRIVILEGES IN SCHEMA public "
        "REVOKE USAGE, SELECT ON SEQUENCES FROM journal_app"
    )
    op.execute(
        "ALTER DEFAULT PRIVILEGES IN SCHEMA public "
        "REVOKE ALL PRIVILEGES ON TABLES FROM journal_admin"
    )
    op.execute(
        "ALTER DEFAULT PRIVILEGES IN SCHEMA public "
        "REVOKE ALL PRIVILEGES ON SEQUENCES FROM journal_admin"
    )

    # journal_app revokes
    op.execute("REVOKE ALL PRIVILEGES ON ALL TABLES IN SCHEMA public FROM journal_app")
    op.execute("REVOKE ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public FROM journal_app")
    op.execute("REVOKE ALL PRIVILEGES ON SCHEMA public FROM journal_app")
    op.execute(f"REVOKE ALL PRIVILEGES ON DATABASE {db} FROM journal_app")

    # journal_admin revokes
    op.execute("REVOKE ALL PRIVILEGES ON ALL TABLES IN SCHEMA public FROM journal_admin")
    op.execute("REVOKE ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public FROM journal_admin")
    op.execute("REVOKE ALL PRIVILEGES ON SCHEMA public FROM journal_admin")
    op.execute(f"REVOKE ALL PRIVILEGES ON DATABASE {db} FROM journal_admin")

    op.execute("DROP ROLE IF EXISTS journal_app")
    op.execute("DROP ROLE IF EXISTS journal_admin")
