"""Create an idempotent ``otel_ro`` Postgres role for read-only observability.

OpenTelemetry collectors (and the ``otelcol-contrib`` postgres receiver in
particular) need access to a handful of monitoring views (``pg_stat_activity``,
``pg_stat_database``, ``pg_stat_replication``, ...) to scrape DB telemetry.
Postgres ships a built-in role -- ``pg_monitor`` -- whose sole purpose is to
grant exactly that visibility.

This migration provisions a dedicated NOLOGIN role called ``otel_ro`` and
grants it ``pg_monitor``.  ``NOLOGIN`` keeps the role inert until an
operator grants login privileges (or attaches a password) at deployment
time; the role exists purely as a stable grant target the rest of the
observability config can reference.

Idempotency: the ``DO $$ ... IF NOT EXISTS`` block makes the role creation
re-runnable.  ``GRANT pg_monitor TO otel_ro`` is itself idempotent in
Postgres (re-granting the same role membership is a no-op), so no guard
is needed there.
"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "0026_otel_ro_role"
down_revision = "0025_messages_search_vector_gin"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create the ``otel_ro`` role (idempotent) and grant ``pg_monitor``."""
    op.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'otel_ro') THEN
                CREATE ROLE otel_ro NOLOGIN;
            END IF;
        END $$;
        """
    )
    op.execute("GRANT pg_monitor TO otel_ro;")


def downgrade() -> None:
    """Revoke ``pg_monitor`` and drop the ``otel_ro`` role (idempotent)."""
    op.execute("REVOKE pg_monitor FROM otel_ro;")
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'otel_ro') THEN
                DROP ROLE otel_ro;
            END IF;
        END $$;
        """
    )
