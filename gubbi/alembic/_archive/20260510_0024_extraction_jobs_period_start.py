"""Add period_start column to extraction_jobs.

Records the billing period (first day of UTC month) at job creation time.
The worker reads this column to ensure the budget delta lands in the same
bucket that ingest pre-charged, regardless of when the worker runs.

Upgrade:
  1. ADD COLUMN period_start DATE NOT NULL DEFAULT (first day of current UTC month).
  2. DROP DEFAULT -- future rows must supply the value explicitly at INSERT.

The DEFAULT-then-DROP pattern avoids backfill complexity.  No live rows
exist at the time of this migration (no production deployment since M3),
so the DEFAULT is only there to satisfy NOT NULL during the ALTER TABLE.

Downgrade:
  Drops the column.
"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "0024_extraction_jobs_period_start"
down_revision = "0023_extraction_jobs_relocate"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add period_start DATE NOT NULL to extraction_jobs."""
    op.execute(
        sa.text(
            "ALTER TABLE extraction_jobs"
            " ADD COLUMN period_start DATE NOT NULL"
            " DEFAULT (date_trunc('month', now() AT TIME ZONE 'UTC')::date)"
        )
    )
    op.execute(sa.text("ALTER TABLE extraction_jobs ALTER COLUMN period_start DROP DEFAULT"))


def downgrade() -> None:
    """Remove period_start column."""
    op.execute(sa.text("ALTER TABLE extraction_jobs DROP COLUMN IF EXISTS period_start"))
