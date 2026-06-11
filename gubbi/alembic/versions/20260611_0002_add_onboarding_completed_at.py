"""Add nullable onboarding_completed_at to users.

Existing rows get NULL, which means "onboarding not completed". The
column is nullable with no default, so this is an instant metadata-only
DDL on PostgreSQL (no table rewrite).
"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "0002_add_onboarding_completed_at"
down_revision = "0001_squashed_baseline"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add the nullable onboarding_completed_at column to users."""
    op.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS onboarding_completed_at timestamptz")


def downgrade() -> None:
    """Drop the onboarding_completed_at column from users."""
    op.execute("ALTER TABLE users DROP COLUMN IF EXISTS onboarding_completed_at")
