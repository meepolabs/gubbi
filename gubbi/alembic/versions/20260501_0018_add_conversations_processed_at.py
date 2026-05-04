from alembic import op

revision = "0018_add_conversations_processed_at"
down_revision = "0017_add_platform_id_to_conversations"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE conversations ADD COLUMN processed_at TIMESTAMPTZ")


def downgrade() -> None:
    op.execute("ALTER TABLE conversations DROP COLUMN IF EXISTS processed_at")
