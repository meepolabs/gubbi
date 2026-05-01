"""Add platform and platform_id columns to conversations table for deduplication.

Adds:
- conversations.platform TEXT (nullable) -- e.g. "chatgpt", "claude"
- conversations.platform_id TEXT (nullable) -- external conversation ID
- UNIQUE INDEX on (user_id, platform, platform_id) WHERE platform_id IS NOT NULL
"""

from alembic import op

revision = "0017_add_platform_id_to_conversations"
down_revision = "0016_audit_log_content_hash_dedup"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE conversations ADD COLUMN platform TEXT")
    op.execute("ALTER TABLE conversations ADD COLUMN platform_id TEXT")
    op.execute(
        """
        CREATE UNIQUE INDEX idx_conv_platform_dedup
            ON conversations (user_id, platform, platform_id)
            WHERE platform_id IS NOT NULL
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_conv_platform_dedup")
    op.execute("ALTER TABLE conversations DROP COLUMN IF EXISTS platform_id")
    op.execute("ALTER TABLE conversations DROP COLUMN IF EXISTS platform")
