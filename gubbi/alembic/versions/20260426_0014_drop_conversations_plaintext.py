"""Drop plaintext conversation title/summary after encrypted backfill.

Order matters: encrypted columns are promoted to NOT NULL first so any
unbackfilled row fails loudly before plaintext columns are dropped.
"""

from alembic import op

revision = "0014_drop_conversations_plaintext"
down_revision = "0013_drop_search_text_encrypt_conversations"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE conversations ALTER COLUMN title_encrypted SET NOT NULL")
    op.execute("ALTER TABLE conversations ALTER COLUMN title_nonce SET NOT NULL")
    op.execute("ALTER TABLE conversations ALTER COLUMN summary_encrypted SET NOT NULL")
    op.execute("ALTER TABLE conversations ALTER COLUMN summary_nonce SET NOT NULL")

    op.execute("ALTER TABLE conversations DROP COLUMN title")
    op.execute("ALTER TABLE conversations DROP COLUMN summary")


def downgrade() -> None:
    op.execute("ALTER TABLE conversations ADD COLUMN title TEXT NOT NULL DEFAULT ''")
    op.execute("ALTER TABLE conversations ADD COLUMN summary TEXT NOT NULL DEFAULT ''")

    op.execute("ALTER TABLE conversations ALTER COLUMN title_encrypted DROP NOT NULL")
    op.execute("ALTER TABLE conversations ALTER COLUMN title_nonce DROP NOT NULL")
    op.execute("ALTER TABLE conversations ALTER COLUMN summary_encrypted DROP NOT NULL")
    op.execute("ALTER TABLE conversations ALTER COLUMN summary_nonce DROP NOT NULL")
