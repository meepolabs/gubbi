"""Drop plaintext search_text and add encrypted conversation fields.

This migration does three things:
1. Removes plaintext ``entries.search_text`` and ``messages.search_text``.
2. Converts ``entries.search_vector`` and ``conversations.search_vector``
   from GENERATED columns to regular ``tsvector`` columns populated by app SQL.
3. Adds nullable encrypted conversation title/summary columns with 12-byte
   nonce checks. NOT NULL is enforced in migration 0014 after backfill.
"""

from alembic import op

revision = "0013_drop_search_text_encrypt_conversations"
down_revision = "0012_audit_log_actor_type"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_entries_fts")
    op.execute("DROP INDEX IF EXISTS idx_conv_fts")

    op.execute("ALTER TABLE entries DROP COLUMN search_vector")
    op.execute("ALTER TABLE conversations DROP COLUMN search_vector")
    op.execute("ALTER TABLE entries DROP COLUMN search_text")
    op.execute("ALTER TABLE messages DROP COLUMN search_text")

    op.execute("ALTER TABLE entries ADD COLUMN search_vector tsvector")
    op.execute("ALTER TABLE conversations ADD COLUMN search_vector tsvector")
    op.execute("ALTER TABLE messages ADD COLUMN search_vector tsvector")

    op.execute("CREATE INDEX IF NOT EXISTS idx_entries_fts ON entries USING GIN (search_vector)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_conv_fts ON conversations USING GIN (search_vector)")

    op.execute("ALTER TABLE conversations ADD COLUMN title_encrypted BYTEA")
    op.execute("ALTER TABLE conversations ADD COLUMN title_nonce BYTEA")
    op.execute("ALTER TABLE conversations ADD COLUMN summary_encrypted BYTEA")
    op.execute("ALTER TABLE conversations ADD COLUMN summary_nonce BYTEA")

    op.execute(
        "ALTER TABLE conversations ADD CONSTRAINT conversations_title_nonce_len "
        "CHECK (title_nonce IS NULL OR octet_length(title_nonce) = 12)"
    )
    op.execute(
        "ALTER TABLE conversations ADD CONSTRAINT conversations_summary_nonce_len "
        "CHECK (summary_nonce IS NULL OR octet_length(summary_nonce) = 12)"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE conversations DROP CONSTRAINT IF EXISTS conversations_title_nonce_len")
    op.execute(
        "ALTER TABLE conversations DROP CONSTRAINT IF EXISTS conversations_summary_nonce_len"
    )

    op.execute("ALTER TABLE conversations DROP COLUMN IF EXISTS summary_nonce")
    op.execute("ALTER TABLE conversations DROP COLUMN IF EXISTS summary_encrypted")
    op.execute("ALTER TABLE conversations DROP COLUMN IF EXISTS title_nonce")
    op.execute("ALTER TABLE conversations DROP COLUMN IF EXISTS title_encrypted")

    op.execute("DROP INDEX IF EXISTS idx_entries_fts")
    op.execute("DROP INDEX IF EXISTS idx_conv_fts")

    op.execute("ALTER TABLE messages DROP COLUMN IF EXISTS search_vector")
    op.execute("ALTER TABLE entries DROP COLUMN IF EXISTS search_vector")
    op.execute("ALTER TABLE conversations DROP COLUMN IF EXISTS search_vector")

    op.execute("ALTER TABLE entries ADD COLUMN search_text TEXT")
    op.execute("ALTER TABLE messages ADD COLUMN search_text TEXT")

    op.execute(
        "ALTER TABLE entries ADD COLUMN search_vector tsvector GENERATED ALWAYS AS "
        "(to_tsvector('english', coalesce(search_text, ''))) STORED"
    )
    op.execute(
        "ALTER TABLE conversations ADD COLUMN search_vector tsvector GENERATED ALWAYS AS "
        "(to_tsvector('english', coalesce(title, '') || ' ' || coalesce(summary, ''))) STORED"
    )

    op.execute("CREATE INDEX IF NOT EXISTS idx_entries_fts ON entries USING GIN (search_vector)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_conv_fts ON conversations USING GIN (search_vector)")
