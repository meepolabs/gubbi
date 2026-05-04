"""Drop legacy plaintext content/reasoning columns after 02.14 backfill.

Apply ONLY after running ``gubbi/scripts/backfill_encrypt.py`` on
every environment -- the script encrypts every legacy plaintext row
into the ciphertext/nonce/search_text columns added by migration 0006.
Running this migration before the backfill completes is a data-loss
path: any row whose ``content_encrypted`` is still NULL has its
plaintext dropped on the floor with no way to recover it.

Promotes ``content_encrypted`` and ``content_nonce`` (both entries and
messages) to NOT NULL. Reasoning encrypted columns stay nullable --
``entries.reasoning`` was nullable pre-02.13 and remains so post-drop.

After this migration:
- ``entries`` no longer has ``content`` or ``reasoning`` plaintext columns.
- ``messages`` no longer has ``content`` plaintext column.
- The repo layer (separate commit) must no longer reference those
  columns in INSERT/UPDATE SQL, nor in ts_headline fallbacks.

Downgrade restores the legacy columns as nullable TEXT so the schema
matches the pre-0008 shape. The plaintext DATA is gone -- the previous
backfill overwrote nothing but the downgrade itself cannot reconstruct
it. A reverse-backfill script (``backfill_decrypt.py``) would be
required to restore plaintext, and is deliberately out of 02.14 scope.
Document this caveat loudly in the docstring and in a runbook entry.
"""

from alembic import op

revision = "0008_drop_plaintext_columns"
down_revision = "0007_rls_policy_null_coalesce"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Drop legacy plaintext columns + promote encrypted columns to NOT NULL.

    Order:
    1. Promote content_encrypted + content_nonce to NOT NULL on entries.
       (reasoning_encrypted / reasoning_nonce STAY nullable -- reasoning
       was always optional.)
    2. Promote content_encrypted + content_nonce to NOT NULL on messages.
    3. Drop entries.content, entries.reasoning.
    4. Drop messages.content.

    Steps 1-2 before 3-4 because SET NOT NULL requires a full scan; we
    want any NULL-valued encrypted cell to error HERE with a clear
    message rather than silently succeed, lose plaintext, and fail later.
    """
    op.execute("ALTER TABLE entries ALTER COLUMN content_encrypted SET NOT NULL")
    op.execute("ALTER TABLE entries ALTER COLUMN content_nonce SET NOT NULL")
    op.execute("ALTER TABLE messages ALTER COLUMN content_encrypted SET NOT NULL")
    op.execute("ALTER TABLE messages ALTER COLUMN content_nonce SET NOT NULL")
    op.execute("ALTER TABLE entries DROP COLUMN content")
    op.execute("ALTER TABLE entries DROP COLUMN reasoning")
    op.execute("ALTER TABLE messages DROP COLUMN content")


def downgrade() -> None:
    """Restore plaintext columns as nullable TEXT and relax encrypted NOT NULL.

    WARNING: the plaintext DATA is unrecoverable from this migration --
    the previous upgrade dropped it. A reverse-backfill script would be
    needed to repopulate plaintext from the encrypted columns; that is
    out of 02.14 scope. This downgrade only restores the SCHEMA shape so
    earlier migrations can be rolled back cleanly.
    """
    op.execute("ALTER TABLE entries ADD COLUMN content TEXT")
    op.execute("ALTER TABLE entries ADD COLUMN reasoning TEXT")
    op.execute("ALTER TABLE messages ADD COLUMN content TEXT")
    op.execute("ALTER TABLE entries ALTER COLUMN content_encrypted DROP NOT NULL")
    op.execute("ALTER TABLE entries ALTER COLUMN content_nonce DROP NOT NULL")
    op.execute("ALTER TABLE messages ALTER COLUMN content_encrypted DROP NOT NULL")
    op.execute("ALTER TABLE messages ALTER COLUMN content_nonce DROP NOT NULL")
