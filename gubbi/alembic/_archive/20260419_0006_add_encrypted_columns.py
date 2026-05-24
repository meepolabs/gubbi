"""Add encrypted columns + switch entries.search_vector to search_text derivation.

Encrypted columns: content_encrypted, content_nonce, reasoning_encrypted,
reasoning_nonce (entries) and content_encrypted, content_nonce, search_text
(messages). All nullable -- NOT NULL enforcement lands in 02.14 /
0007 after backfill so this migration is safe to apply on empty or
populated databases.

Why app-layer AES-256-GCM (Q11): database-layer encryption (pgcrypto
 pgp_sym_encrypt, TDE, or cloud KMS) encrypts at rest but the
 ciphertext is visible in query plans, logs, and backups. By encrypting
 at the app layer with AES-256-GCM (AEAD: authenticated encryption),
 plaintext never touches the database. The database only ever sees
 ciphertext + AEAD nonce, which leaks no metadata about the plaintext.

Why BYTEA + nonce split (not a single BLOB): AEAD schemes (AES-256-GCM,
 XChaCha20-Poly1305) require a unique nonce per encryption. Storing the
 nonce alongside the ciphertext in a single BYTEA column forces manual
 packing/unpacking (e.g. nonce || ciphertext). A dedicated 12-byte
 (GCM) or 24-byte (XChaCha20) column is explicit, indexable (for nonce
 uniqueness checks), and survives column reordering. A future version
 may add a UNIQUE constraint on (content_encrypted, content_nonce) to
 prevent nonce reuse -- a catastrophic AEAD failure if violated.

Historical note: this migration introduced ``search_text`` as plaintext
 and the repository wrote full content verbatim, not a subset. That
 design was later corrected in migration 0013, which drops
 ``entries.search_text`` / ``messages.search_text`` and writes
 ``search_vector`` from ephemeral plaintext parameters via
 ``to_tsvector('english', $N)`` so prose is not persisted in the DB.

Explicit window risk: between this migration (02.12) and the 02.14
backfill migration (0007), entries.search_vector will be empty because
search_text is NULL on existing rows. Deploy 02.12 + 02.14 together in
the same maintenance window. Direct reads of content/reasoning continue
to work because those columns are untouched by this migration.

Lock duration on large tables: step 4 (``ADD COLUMN search_vector ...
GENERATED ALWAYS AS ... STORED``) rewrites every row of entries to
compute the new generated value. At founder scale (~600 entries) this
is instant; at 5M rows it holds ``AccessExclusiveLock`` for minutes.
The GIN recreation (step 5) also holds ``ShareLock`` blocking writes;
it is not done ``CONCURRENTLY`` because Alembic wraps the migration in
a transaction and ``CREATE INDEX CONCURRENTLY`` cannot run inside one.
At multi-tenant scale, schedule this migration during a maintenance
window or split the index recreation into a separate migration that
drives its own transaction.

Nonce length invariants (GCM = 12 bytes): CHECK constraints on every
nonce column block a buggy writer from landing the wrong shape and
turning a write bug into a decryption-time outage. Relax to an IN list
when XChaCha20-Poly1305 support (24-byte nonces) is introduced; the
cipher layer already distinguishes via the version byte in nonce[0].
"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "0006_add_encrypted_columns"
down_revision = "0005_enable_rls"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add encrypted columns, switch search_vector derivation to search_text.

    Upgrade path (one step per numbered comment below):
      1. Add nullable encrypted columns to entries.
      2. Add nullable encrypted columns to messages.
      3. Drop the old entries.search_vector generated column (also drops
         idx_entries_fts automatically since it indexes that column).
      4. Re-add entries.search_vector with the new derivation from
         search_text (NULL-safe: coalesce to empty string).
      5. Re-create the GIN index on the new column.
      6. Add CHECK constraints pinning nonce columns to 12 bytes
         (GCM invariant; the cipher layer contract).

    Column ADD/DROP omit ``IF NOT EXISTS`` / ``IF EXISTS`` intentionally:
    they should fail loudly if the schema has drifted from expectations.
    The GIN index uses ``IF NOT EXISTS`` for safe re-runs.
    """
    # -- 1. entries encrypted columns (nullable; NOT NULL enforcement in 0007) --
    op.execute("ALTER TABLE entries ADD COLUMN content_encrypted BYTEA")
    op.execute("ALTER TABLE entries ADD COLUMN content_nonce BYTEA")
    op.execute("ALTER TABLE entries ADD COLUMN reasoning_encrypted BYTEA")
    op.execute("ALTER TABLE entries ADD COLUMN reasoning_nonce BYTEA")
    op.execute("ALTER TABLE entries ADD COLUMN search_text TEXT")

    # -- 2. messages encrypted columns (nullable; NOT NULL enforcement in 0007) --
    op.execute("ALTER TABLE messages ADD COLUMN content_encrypted BYTEA")
    op.execute("ALTER TABLE messages ADD COLUMN content_nonce BYTEA")
    op.execute("ALTER TABLE messages ADD COLUMN search_text TEXT")

    # -- 3. Drop old search_vector (also drops idx_entries_fts automatically) --
    op.execute("ALTER TABLE entries DROP COLUMN search_vector")

    # -- 4. Re-add search_vector derived from the new search_text column --
    op.execute(
        "ALTER TABLE entries ADD COLUMN search_vector tsvector"
        " GENERATED ALWAYS AS"
        " (to_tsvector('english', coalesce(search_text, ''))) STORED"
    )

    # -- 5. Re-create the GIN index on the new column --
    op.execute("CREATE INDEX IF NOT EXISTS idx_entries_fts " "ON entries USING GIN (search_vector)")

    # -- 6. Nonce-length CHECK constraints (GCM invariant; 12 bytes). NULL is
    # permitted because backfill in 02.14 fills nonces row-by-row; the check
    # only fires when a non-NULL value is present. Relax to IN (12, 24) when
    # XChaCha20-Poly1305 support ships; the cipher version byte in nonce[0]
    # already distinguishes the two schemes.
    op.execute(
        "ALTER TABLE entries ADD CONSTRAINT entries_content_nonce_len "
        "CHECK (content_nonce IS NULL OR octet_length(content_nonce) = 12)"
    )
    op.execute(
        "ALTER TABLE entries ADD CONSTRAINT entries_reasoning_nonce_len "
        "CHECK (reasoning_nonce IS NULL OR octet_length(reasoning_nonce) = 12)"
    )
    op.execute(
        "ALTER TABLE messages ADD CONSTRAINT messages_content_nonce_len "
        "CHECK (content_nonce IS NULL OR octet_length(content_nonce) = 12)"
    )


def downgrade() -> None:
    """Reverse the upgrade: restore old search_vector, drop encrypted columns.

    Downgrade path (reverse order of upgrade):
      1. Drop nonce CHECK constraints (must precede column DROP).
      2. Drop the GIN index (symmetry with upgrade step 5).
      3. Drop the new search_vector, re-add it with the old derivation
         ``to_tsvector('english', coalesce(content, '') || ' ' ||
         coalesce(reasoning, ''))`` -- restores FTS on the original data.
      4. Re-create the GIN index on the restored column.
      5. Drop the new columns on messages.
      6. Drop the new columns on entries.

    Column DROP uses ``IF EXISTS`` so a partial-upgrade rollback is robust
    (e.g. if 02.14 ran after this migration, its backfilled columns would
    already exist; without IF EXISTS the drop would fail).

    CAVEAT: rolling back AFTER 02.14 has run leaves full-text search
    broken for rows whose ``content`` / ``reasoning`` were cleared by the
    post-backfill NOT NULL flip -- the restored generated column would
    compute empty tsvectors for those rows. The backfill data itself is
    still present in ``content_encrypted``; the downgrade drops the
    encrypted columns too, so on a cleared-plaintext DB, downgrade is
    NOT reversible to a searchable state. Take a backup before running
    downgrade against a post-backfill DB.
    """
    # -- 1. Drop nonce CHECK constraints (must precede dropping the columns) --
    op.execute("ALTER TABLE entries DROP CONSTRAINT IF EXISTS entries_content_nonce_len")
    op.execute("ALTER TABLE entries DROP CONSTRAINT IF EXISTS entries_reasoning_nonce_len")
    op.execute("ALTER TABLE messages DROP CONSTRAINT IF EXISTS messages_content_nonce_len")

    # -- 2. Drop GIN index (symmetry with upgrade step 5) --
    op.execute("DROP INDEX IF EXISTS idx_entries_fts")

    # -- 3. entries: drop new search_vector, re-add with old derivation --
    op.execute("ALTER TABLE entries DROP COLUMN IF EXISTS search_vector")
    op.execute(
        "ALTER TABLE entries ADD COLUMN search_vector tsvector"
        " GENERATED ALWAYS AS"
        " (to_tsvector('english',"
        " coalesce(content, '') || ' ' || coalesce(reasoning, ''))) STORED"
    )

    # -- 4. Re-create GIN index on the restored column --
    op.execute("CREATE INDEX IF NOT EXISTS idx_entries_fts " "ON entries USING GIN (search_vector)")

    # -- 5. Drop messages new columns --
    op.execute("ALTER TABLE messages DROP COLUMN IF EXISTS search_text")
    op.execute("ALTER TABLE messages DROP COLUMN IF EXISTS content_nonce")
    op.execute("ALTER TABLE messages DROP COLUMN IF EXISTS content_encrypted")

    # -- 6. Drop entries new columns --
    op.execute("ALTER TABLE entries DROP COLUMN IF EXISTS search_text")
    op.execute("ALTER TABLE entries DROP COLUMN IF EXISTS reasoning_nonce")
    op.execute("ALTER TABLE entries DROP COLUMN IF EXISTS reasoning_encrypted")
    op.execute("ALTER TABLE entries DROP COLUMN IF EXISTS content_nonce")
    op.execute("ALTER TABLE entries DROP COLUMN IF EXISTS content_encrypted")
