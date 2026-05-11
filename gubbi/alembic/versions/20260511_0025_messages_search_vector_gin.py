"""Add GIN index on messages.search_vector for FTS path parity with entries/conversations.

The ``messages.search_vector`` tsvector column has been written by
``conversations.append_messages`` since migration 0013, but the GIN
index counterpart was never created.  ``to_tsquery``-style scans
against messages currently degrade to a sequential scan over the
whole table; on small datasets that is invisible, but the cost grows
linearly with message volume.  Adding the GIN index now -- ahead of
the multi-tenant rollout -- gives messages FTS the same plan shape
that ``entries`` (``idx_entries_fts``) and ``conversations``
(``idx_conv_fts``) already enjoy.

Naming: ``idx_messages_fts`` matches the sibling indexes on entries
and conversations.

Decision: keep the column, add the index.  An alternative discussed
in backlog triage was to drop the column outright (messages FTS is
not yet a product feature).  The keep-and-index path was chosen so
the future search-over-messages feature does not require another
schema migration; the index cost is bounded by the row count.

Concurrency: this migration uses a plain ``CREATE INDEX`` (not
``CREATE INDEX CONCURRENTLY``).  The messages table has no production
volume at the time of this migration -- pre-launch -- so the table
lock taken by a regular CREATE INDEX is acceptable.  Existing
migrations in this tree all use plain CREATE INDEX for the same
reason (see 20260419_0006, 20260426_0013).  Switching to CONCURRENTLY
would also require running outside an Alembic transaction, which the
env wires unconditionally.

``IF NOT EXISTS`` makes the upgrade idempotent so reruns after a
partial-apply failure complete cleanly.
"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "0025_messages_search_vector_gin"
down_revision = "0024_extraction_jobs_period_start"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create GIN index on ``messages.search_vector``."""
    op.execute("CREATE INDEX IF NOT EXISTS idx_messages_fts ON messages USING GIN (search_vector)")


def downgrade() -> None:
    """Drop the GIN index."""
    op.execute("DROP INDEX IF EXISTS idx_messages_fts")
