"""Add extraction provenance, near-duplicate, usage and change-tracking columns.

entries:
  ``duplicate_of_entry_id`` -- non-NULL flags a near-duplicate and links the
  entry it resembles. ``client`` -- the AI client an entry came from, in the
  ingest platform vocabulary. ``source_conversation_id`` -- the conversation an
  extracted entry was drawn from. It is distinct from ``conversation_id``,
  which marks a conversation's own summary entry and hides that row from the
  timeline and briefing.

extraction_jobs: token, call and chunk counters summed over every model call of
a job, failed attempts included.

conversations: ``extracted_content_hash`` and ``extraction_version`` record what
was last extracted, so an unchanged conversation is not extracted again.

Every column is nullable or carries a constant default, so each ADD COLUMN is a
catalog-only change on PostgreSQL 11+ and no existing row is rewritten. The two
foreign keys are ON DELETE SET NULL: removing the referenced entry or
conversation unlinks rather than blocks or cascades. Their indexes keep that
SET NULL from scanning ``entries`` on every referenced delete. A partial index
on ``(user_id, date, client)`` over live rows serves extraction dedupe, which
reads a user's live entries of given event days and clients.

All three tables already carry FORCE ROW LEVEL SECURITY and table-level grants
to journal_app and journal_admin, so the new columns inherit both; no policy or
grant changes here.

The indexes are built with plain ``CREATE INDEX``: the columns are NULL on
every existing row and ``entries`` is per-user journal data, so the build is a
short scan rather than a long write block.

No statement uses IF [NOT] EXISTS. A column or index of the same name already
present means the schema drifted from this chain, so the upgrade fails and
rolls back rather than recording the revision over an incompatible object; the
downgrade likewise drops only what it expects to find.
"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "0004_extraction_tracking_columns"
down_revision = "0003_audit_log_app_dedup_read"
branch_labels = None
depends_on = None

_EXTRACTION_JOB_COUNTERS = (
    "input_tokens",
    "output_tokens",
    "cached_input_tokens",
    "reasoning_tokens",
    "llm_calls",
    "chunks",
)

_UPGRADE_ENTRIES = """
ALTER TABLE public.entries
    ADD COLUMN duplicate_of_entry_id integer
        CONSTRAINT entries_duplicate_of_entry_id_fkey
        REFERENCES public.entries(id) ON DELETE SET NULL,
    ADD COLUMN client text,
    ADD COLUMN source_conversation_id integer
        CONSTRAINT entries_source_conversation_id_fkey
        REFERENCES public.conversations(id) ON DELETE SET NULL
"""

_UPGRADE_CONVERSATIONS = """
ALTER TABLE public.conversations
    ADD COLUMN extracted_content_hash text,
    ADD COLUMN extraction_version integer
"""

_CREATE_INDEXES = (
    "CREATE INDEX idx_entries_duplicate_of_entry_id "
    "ON public.entries USING btree (duplicate_of_entry_id)",
    "CREATE INDEX idx_entries_source_conversation_id "
    "ON public.entries USING btree (source_conversation_id)",
    "CREATE INDEX idx_entries_user_date_client "
    "ON public.entries USING btree (user_id, date, client) WHERE (deleted_at IS NULL)",
)

_DROP_INDEXES = (
    "DROP INDEX public.idx_entries_user_date_client",
    "DROP INDEX public.idx_entries_source_conversation_id",
    "DROP INDEX public.idx_entries_duplicate_of_entry_id",
)

_DOWNGRADE_ENTRIES = """
ALTER TABLE public.entries
    DROP COLUMN source_conversation_id,
    DROP COLUMN client,
    DROP COLUMN duplicate_of_entry_id
"""

_DOWNGRADE_CONVERSATIONS = """
ALTER TABLE public.conversations
    DROP COLUMN extraction_version,
    DROP COLUMN extracted_content_hash
"""


def _extraction_jobs_alter(clause: str) -> str:
    """One ALTER TABLE on extraction_jobs applying ``clause`` to every counter column."""
    clauses = ",\n    ".join(clause.format(column=column) for column in _EXTRACTION_JOB_COUNTERS)
    return f"ALTER TABLE public.extraction_jobs\n    {clauses}"


def upgrade() -> None:
    """Add the entries, extraction_jobs and conversations columns plus their indexes."""
    op.execute(_UPGRADE_ENTRIES)
    op.execute(_extraction_jobs_alter("ADD COLUMN {column} integer DEFAULT 0 NOT NULL"))
    op.execute(_UPGRADE_CONVERSATIONS)
    for statement in _CREATE_INDEXES:
        op.execute(statement)


def downgrade() -> None:
    """Drop every index and column this revision added."""
    for statement in _DROP_INDEXES:
        op.execute(statement)
    op.execute(_DOWNGRADE_CONVERSATIONS)
    op.execute(_extraction_jobs_alter("DROP COLUMN {column}"))
    op.execute(_DOWNGRADE_ENTRIES)
