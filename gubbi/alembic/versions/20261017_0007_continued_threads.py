"""Key synced conversations by source id and track per-message source ids.

conversations: the ``UNIQUE (topic_id, slug)`` constraint
``conversations_topic_id_slug_key`` becomes the partial unique index
``conversations_topic_id_slug_mcp_key`` on ``(topic_id, slug) WHERE platform_id
IS NULL``. Conversations saved by title keep the title as their identity;
synced conversations (``platform_id`` set) are identified only by
``idx_conv_platform_dedup (user_id, platform, platform_id)``, so two synced
threads with the same title in one topic are separate rows. The index is
created before the constraint is dropped, so the title key is never absent.
Narrowing a unique key cannot create a violation, so no row is repaired.

``extracted_through_position``: the highest message position already
extracted. It is backfilled to ``max(messages.position)`` for every
conversation with ``processed_at`` set, so an extracted thread that later
grows is not extracted again from the start; every other row stays NULL.

The backfill is only correct when no extraction run is in flight: a run of the
previous version that completes after it sets ``processed_at`` without a
watermark, so the thread's next append would be extracted again from the
start. Required deploy order: stop the extraction workers and the extraction
scheduler, let running jobs finish (or mark them failed), run this migration,
then start the new version. The upgrade enforces the middle step: it takes a
SHARE lock on ``extraction_jobs``, which holds off any job claim until it
commits, and refuses to run while any job has status ``running``.

messages: ``source_message_id`` is the id the source platform gave a message,
NULL when it sent none. The partial unique index
``idx_messages_conversation_source_id`` on ``(conversation_id,
source_message_id) WHERE source_message_id IS NOT NULL`` stores each source
message at most once per conversation.

Both tables already carry FORCE ROW LEVEL SECURITY and table-level grants to
journal_app and journal_admin, so the new columns inherit both; no policy or
grant changes here. Each ADD COLUMN is nullable without a default, a
catalog-only change.

The downgrade restores ``conversations_topic_id_slug_key``. Once synced
conversations exist, a topic can hold two of them with the same title; the
downgrade then fails on the restored constraint (a unique violation naming
``conversations_topic_id_slug_key``) and rolls back, leaving this revision in
place. Rename or remove one row of each such pair before downgrading.

No statement uses IF [NOT] EXISTS. An object of the same name already present
means the schema drifted from this chain, so the upgrade fails and rolls back
rather than recording the revision over an incompatible object; the downgrade
likewise drops only what it expects to find.
"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "0007_continued_threads"
down_revision = "0006_llm_wallet"
branch_labels = None
depends_on = None

_LOCK_JOBS = "LOCK TABLE public.extraction_jobs IN SHARE MODE"

_COUNT_RUNNING_JOBS = "SELECT count(*) FROM public.extraction_jobs WHERE status = 'running'"

_RUNNING_JOBS_MESSAGE = (
    "{count} extraction job(s) are running. Stop the extraction workers and the "
    "extraction scheduler, let running jobs finish or mark them failed, then run "
    "this migration again before starting the new version."
)

_CREATE_TITLE_INDEX = (
    "CREATE UNIQUE INDEX conversations_topic_id_slug_mcp_key "
    "ON public.conversations USING btree (topic_id, slug) WHERE (platform_id IS NULL)"
)

_DROP_TITLE_CONSTRAINT = (
    "ALTER TABLE public.conversations DROP CONSTRAINT conversations_topic_id_slug_key"
)

_ADD_TITLE_CONSTRAINT = (
    "ALTER TABLE public.conversations "
    "ADD CONSTRAINT conversations_topic_id_slug_key UNIQUE (topic_id, slug)"
)

_DROP_TITLE_INDEX = "DROP INDEX public.conversations_topic_id_slug_mcp_key"

_ADD_CONVERSATIONS_COLUMN = (
    "ALTER TABLE public.conversations ADD COLUMN extracted_through_position integer"
)

_DROP_CONVERSATIONS_COLUMN = (
    "ALTER TABLE public.conversations DROP COLUMN extracted_through_position"
)

_ADD_MESSAGES_COLUMN = "ALTER TABLE public.messages ADD COLUMN source_message_id text"

_DROP_MESSAGES_COLUMN = "ALTER TABLE public.messages DROP COLUMN source_message_id"

_CREATE_SOURCE_ID_INDEX = (
    "CREATE UNIQUE INDEX idx_messages_conversation_source_id "
    "ON public.messages USING btree (conversation_id, source_message_id) "
    "WHERE (source_message_id IS NOT NULL)"
)

_DROP_SOURCE_ID_INDEX = "DROP INDEX public.idx_messages_conversation_source_id"

_BACKFILL_EXTRACTED_THROUGH = """
UPDATE public.conversations c
SET extracted_through_position = m.max_position
FROM (
    SELECT conversation_id, max("position") AS max_position
    FROM public.messages
    GROUP BY conversation_id
) m
WHERE c.id = m.conversation_id
  AND c.processed_at IS NOT NULL
"""


def _refuse_running_jobs() -> None:
    op.execute(_LOCK_JOBS)
    running = op.get_bind().execute(sa.text(_COUNT_RUNNING_JOBS)).scalar_one()
    if running:
        raise RuntimeError(_RUNNING_JOBS_MESSAGE.format(count=running))


def upgrade() -> None:
    """Re-key conversation titles, add the source-id and watermark columns, backfill."""
    _refuse_running_jobs()
    op.execute(_CREATE_TITLE_INDEX)
    op.execute(_DROP_TITLE_CONSTRAINT)
    op.execute(_ADD_CONVERSATIONS_COLUMN)
    op.execute(_ADD_MESSAGES_COLUMN)
    op.execute(_CREATE_SOURCE_ID_INDEX)
    op.execute(_BACKFILL_EXTRACTED_THROUGH)


def downgrade() -> None:
    """Restore the title constraint and drop every column and index this revision added."""
    op.execute(_DROP_SOURCE_ID_INDEX)
    op.execute(_DROP_MESSAGES_COLUMN)
    op.execute(_DROP_CONVERSATIONS_COLUMN)
    op.execute(_ADD_TITLE_CONSTRAINT)
    op.execute(_DROP_TITLE_INDEX)
