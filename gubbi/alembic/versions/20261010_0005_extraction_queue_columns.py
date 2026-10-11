"""Add scheduling and admission-charge columns to extraction_jobs.

Scheduling: ``lane`` (``backfill`` | ``live`` | ``paste``), ``run_after`` (the
earliest time the job may run), ``defer_reason`` (NULL, or why admission held
the job back: ``budget`` | ``global`` | ``provider``), ``attempts`` and
``activity_at`` (the conversation's latest message time).

Admission charge, written once when the job is admitted: ``filtered_tokens``,
``token_count_version``, ``admitted_at`` and ``budget_period_start``.

``period_start`` gains a column default of the first day of the current UTC
month, so an INSERT may omit it.

Existing rows read ``lane = 'backfill'``. Rows still ``pending`` or ``failed``
get ``run_after = 'infinity'``: they are held until released explicitly, so
applying this revision never puts a backlog of old jobs back in the queue.

Two partial indexes: runnable selection over ``(status, run_after)`` for
pending and failed rows, and per-user period usage over
``(user_id, budget_period_start)`` for admitted rows.

The table already carries FORCE ROW LEVEL SECURITY and table-level grants to
journal_app and journal_admin, so the new columns inherit both; no policy or
grant changes here. Every added column has a constant or stable default or is
nullable, so no ADD COLUMN rewrites the table.

No statement uses IF [NOT] EXISTS. A column, constraint or index of the same
name already present means the schema drifted from this chain, so the upgrade
fails and rolls back rather than recording the revision over an incompatible
object; the downgrade likewise drops only what it expects to find.
"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "0005_extraction_queue_columns"
down_revision = "0004_extraction_tracking_columns"
branch_labels = None
depends_on = None

_ADD_COLUMNS = """
ALTER TABLE public.extraction_jobs
    ADD COLUMN lane text DEFAULT 'backfill' NOT NULL
        CONSTRAINT extraction_jobs_lane_check
        CHECK (lane IN ('backfill', 'live', 'paste')),
    ADD COLUMN run_after timestamp with time zone DEFAULT now() NOT NULL,
    ADD COLUMN defer_reason text
        CONSTRAINT extraction_jobs_defer_reason_check
        CHECK (defer_reason IN ('budget', 'global', 'provider')),
    ADD COLUMN attempts integer DEFAULT 0 NOT NULL,
    ADD COLUMN activity_at timestamp with time zone,
    ADD COLUMN filtered_tokens integer DEFAULT 0 NOT NULL,
    ADD COLUMN token_count_version smallint,
    ADD COLUMN admitted_at timestamp with time zone,
    ADD COLUMN budget_period_start date,
    ALTER COLUMN period_start SET DEFAULT (date_trunc('month', now() AT TIME ZONE 'UTC'))::date
"""

_HOLD_UNFINISHED_JOBS = (
    "UPDATE public.extraction_jobs SET run_after = 'infinity' WHERE status IN ('pending', 'failed')"
)

_CREATE_INDEXES = (
    "CREATE INDEX idx_extraction_jobs_runnable "
    "ON public.extraction_jobs USING btree (status, run_after) "
    "WHERE (status IN ('pending', 'failed'))",
    "CREATE INDEX idx_extraction_jobs_user_period_admitted "
    "ON public.extraction_jobs USING btree (user_id, budget_period_start) "
    "WHERE (admitted_at IS NOT NULL)",
)

_DROP_INDEXES = (
    "DROP INDEX public.idx_extraction_jobs_user_period_admitted",
    "DROP INDEX public.idx_extraction_jobs_runnable",
)

_DROP_COLUMNS = """
ALTER TABLE public.extraction_jobs
    ALTER COLUMN period_start DROP DEFAULT,
    DROP COLUMN budget_period_start,
    DROP COLUMN admitted_at,
    DROP COLUMN token_count_version,
    DROP COLUMN filtered_tokens,
    DROP COLUMN activity_at,
    DROP COLUMN attempts,
    DROP COLUMN defer_reason,
    DROP COLUMN run_after,
    DROP COLUMN lane
"""


def upgrade() -> None:
    """Add the columns and indexes, then hold every unfinished job."""
    op.execute(_ADD_COLUMNS)
    op.execute(_HOLD_UNFINISHED_JOBS)
    for statement in _CREATE_INDEXES:
        op.execute(statement)


def downgrade() -> None:
    """Drop every index and column this revision added and the period_start default."""
    for statement in _DROP_INDEXES:
        op.execute(statement)
    op.execute(_DROP_COLUMNS)
