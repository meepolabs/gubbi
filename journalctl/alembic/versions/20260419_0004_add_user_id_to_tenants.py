"""Add user_id FK to every tenant table + backfill existing rows to the founder.

Every tenant-scoped table (topics, entries, conversations, messages,
entry_embeddings) gains ``user_id UUID NOT NULL REFERENCES users(id)
ON DELETE CASCADE`` so the app layer and upcoming RLS policies (02.05)
can scope by user.

Six-phase migration inside one alembic transaction:

1. Add nullable user_id column to each tenant table.
2. Seed a single "founder" row in users (email from env, UUID generated).
3. Backfill every tenant row's user_id to the founder's UUID.
4. Guard: verify zero NULL user_id rows remain after backfill (operator-
   friendly early failure, before Phase 5 surfaces it as cryptic PG error).
5. Promote user_id to NOT NULL on every tenant table.
6. Add composite indexes for the hot-path queries under RLS.

Requires env var ``JOURNAL_FOUNDER_EMAIL`` to be set before ``alembic
upgrade``. This is a one-time seed value for the sole pre-Kratos tenant.

KRATOS REBIND — when Kratos is wired later and the operator signs up with
this same email, the internal users.id (random UUID generated here) needs
to be reconciled with the Kratos identity UUID. Do NOT attempt a naive
``UPDATE users SET id = <kratos-uuid> WHERE email = ...`` — all five
tenant FKs were created with the default ``ON UPDATE NO ACTION`` and the
UPDATE will fail on the first referencing row. The correct path is a
follow-up migration that either (a) adds a separate ``kratos_identity_id
UUID UNIQUE`` column and keeps the internal UUID as stable PK (preferred
— decouples internal identity from external auth provider), or (b)
rebuilds the FK constraints as ``ON UPDATE CASCADE`` first, then runs
the UPDATE inside a single transaction. Option (a) is the intended path
and will be tracked under Track C as part of Kratos wiring (02.08/02.09).

ON DELETE CASCADE from tenant tables → users: matches GDPR
right-to-erasure (hard-deleting a user wipes their data). Soft delete
via users.deleted_at is the normal path; RLS policies will filter
``users.deleted_at IS NULL`` to hide soft-deleted tenants.

SCALE NOTE — for future migrations on tables with 10k+ rows, the
``ADD COLUMN ... REFERENCES`` + ``SET NOT NULL`` pattern used here takes
ACCESS EXCLUSIVE for the whole validation scan and will block reads/writes
for seconds to minutes. The low-downtime pattern at that scale is
two-phase: ``ADD COLUMN ... NOT VALID`` (no scan) then ``VALIDATE
CONSTRAINT`` later (SHARE UPDATE EXCLUSIVE, compatible with reads). For
NOT NULL specifically, add a ``CHECK (col IS NOT NULL) NOT VALID``
constraint, ``VALIDATE CONSTRAINT`` outside the hot lock, then ``SET NOT
NULL`` runs fast because PG can short-circuit using the validated CHECK.
Current row counts (~600 entries) make this unnecessary for 0004 itself.
"""

import os

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "0004_add_user_id_to_tenants"
down_revision = "0003_create_users_table"
branch_labels = None
depends_on = None


_TENANT_TABLES = (
    "topics",
    "entries",
    "conversations",
    "messages",
    "entry_embeddings",
)

# Defense in depth — these names are f-string-interpolated into DDL statements.
# A non-identifier here would be an injection surface for any future contributor
# who adds a dynamic entry without realizing. Runtime check (not assert) so
# `python -O` cannot strip the guard.
if not all(t.isidentifier() for t in _TENANT_TABLES):
    raise ValueError(
        "All entries in _TENANT_TABLES must be valid Python identifiers; " f"got {_TENANT_TABLES!r}"
    )


def _founder_email() -> str:
    """Read the founder email from env or fail loudly."""
    email = os.environ.get("JOURNAL_FOUNDER_EMAIL")
    if not email:
        raise RuntimeError(
            "JOURNAL_FOUNDER_EMAIL must be set before running migration 0004. "
            "Set it to the existing operator's email (e.g. 'you@example.com'). "
            "This seeds the one founder row into the users table and backfills "
            "all pre-multitenant data to that user."
        )
    return email


def upgrade() -> None:
    """Add user_id FK to tenant tables, seed founder, backfill, enforce NOT NULL."""
    email = _founder_email()

    # Phase 1 — add nullable user_id column on every tenant table.
    # Table names are f-string-interpolated here because they are DDL
    # identifiers (parameter binding is not valid for DDL identifier
    # positions); source is the identifier-validated _TENANT_TABLES tuple,
    # never user input.
    for table in _TENANT_TABLES:
        op.execute(
            f"""
            ALTER TABLE {table}
            ADD COLUMN IF NOT EXISTS user_id UUID
            REFERENCES users (id) ON DELETE CASCADE
            """  # noqa: S608 — table from identifier-validated tuple
        )

    # Phase 2 — seed founder user row (idempotent via partial unique email index)
    op.execute(
        sa.text(
            """
            INSERT INTO users (id, email, timezone)
            VALUES (gen_random_uuid(), :email, 'UTC')
            ON CONFLICT (email) WHERE deleted_at IS NULL DO NOTHING
            """
        ).bindparams(email=email)
    )

    # Phase 3 — backfill every tenant row's user_id to the founder.
    # One UPDATE per tenant table; the partial unique index on users.email
    # (from 0003) makes the subquery lookup an index-only probe.
    for table in _TENANT_TABLES:
        op.execute(
            sa.text(
                f"""
                UPDATE {table}
                SET user_id = (
                    SELECT id FROM users
                    WHERE email = :email AND deleted_at IS NULL
                )
                WHERE user_id IS NULL
                """  # noqa: S608 — table name from a fixed tuple, not user input
            ).bindparams(email=email)
        )

    # Phase 4 — guard: catch silent NULL propagation before Phase 5 turns it
    # into a cryptic "column contains null values" Postgres error. The most
    # common cause of reaching this guard is a typo in JOURNAL_FOUNDER_EMAIL
    # that caused the Phase 3 subquery to return NULL.
    bind = op.get_bind()
    for table in _TENANT_TABLES:
        null_count = bind.execute(
            sa.text(f"SELECT COUNT(*) FROM {table} WHERE user_id IS NULL")  # noqa: S608
        ).scalar_one()
        if null_count > 0:
            raise RuntimeError(
                f"Backfill left {null_count} NULL user_id rows in {table}. "
                "This usually means JOURNAL_FOUNDER_EMAIL does not match "
                "an active row in users, or the founder insert silently "
                "conflicted. Aborting before SET NOT NULL to preserve state."
            )

    # Phase 5 — promote user_id to NOT NULL now that backfill is complete
    for table in _TENANT_TABLES:
        op.execute(
            f"ALTER TABLE {table} ALTER COLUMN user_id SET NOT NULL"  # noqa: S608
        )

    # Phase 6 — composite indexes for the hot-path queries under RLS
    op.execute("CREATE INDEX IF NOT EXISTS idx_topics_user " "ON topics (user_id)")
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_entries_user_topic_date "
        "ON entries (user_id, topic_id, date DESC) WHERE deleted_at IS NULL"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_entries_user_indexed "
        "ON entries (user_id, id) WHERE indexed_at IS NULL"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_conv_user_topic " "ON conversations (user_id, topic_id)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_messages_user_conv_pos "
        "ON messages (user_id, conversation_id, position)"
    )
    op.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_user " "ON entry_embeddings (user_id)")


def downgrade() -> None:
    """Drop composite indexes and user_id columns from every tenant table.

    The founder users row is intentionally left in place — it belongs to the
    users table created in 0003 and is orthogonal to the tenant-FK addition.
    """
    # Drop composite indexes first (safe even if already gone)
    op.execute("DROP INDEX IF EXISTS idx_embeddings_user")
    op.execute("DROP INDEX IF EXISTS idx_messages_user_conv_pos")
    op.execute("DROP INDEX IF EXISTS idx_conv_user_topic")
    op.execute("DROP INDEX IF EXISTS idx_entries_user_indexed")
    op.execute("DROP INDEX IF EXISTS idx_entries_user_topic_date")
    op.execute("DROP INDEX IF EXISTS idx_topics_user")

    # Drop user_id columns (FK constraint + NOT NULL drop implicitly)
    for table in _TENANT_TABLES:
        op.execute(f"ALTER TABLE {table} DROP COLUMN IF EXISTS user_id")
