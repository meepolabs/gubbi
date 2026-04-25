"""Add user_id FK to every tenant table -- pure DDL, no user-row creation.

Every tenant-scoped table (topics, entries, conversations, messages,
entry_embeddings) gains ``user_id UUID NOT NULL REFERENCES users(id)
ON DELETE CASCADE`` so the app layer and upcoming RLS policies (02.05)
can scope by user.

Three-phase migration inside one alembic transaction:

1. Add nullable user_id column to each tenant table.
5. Promote user_id to NOT NULL on every tenant table.
6. Add composite indexes for the hot-path queries under RLS.

This migration does NOT create any users rows or backfill data.
The operator row must be provisioned separately (see
journalctl/scripts/provision_operator.py). If JOURNAL_OPERATOR_EMAIL
is set, the app will look it up at startup; if no matching user row
exists in ``users``, operator-identity auth reaches DB code without
a user binding and MissingUserIdError surfaces as a 500.

KRATOS REBIND -- when Kratos is wired later and the operator signs up with
this same email, the internal users.id (random UUID generated separately) needs
to be reconciled with the Kratos identity UUID. Do NOT attempt a naive
``UPDATE users SET id = <kratos-uuid> WHERE email = ...`` -- all five
tenant FKs were created with the default ``ON UPDATE NO ACTION`` and the
UPDATE will fail on the first referencing row. The correct path is a
follow-up migration that either (a) adds a separate ``kratos_identity_id
UUID UNIQUE`` column and keeps the internal UUID as stable PK (preferred
-- decouples internal identity from external auth provider), or (b)
rebuilds the FK constraints as ``ON UPDATE CASCADE`` first, then runs
the UPDATE inside a single transaction. Option (a) is the intended path
and will be tracked under Track C as part of Kratos wiring (02.08/02.09).

ON DELETE CASCADE from tenant tables -> users: matches GDPR
right-to-erasure (hard-deleting a user wipes their data). Soft delete
via users.deleted_at is the normal path; RLS policies will filter
``users.deleted_at IS NULL`` to hide soft-deleted tenants.

SCALE NOTE -- for future migrations on tables with 10k+ rows, the
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

# Defense in depth -- these names are f-string-interpolated into DDL statements.
# A non-identifier here would be an injection surface for any future contributor
# who adds a dynamic entry without realizing. Runtime check (not assert) so
# ``python -O`` cannot strip the guard.
if not all(t.isidentifier() for t in _TENANT_TABLES):
    raise ValueError(
        "All entries in _TENANT_TABLES must be valid Python identifiers; " f"got {_TENANT_TABLES!r}"
    )


def upgrade() -> None:
    """Add user_id FK to tenant tables, enforce NOT NULL, create indexes."""

    # Phase 1 -- add nullable user_id column on every tenant table.
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
            """  # noqa: S608 -- table from identifier-validated tuple
        )

    # Pre-Phase-5 guard: fail early if any tenant rows have NULL user_id.
    # On a fresh DB this is a no-op. On a Mode 1/2 DB with existing data, this
    # fires when provision_operator.py has not been run yet (no users row to
    # backfill against). The error message points operators at the fix.
    bind = op.get_bind()
    for table in _TENANT_TABLES:
        null_count = bind.execute(
            sa.text(f"SELECT COUNT(*) FROM {table} WHERE user_id IS NULL")  # noqa: S608
        ).scalar_one()
        if null_count > 0:
            raise RuntimeError(
                f"Migration 0004 aborted: {null_count} row(s) in '{table}' have NULL user_id. "
                "Run `python deployment/scaffold_self_host.py` to create the operator "
                "row, then backfill user_id manually before re-running alembic upgrade."
            )

    # Phase 5 -- promote user_id to NOT NULL now that column is present on all rows
    for table in _TENANT_TABLES:
        op.execute(
            f"ALTER TABLE {table} ALTER COLUMN user_id SET NOT NULL"  # noqa: S608
        )

    # Phase 6 -- composite indexes for the hot-path queries under RLS
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

    The operator users row is intentionally left in place -- it belongs to the
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
