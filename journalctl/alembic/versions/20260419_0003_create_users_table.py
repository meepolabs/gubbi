"""Create users table — minimal local mirror of Kratos identities.

Provisions the users table that serves as the system-of-record for
application-level user data. `id` is set by the caller to the Kratos
identity UUID, enabling RLS policies and FK relationships on tenant
tables (TASK-02.03). Kratos identity webhook sync is Track C scope.

`updated_at` is bumped by the application layer on INSERT/UPDATE; no
trigger is created here, matching existing repository convention on
topics.updated_at and conversations.updated_at (bumped via CTEs in
Python repos).

Email uniqueness is enforced via a partial unique index scoped to
active rows (deleted_at IS NULL). This allows email reuse after a
user is hard-deleted per GDPR right-to-erasure while preserving
uniqueness among active tenants — preferred over a table-level
UNIQUE constraint which would block rehydration.
"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "0003_create_users_table"
down_revision = "0002_create_db_roles"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create users table + supporting indexes. Idempotent at statement level."""
    op.execute(
        sa.text(
            """
            CREATE TABLE IF NOT EXISTS users (
                id UUID PRIMARY KEY,
                email TEXT NOT NULL,
                timezone TEXT NOT NULL DEFAULT 'UTC',
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                deleted_at TIMESTAMPTZ
            )
            """
        )
    )

    # Email unique among active users; allows email reuse after hard-delete
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email_active "
        "ON users (email) WHERE deleted_at IS NULL"
    )


def downgrade() -> None:
    """Drop users table using RESTRICT so cross-migration rollback is safe.

    RESTRICT forces Postgres to refuse the drop if any tenant table still
    references users(id) — the operator must downgrade later migrations
    (0004+, which add user_id FK columns) first. This prevents a one-shot
    `alembic downgrade 0002` from silently cascade-deleting every tenant
    table along with users.
    """
    op.execute("DROP TABLE IF EXISTS users RESTRICT")
