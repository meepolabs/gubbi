"""Squashed baseline for the gubbi schema chain.

Replaces the 0001-0031 evolution with a single mechanically-derived
baseline. The schema is captured at the original-chain head (gubbi
0031, gubbi-cloud 0020) by pg_dump on a fully-migrated testbench DB,
so it is provably equivalent to running the chain end-to-end.

Cross-repo deploy ordering: gubbi MUST upgrade head before gubbi-cloud
because gubbi-cloud's tenants.user_id FKs to gubbi's users(id). This
migration creates users (with email_verified_at -- previously added
out-of-band by gubbi-cloud migration 0007, now correctly owned here);
gubbi-cloud's baseline assumes users already exists.

Roles journal_app + journal_admin must exist before this migration
runs; they are pre-created with passwords by testbench
config/postgres/init.sh and prod gubbi-stack/postgres-init.sh. The
otel_ro role IS created by this migration (LOGIN, pg_monitor-only, no
data grants; password set out-of-band at deploy time) since pg_dump
--schema-only does not capture cluster-global roles.

The original 0001-0031 chain lives at _archive/ (sibling of
versions/) for forward dev-DB migration; alembic does not load it.
"""

from pathlib import Path

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "0001_squashed_baseline"
down_revision = None
branch_labels = None
depends_on = None

# Schema is captured in a sibling .sql file rather than inlined as a
# Python string so the SQL stays diff-friendly and matches the pg_dump
# source byte-for-byte modulo the header we strip.
_BASELINE_SQL = Path(__file__).resolve().parents[1] / "baseline_schema.sql"


def upgrade() -> None:
    """Apply the entire gubbi-owned schema in one statement batch."""
    sql = _BASELINE_SQL.read_text(encoding="utf-8")
    op.execute(sa.text(sql))


def downgrade() -> None:
    """No meaningful downgrade.

    A squashed baseline is the floor of the chain; rolling back means
    dropping the whole gubbi schema, which is not something any
    deploy path needs. If you genuinely need to wipe the schema, use
    `DROP SCHEMA public CASCADE` outside alembic and restore from the
    most recent pg_dump.
    """
    raise NotImplementedError(
        "0001_squashed_baseline has no downgrade -- "
        "wipe the schema with DROP SCHEMA public CASCADE if needed."
    )
