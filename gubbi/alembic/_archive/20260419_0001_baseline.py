"""Baseline schema from schema.sql.

Apply the full idempotent schema as the initial migration revision.
The schema SQL uses CREATE TABLE IF NOT EXISTS throughout, so
running upgrade head repeatedly is safe.
"""

from pathlib import Path

from alembic import op

# revision identifiers, used by Alembic.
revision = "0001_baseline"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Apply the full baseline schema."""
    schema_path = Path(__file__).parents[3] / "gubbi" / "storage" / "schema.sql"
    sql = schema_path.read_text(encoding="utf-8")
    op.execute(sql)


def downgrade() -> None:
    """Drop all tables in reverse FK-dependency order.

    pgvector extension is intentionally left in place since other
    database objects may depend on it.
    """
    op.execute(
        "DROP TABLE IF EXISTS entry_embeddings CASCADE;"
        " DROP TABLE IF EXISTS messages CASCADE;"
        " DROP TABLE IF EXISTS entries CASCADE;"
        " DROP TABLE IF EXISTS conversations CASCADE;"
        " DROP TABLE IF EXISTS topics CASCADE;"
    )
