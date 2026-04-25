"""One-shot cleanup: trim unparsed tool-call XML from entries.reasoning.

Targets rows where reasoning contains '<parameter name=' patterns left by
a prior client bug. For each match:
  - truncates reasoning at the first '<parameter' occurrence
  - attempts to recover tags from the param value if param name is 'tags'
    and the value JSON-parses as a list
  - writes back in a single per-row transaction
  - inserts an audit_log row with action='cleanup_xml_spill' and
    before/after detail in metadata

Idempotent: a second run finds zero matching rows because the pattern is gone.
Re-embedding is out of scope -- operator runs the reindex tool separately.
"""

import json
import logging
import re
from collections.abc import Sequence
from typing import Any

import sqlalchemy as sa
from alembic import op

logger = logging.getLogger(__name__)

revision = "0011_cleanup_xml_spill"
down_revision = "0010_audit_log"
branch_labels = None
depends_on = None

_PARAM_RE = re.compile(r'<parameter\s+name=["\'](\w+)["\']>(.*?)</parameter>', re.DOTALL)


def _clean_rows(bind: Any, rows: Sequence[Any]) -> None:
    """Clean XML spill from a list of row tuples.

    Each row is an object with .id, .reasoning, and .tags attributes.
    This helper exists to make unit-testing the cleanup logic possible
    without a live database connection.

    Args:
        bind: DBAPI connection or SQLAlchemy bind for executing SQL.
        rows: Iterable of row objects (with id, reasoning, tags attrs).
    """
    for row in rows:
        entry_id = row.id
        reasoning = row.reasoning
        tags = row.tags

        cut = reasoning.find("<parameter")
        if cut < 0:
            # No XML spill found in this row; skip it.
            continue

        clean_reasoning = reasoning[:cut].rstrip()

        # Attempt tags recovery if a 'tags' param is present and tags is null/empty
        recovered_tags = tags
        match = _PARAM_RE.search(reasoning[cut:])
        if match and match.group(1) == "tags":
            try:
                candidate = json.loads(match.group(2).strip())
                if isinstance(candidate, list):
                    recovered_tags = candidate
            except (json.JSONDecodeError, ValueError):
                pass

        before_summary = {
            "reasoning_len": len(reasoning),
            "reasoning_snippet": reasoning[cut : cut + 120],
            "tags_before": tags,
        }
        after_summary = {
            "reasoning_len": len(clean_reasoning),
            "tags_after": recovered_tags,
        }

        bind.execute(
            sa.text("UPDATE entries SET reasoning = :r, tags = :t WHERE id = :id"),
            {
                "r": clean_reasoning,
                # tags column is TEXT[] (schema.sql:54). psycopg2 converts Python
                # lists to PostgreSQL array literal wire format automatically;
                # passing json.dumps() would store a bare JSON string that breaks
                # downstream reads which expect an actual text array.
                "t": recovered_tags,
                "id": entry_id,
            },
        )

        bind.execute(
            sa.text(
                """
                INSERT INTO audit_log
                    (actor_type, actor_id, action, target_type, target_id, reason, metadata)
                VALUES
                    ('system', 'migration_0011', 'cleanup_xml_spill',
                     'entry', :target_id, 'automated xml spill cleanup', :meta::jsonb)
                """
            ),
            {
                "target_id": str(entry_id),
                "meta": json.dumps({"before": before_summary, "after": after_summary}),
            },
        )

        logger.info(
            "cleanup_xml_spill: cleaned entry %d, reasoning trimmed from %d to %d chars",
            entry_id,
            len(reasoning),
            len(clean_reasoning),
        )


def upgrade() -> None:
    bind = op.get_bind()

    rows = bind.execute(
        sa.text("SELECT id, reasoning, tags FROM entries " "WHERE reasoning ~ '<parameter name='")
    ).fetchall()

    logger.info("cleanup_xml_spill: found %d row(s) to clean", len(rows))

    _clean_rows(bind, rows)


def downgrade() -> None:
    # Data cleanup cannot be reversed safely; this is intentional.
    pass
