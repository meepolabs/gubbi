"""Per-user uniqueness on ``topics(user_id, path)``.

Replaces the global ``topics_path_key UNIQUE (path)`` constraint
(carried forward from the baseline schema in
``gubbi/storage/schema.sql``) with the composite
``topics_user_path_key UNIQUE (user_id, path)``. Also drops the
now-redundant single-column ``idx_topics_user`` -- the new composite
UNIQUE creates an implicit btree on ``(user_id, path)`` whose
leftmost prefix already covers any query that filters by
``user_id`` alone, so the standalone index just costs storage and
INSERT/UPDATE overhead.

Background: migration 0004 (``add_user_id_to_tenants``, 2026-04-19)
added ``topics.user_id UUID NOT NULL REFERENCES users(id) ON DELETE
CASCADE`` so RLS could scope topics to their owner, but the unique
constraint was not updated to match. Under the old single-column
unique:

  - Two users could not both create a topic with the same path
    (e.g. ``work``, ``personal``, ``projects/X``). The second
    user's INSERT raised ``UniqueViolationError``, which
    ``repositories/topics.py:create()`` translates to ``ValueError``
    and the MCP tool surfaces as an ``ALREADY_EXISTS`` envelope --
    even though the calling user has no such topic.
  - The error envelope's ``Topic already exists: '{path}'`` text at
    ``gubbi/tools/errors.py:103`` confirms a topic exists somewhere
    in the system, leaking cross-tenant existence information that
    the rest of the stack (RLS-scoped reads, FORCED RLS policies)
    is careful never to leak.
  - The supporting hint that per-user scoping was the original
    intent is the ``idx_topics_user btree (user_id)`` index (also
    from migration 0004). Under the global ``UNIQUE (path)``,
    ``user_id`` already repeats freely across rows (one user owns
    many topics with different paths) so the index is well-shaped;
    what was missing was the structural per-user uniqueness on
    ``(user_id, path)`` that prevents the cross-tenant leak.

Post-migration, two users can each have a ``work`` topic; the
``ALREADY_EXISTS`` envelope only fires when a single user tries to
duplicate their own path -- the message is then accurate, not a
leak.

No application-code semantics change. The race-tolerant topic
upsert in ``extraction/jobs/extract_conversation.py`` (the only
caller that swallows duplicate-create exceptions) catches
``topic_repo.TopicAlreadyExists`` post-migration; same behaviour
as before, just a typed exception instead of a string-match. The
only case where ``UniqueViolationError`` fires post-migration is
the same-user duplicate case, which is exactly the case that
caller already handles.

Pre-flight: under the existing ``UNIQUE (path)`` constraint, no
``(user_id, path)`` duplicate group can exist by construction. The
explicit pre-flight is cheap insurance against weird state from a
downgrade/upgrade cycle on a populated DB; the migration aborts
loudly rather than silently producing a half-migrated schema.

Lock duration: the table is small (10s-100s of rows in practice;
~100 in the testbench corpus). Plain ``ALTER TABLE ... ADD
CONSTRAINT`` takes an ``AccessExclusiveLock`` for the duration of
the constraint creation, but the index build is fast at this scale.
Plain unique-index creation matches migration 0025
(``messages_search_vector_gin``) which deliberately does not use
``CONCURRENTLY`` for small / pre-launch tables -- ``CONCURRENTLY``
brings transaction-management complexity (``autocommit_block`` in
``gubbi.alembic._helpers``) that is only justified for large
production tables.

Downgrade: symmetric -- restore the global ``UNIQUE (path)``. If
the post-upgrade table contains rows where two users share a
``path`` value, the downgrade FAILS at constraint creation. That
is intentional: the upgraded state is unrepresentable in the
pre-upgrade schema, and silently dropping rows would be worse than
raising. Operators wanting to roll back must first reconcile the
duplicates manually.
"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "0030_topics_user_path_unique"
down_revision = "0029_audit_log_target_kind_check"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Replace ``topics.UNIQUE(path)`` with ``UNIQUE(user_id, path)``."""
    conn = op.get_bind()
    duplicate_groups = conn.execute(
        sa.text(
            "SELECT count(*) FROM ("
            "  SELECT user_id, path FROM topics"
            "  GROUP BY user_id, path HAVING count(*) > 1"
            ") d"
        )
    ).scalar()
    if duplicate_groups:
        # ``duplicate_groups`` is a server-side integer count, not
        # user input; the SELECT shape inside the f-string is a
        # diagnostic snippet for the operator, not a query that
        # gets executed. The S608 noqa scopes the suppression to
        # the literal that ruff flags.
        raise RuntimeError(
            f"topics has {duplicate_groups} (user_id, path) duplicate "  # noqa: S608
            "group(s); refusing to apply 0030. Investigate via "
            "`SELECT user_id, path, count(*) FROM topics GROUP BY "
            "user_id, path HAVING count(*) > 1` and reconcile before "
            "re-running."
        )
    op.execute("ALTER TABLE topics DROP CONSTRAINT IF EXISTS topics_path_key")
    op.execute("ALTER TABLE topics ADD CONSTRAINT topics_user_path_key UNIQUE (user_id, path)")
    # idx_topics_user (user_id) is now redundant: the composite
    # topics_user_path_key UNIQUE (user_id, path) creates an implicit
    # btree whose leftmost prefix covers user_id-only queries. Drop the
    # single-column index to save storage + INSERT/UPDATE overhead.
    op.execute("DROP INDEX IF EXISTS idx_topics_user")


def downgrade() -> None:
    """Restore the global ``UNIQUE (path)`` constraint.

    NOTE: if the table contains rows where two users share a
    ``path`` value (a state that is reachable post-upgrade but
    unrepresentable pre-upgrade), the constraint creation FAILS
    here. That is intentional -- silently dropping rows would be
    worse than raising. Investigate via
    ``SELECT path, count(*) FROM topics GROUP BY path HAVING count(*) > 1``
    and reconcile cross-tenant duplicates manually before
    re-attempting the downgrade.
    """
    op.execute("ALTER TABLE topics DROP CONSTRAINT IF EXISTS topics_user_path_key")
    op.execute("ALTER TABLE topics ADD CONSTRAINT topics_path_key UNIQUE (path)")
    # Restore idx_topics_user since the composite index is gone.
    op.execute("CREATE INDEX IF NOT EXISTS idx_topics_user ON topics (user_id)")
