"""Enable row-level security + tenant_isolation policy on every tenant table.

Defense-in-depth tenant isolation: every query run by ``journal_app``
against topics, entries, conversations, messages, or entry_embeddings
is automatically filtered to rows where
``user_id = current_setting('app.current_user_id', true)::uuid``.

Migrations run under ``journal_admin`` (BYPASSRLS, from 0002), so this
migration is unaffected by the policies it creates. ``journal_app`` has
no BYPASSRLS attribute and therefore sees only rows for the user whose
UUID is bound via ``SET LOCAL app.current_user_id = $1`` — that
connection-scoping helper lands in TASK-02.06.

Default-deny semantics: the ``true`` flag on ``current_setting(..., true)``
makes it **missing-safe** — if the session variable is unset, it returns
NULL. ``user_id = NULL`` is never true in SQL, so an unscoped connection
sees zero rows. This is exactly what we want: a caller who forgets to
attach a user context reads an empty database, not someone else's data.

``FOR ALL TO journal_app`` covers SELECT / INSERT / UPDATE / DELETE in
one policy. ``WITH CHECK`` is the belt-and-braces guard against a
compromised or buggy app writing rows belonging to another user —
without it, ``INSERT INTO entries (user_id, ...) VALUES ('other', ...)``
would succeed (USING only governs visibility, not writes).

pgvector HNSW + RLS: at very large scale, HNSW scans on entry_embeddings
happen index-first, then RLS filters results. The denormalised
``entry_embeddings.user_id`` column + the ``idx_embeddings_user`` index
added in 0004 keep this fast at v1 scale. A per-tenant partition scheme
is a future optimisation, not a launch blocker. Semantic-search queries
should also ``SET LOCAL hnsw.ef_search`` higher under multi-tenancy (or
use pgvector 0.8+ iterative scan) so HNSW returns enough candidates that
survive the RLS filter — tracked under TASK-02.06.

SECURITY DEFINER WARNING: any Postgres function created as
``SECURITY DEFINER`` owned by ``journal_admin`` will execute with
BYPASSRLS and silently skip every policy defined here. Treat such
functions as a trust boundary: either own them with a non-BYPASSRLS role,
or explicitly ``SET LOCAL row_security = on`` inside the function body.
No such functions exist today; this is a forward-looking guardrail.
"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "0005_enable_rls"
down_revision = "0004_add_user_id_to_tenants"
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


_POLICY_COMMENT = (
    "Default-deny tenant isolation. Matches rows where user_id equals the "
    "session variable app.current_user_id (set per-request by the app via "
    "SET LOCAL). A missing session variable returns NULL and no rows are "
    "visible — this is fail-safe. BYPASSRLS roles (journal_admin) skip "
    "this policy entirely and see every row."
)


def upgrade() -> None:
    """Enable RLS + FORCE RLS and create tenant_isolation policy per tenant table.

    FORCE ROW LEVEL SECURITY ensures the policy applies even to the table owner
    (journal_admin today, has BYPASSRLS so no runtime effect). Protects against
    a future ownership transfer to a non-BYPASSRLS role silently bypassing the
    policy.

    USING/WITH CHECK wrap ``current_setting(...)::uuid`` in a scalar subquery so
    PG treats it as a one-shot InitPlan evaluated once per query rather than
    per-row VOLATILE. This is the Supabase/PostgREST-idiomatic pattern; at
    100k+ rows per tenant the per-row evaluation would be the dominant cost
    of policy enforcement.
    """
    for table in _TENANT_TABLES:
        # Table name from identifier-validated _TENANT_TABLES tuple — DDL only,
        # never user input.
        op.execute(
            f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY"  # noqa: S608
        )
        op.execute(
            f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY"  # noqa: S608
        )
        # USING governs SELECT/UPDATE/DELETE visibility.
        # WITH CHECK governs INSERT/UPDATE writes. Deliberately identical —
        # a user may only read and write their own rows. Keep the two in
        # lockstep; asymmetric edits here are almost always a bug.
        op.execute(
            f"""
            CREATE POLICY tenant_isolation ON {table}
                FOR ALL TO journal_app
                USING (user_id = (SELECT current_setting('app.current_user_id', true)::uuid))
                WITH CHECK (user_id = (SELECT current_setting('app.current_user_id', true)::uuid))
            """  # noqa: S608
        )
        op.execute(
            f"COMMENT ON POLICY tenant_isolation ON {table} IS "  # noqa: S608
            f"$policy${_POLICY_COMMENT}$policy$"
        )


def downgrade() -> None:
    """Drop policies and disable RLS (+ clear FORCE) on every tenant table."""
    for table in _TENANT_TABLES:
        op.execute(
            f"DROP POLICY IF EXISTS tenant_isolation ON {table}"  # noqa: S608
        )
        op.execute(
            f"ALTER TABLE {table} NO FORCE ROW LEVEL SECURITY"  # noqa: S608
        )
        op.execute(
            f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY"  # noqa: S608
        )
