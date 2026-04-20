"""Harden RLS tenant_isolation policy against empty-string current_setting.

Once an unprivileged GUC has been set in a session (via ``SET LOCAL`` in
any prior transaction), PostgreSQL keeps it registered for the rest of
the session. After the transaction commits the ``SET LOCAL`` value is
cleared, but subsequent ``current_setting('name', true)`` calls return
empty string ``""`` rather than NULL -- the missing-safe flag only
short-circuits when the GUC has *never* been touched on the connection.

The 0005 policy does ``current_setting('app.current_user_id', true)::uuid``
which RAISES ``invalid input syntax for type uuid: ""`` when the GUC was
previously set and then not re-bound before the next query. A pooled
connection that served a tenant-scoped request and then gets reused for
an unscoped query hits exactly this path.

Fix: wrap the current_setting call with ``NULLIF(..., '')`` so the empty-
string residue is treated as missing. The missing -> NULL -> ``user_id =
NULL`` evaluates to UNKNOWN (never true), so the row is filtered out.
Default-deny semantics are preserved and the cast-on-empty-string error
goes away.

Also relevant in prod: an operator who forgets to wire
``user_scoped_connection`` into a new code path would otherwise see the
error rather than a clean default-deny. The policy should fail closed,
not explode.
"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "0007_rls_policy_null_coalesce"
down_revision = "0006_add_encrypted_columns"
branch_labels = None
depends_on = None


_TENANT_TABLES = (
    "topics",
    "entries",
    "conversations",
    "messages",
    "entry_embeddings",
)


if not all(t.isidentifier() for t in _TENANT_TABLES):
    raise ValueError(
        "All entries in _TENANT_TABLES must be valid Python identifiers; " f"got {_TENANT_TABLES!r}"
    )


_POLICY_COMMENT = (
    "Default-deny tenant isolation. Matches rows where user_id equals the "
    "session variable app.current_user_id (set per-request by the app via "
    "SET LOCAL). NULLIF(..., '') treats an empty-string residue (left over "
    "after a previous SET LOCAL on the same pooled connection) as NULL, so "
    "an unscoped connection sees zero rows instead of raising on the cast. "
    "BYPASSRLS roles (journal_admin) skip this policy entirely and see every row."
)


def upgrade() -> None:
    """Replace each tenant_isolation policy with a NULLIF-safe version."""
    for table in _TENANT_TABLES:
        op.execute(f"DROP POLICY IF EXISTS tenant_isolation ON {table}")  # noqa: S608
        op.execute(
            f"""
            CREATE POLICY tenant_isolation ON {table}
                FOR ALL TO journal_app
                USING (user_id = (
                    SELECT NULLIF(current_setting('app.current_user_id', true), '')::uuid
                ))
                WITH CHECK (user_id = (
                    SELECT NULLIF(current_setting('app.current_user_id', true), '')::uuid
                ))
            """  # noqa: S608
        )
        op.execute(
            f"COMMENT ON POLICY tenant_isolation ON {table} IS "  # noqa: S608
            f"$policy${_POLICY_COMMENT}$policy$"
        )


def downgrade() -> None:
    """Restore the 0005 policy shape (no NULLIF wrap)."""
    old_comment = (
        "Default-deny tenant isolation. Matches rows where user_id equals the "
        "session variable app.current_user_id (set per-request by the app via "
        "SET LOCAL). A missing session variable returns NULL and no rows are "
        "visible -- this is fail-safe. BYPASSRLS roles (journal_admin) skip "
        "this policy entirely and see every row."
    )
    for table in _TENANT_TABLES:
        op.execute(f"DROP POLICY IF EXISTS tenant_isolation ON {table}")  # noqa: S608
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
            f"$policy${old_comment}$policy$"
        )
