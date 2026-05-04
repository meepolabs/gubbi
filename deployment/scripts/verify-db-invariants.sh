#!/usr/bin/env bash
set -euo pipefail

# verify-db-invariants.sh -- Postgres GRANT/RLS/policy sanity check.
#
# Asserts that journal_app and journal_admin have the privileges expected
# after running migrations 0002 + 0009 + 0010 on a restored database.
# Table sets are hardcoded; the canonical source of expected state is
# deployment/scripts/grants.sql. Exit 0 on pass, 1 on fail, 2 if no DSN.

# ---------------------------------------------------------------------------
# Resolve DSN (same fallback chain as gubbi/alembic/env.py)
# ---------------------------------------------------------------------------
DB_URL="${JOURNAL_DB_MIGRATION_URL:-${JOURNAL_DB_ADMIN_URL:-}}"

if [[ -z "${DB_URL}" ]]; then
    echo "ERROR: Neither JOURNAL_DB_MIGRATION_URL nor JOURNAL_DB_ADMIN_URL is set." >&2
    exit 2
fi

# ---------------------------------------------------------------------------
# Canonical table sets (sourced from migration 0005 RLS list + 0010).
# alembic_version is managed by alembic; do not include here.
# ---------------------------------------------------------------------------
TENANT_TABLES=(topics entries conversations messages entry_embeddings)
USERS_TABLE=users
AUDIT_LOG_TABLE=audit_log

PASS=true
FAILURES=()

# ---------------------------------------------------------------------------
# Helper: record a failure line.
# ---------------------------------------------------------------------------
fail_line() {
    local table=$1 role=$2 priv=$3
    FAILURES+=("${table}: role=${role} missing=${priv}")
    PASS=false
}

# ---------------------------------------------------------------------------
# Helper: assert a privilege IS held.
# ---------------------------------------------------------------------------
assert_has_priv() {
    local table=$1 role=$2 priv=$3
    if ! psql -v ON_ERROR_STOP=1 -tAc \
        "SELECT has_table_privilege('${role}', quote_ident('${table}'), '${priv}')" \
        "${DB_URL}" | grep -q "t"; then
        fail_line "${table}" "${role}" "${priv}"
    fi
}

# ---------------------------------------------------------------------------
# Helper: assert a privilege is NOT held.
# ---------------------------------------------------------------------------
assert_lacks_priv() {
    local table=$1 role=$2 priv=$3
    if psql -v ON_ERROR_STOP=1 -tAc \
        "SELECT has_table_privilege('${role}', quote_ident('${table}'), '${priv}')" \
        "${DB_URL}" | grep -q "t"; then
        FAILURES+=("${table}: role=${role} should_not_have=${priv}")
        PASS=false
    fi
}

# ---------------------------------------------------------------------------
# Per-table checks for tenant tables: full CRUD for both roles.
# ---------------------------------------------------------------------------
for tbl in "${TENANT_TABLES[@]}"; do
    # journal_app -- SELECT, INSERT, UPDATE, DELETE
    for priv in SELECT INSERT UPDATE DELETE; do
        assert_has_priv "${tbl}" journal_app "${priv}"
    done

    # journal_admin -- INSERT, UPDATE, DELETE, REFERENCES, TRIGGER (destructive subset of ALL)
    for priv in INSERT UPDATE DELETE REFERENCES TRIGGER; do
        assert_has_priv "${tbl}" journal_admin "${priv}"
    done

    # RLS enabled (relrowsecurity)
    if ! psql -v ON_ERROR_STOP=1 -tAc \
        "SELECT c.relrowsecurity FROM pg_class c
         JOIN pg_namespace n ON n.oid = c.relnamespace
         WHERE n.nspname = 'public' AND c.relname = '${tbl}'" \
        "${DB_URL}" | grep -q "t"; then
        fail_line "${tbl}" rls relrowsecurity
    fi

    # At least one policy attached
    if ! psql -v ON_ERROR_STOP=1 -tAc \
        "SELECT COUNT(*) FROM pg_policies WHERE tablename = '${tbl}' AND schemaname = 'public'" \
        "${DB_URL}" | grep -qE '^\s*[1-9]'; then
        fail_line "${tbl}" rls at_least_one_policy
    fi
done

# ---------------------------------------------------------------------------
# users table: same CRUD as tenant tables; no RLS check.
# ---------------------------------------------------------------------------
for priv in SELECT INSERT UPDATE DELETE; do
    assert_has_priv "${USERS_TABLE}" journal_app "${priv}"
done
for priv in INSERT UPDATE DELETE REFERENCES TRIGGER; do
    assert_has_priv "${USERS_TABLE}" journal_admin "${priv}"
done

# ---------------------------------------------------------------------------
# audit_log: append-only least-privilege (migration 0010).
#   journal_app: INSERT only -- no SELECT, UPDATE, DELETE
#   journal_admin: SELECT + INSERT only -- no UPDATE, DELETE
# ---------------------------------------------------------------------------
assert_lacks_priv "${AUDIT_LOG_TABLE}" journal_app SELECT
assert_has_priv   "${AUDIT_LOG_TABLE}" journal_app INSERT
assert_lacks_priv "${AUDIT_LOG_TABLE}" journal_app UPDATE
assert_lacks_priv "${AUDIT_LOG_TABLE}" journal_app DELETE

assert_has_priv   "${AUDIT_LOG_TABLE}" journal_admin SELECT
assert_has_priv   "${AUDIT_LOG_TABLE}" journal_admin INSERT
assert_lacks_priv "${AUDIT_LOG_TABLE}" journal_admin UPDATE
assert_lacks_priv "${AUDIT_LOG_TABLE}" journal_admin DELETE

# ---------------------------------------------------------------------------
# Default privileges -- expect rows in pg_default_acl for both roles.
# ---------------------------------------------------------------------------
for role in journal_app journal_admin; do
    count=$(psql -v ON_ERROR_STOP=1 -tAc \
        "SELECT COUNT(*) FROM pg_default_acl
         WHERE defaclrole = '${role}'::regrole
           AND defaclnamespace = 'public'::regnamespace" \
        "${DB_URL}")
    if [[ "${count}" -lt 1 ]]; then
        fail_line "__default_privs__" "${role}" pg_default_acl_entry
    fi
done

# ---------------------------------------------------------------------------
# Report.
# ---------------------------------------------------------------------------
total_tables=$(( ${#TENANT_TABLES[@]} + 2 ))  # tenant + users + audit_log
if [[ ${PASS} == "true" && ${#FAILURES[@]} -eq 0 ]]; then
    echo "verify-db-invariants: OK -- ${total_tables} tables, all grants/RLS/policies present"
    exit 0
fi

echo "--- invariant check failed ---" >&2
for f in "${FAILURES[@]}"; do
    echo "FAIL: ${f}" >&2
done
echo "" >&2
echo "Remedy: run restore-db.sh --repair-grants <dump-file> to replay deployment/scripts/grants.sql." >&2

exit 1
