#!/usr/bin/env bash
set -euo pipefail

# verify-db-invariants.sh -- Postgres GRANT/RLS/policy sanity check.
#
# Discovers tenant tables dynamically from pg_tables, then asserts that
# journal_app and journal_admin have the privileges expected after running
# migration 0002 + 0009 on a restored database. Exit 0 on pass, 1 on fail,
# 2 if no DSN is available.

# ---------------------------------------------------------------------------
# Resolve DSN (same fallback chain as journalctl/alembic/env.py)
# ---------------------------------------------------------------------------
DB_URL="${JOURNAL_DB_MIGRATION_URL:-${JOURNAL_DB_ADMIN_URL:-}}"

if [[ -z "${DB_URL}" ]]; then
    echo "ERROR: Neither JOURNAL_DB_MIGRATION_URL nor JOURNAL_DB_ADMIN_URL is set." >&2
    exit 2
fi

# ---------------------------------------------------------------------------
# Discover tenant tables dynamically.
# ---------------------------------------------------------------------------
_psql_rc=0
_psql_out=$(psql -v ON_ERROR_STOP=1 -tAc \
    "SELECT tablename FROM pg_tables WHERE schemaname = 'public' AND tableowner != 'postgres'" \
    "${DB_URL}" 2>&1) || _psql_rc=$?
if [[ "${_psql_rc}" -ne 0 ]]; then
    echo "ERROR: psql failed to query tenant tables (exit ${_psql_rc}): ${_psql_out}" >&2
    exit 2
fi
if [[ -z "${_psql_out}" ]]; then
    echo "verify-db-invariants: OK -- 0 tenant tables, no invariants to check"
    exit 0
fi
mapfile -t TENANT_TABLES < <(printf '%s\n' "${_psql_out}")

if [[ ${#TENANT_TABLES[@]} -eq 0 ]]; then
    echo "verify-db-invariants: OK -- 0 tenant tables, no invariants to check"
    exit 0
fi

PASS=true
FAILURES=()

# ---------------------------------------------------------------------------
# Helper: format a failure line.
# ---------------------------------------------------------------------------
fail_line() {
    local table=$1 role=$2 priv=$3
    FAILURES+=("${table}: role=${role} missing=${priv}")
}

# ---------------------------------------------------------------------------
# Per-table checks: journal_app and journal_admin privileges.
# ---------------------------------------------------------------------------
for tbl in "${TENANT_TABLES[@]}"; do
    # a) journal_app -- SELECT, INSERT, UPDATE, DELETE
    has_app=false
    if psql -v ON_ERROR_STOP=1 -tAc \
        "SELECT has_table_privilege('journal_app', quote_ident('${tbl}'), 'SELECT, INSERT, UPDATE, DELETE')" \
        "${DB_URL}" | grep -q "t"; then
        has_app=true
    fi

    if [[ "${has_app}" != "true" ]]; then
        # Drill down to find the exact missing privilege(s)
        for priv in SELECT INSERT UPDATE DELETE; do
            if ! psql -v ON_ERROR_STOP=1 -tAc \
                "SELECT has_table_privilege('journal_app', quote_ident('${tbl}'), '${priv}')" \
                "${DB_URL}" | grep -q "t"; then
                fail_line "${tbl}" journal_app "${priv}"
            fi
        done
    fi

    # b) journal_admin -- INSERT, UPDATE, DELETE, REFERENCES, TRIGGER (destructive subset of ALL)
    has_admin=false
    if psql -v ON_ERROR_STOP=1 -tAc \
        "SELECT has_table_privilege('journal_admin', quote_ident('${tbl}'), 'INSERT, UPDATE, DELETE, REFERENCES, TRIGGER')" \
        "${DB_URL}" | grep -q "t"; then
        has_admin=true
    fi

    if [[ "${has_admin}" != "true" ]]; then
        for priv in INSERT UPDATE DELETE REFERENCES TRIGGER; do
            if ! psql -v ON_ERROR_STOP=1 -tAc \
                "SELECT has_table_privilege('journal_admin', quote_ident('${tbl}'), '${priv}')" \
                "${DB_URL}" | grep -q "t"; then
                fail_line "${tbl}" journal_admin "${priv}"
            fi
        done
    fi

    # c) RLS enabled on this table (relrowsecurity).
    if ! psql -v ON_ERROR_STOP=1 -tAc \
        "SELECT c.relrowsecurity FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace WHERE n.nspname = 'public' AND c.relname = '${tbl}'" \
        "${DB_URL}" | grep -q "t"; then
        fail_line "${tbl}" rls relrowsecurity
    fi

    # d) At least one policy attached (via pg_policies).
    if psql -v ON_ERROR_STOP=1 -tAc \
        "SELECT COUNT(*) FROM pg_policies WHERE tablename = '${tbl}' AND schemaname = 'public'" \
        "${DB_URL}" | grep -qE '^\s*[1-9]'; then
        : # at least one policy exists
    else
        fail_line "${tbl}" rls at_least_one_policy
    fi
done

# ---------------------------------------------------------------------------
# e) Default privileges -- expect rows in pg_default_acl for both roles.
# ---------------------------------------------------------------------------
for role in journal_app journal_admin; do
    count=$(psql -v ON_ERROR_STOP=1 -tAc \
        "SELECT COUNT(*) FROM pg_default_acl WHERE defaclrole = '${role}'::regrole AND defaclnamespace = 'public'::regnamespace" \
        "${DB_URL}")
    if [[ "${count}" -lt 1 ]]; then
        fail_line "__default_privs__" "${role}" pg_default_acl_entry
    fi
done

# ---------------------------------------------------------------------------
# Report.
# ---------------------------------------------------------------------------
if [[ ${PASS} == "true" && ${#FAILURES[@]} -eq 0 ]]; then
    echo "verify-db-invariants: OK -- ${#TENANT_TABLES[@]} tenant tables, all grants/RLS/policies present"
    exit 0
fi

echo "--- invariant check failed ---" >&2
for f in "${FAILURES[@]}"; do
    echo "FAIL: ${f}" >&2
done
echo "" >&2
echo "Remedy: run restore-db.sh --repair-grants <dump-file> to replay migration 0002 + 0009 grants." >&2

exit 1
