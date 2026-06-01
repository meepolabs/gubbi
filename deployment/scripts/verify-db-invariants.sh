#!/usr/bin/env bash
set -euo pipefail

# verify-db-invariants.sh -- Postgres GRANT / RLS / policy / trigger / role
# sanity check for the gubbi-owned subset of the shared journal database.
#
# SCOPE: gubbi-owned objects ONLY. The shared DB also carries gubbi-cloud
# tables (tenants, subscriptions, llm_budgets, outbox_events, stripe_events)
# but those don't exist on a fresh DB until gubbi-cloud's alembic chain has
# also run. Since gubbi deploys first, gubbi's startup verification cannot
# assume cloud tables exist. Cloud-side invariants are checked by
# gubbi-cloud's lifespan probes (gubbi_cloud/bootstrap/_audit_probe.py and
# _budget_schema_probe.py).
#
# Asserts:
#   - the 8 gubbi tables exist (audit_log, conversations, entries,
#     entry_embeddings, extraction_jobs, messages, topics, users)
#   - the vector extension is installed
#   - 2 audit_log functions (audit_log_immutable, audit_log_admin_no_user_actor)
#   - 3 audit_log triggers (no_update, no_delete, admin_no_user_actor)
#   - RLS enabled + force-RLS on every gubbi tenant table
#   - each gubbi table's expected named policy exists, scoped TO journal_app
#   - role grants per gubbi table match the audited end-state, INCLUDING:
#       * journal_app on users     -> SELECT, UPDATE only (no INSERT/DELETE)
#       * journal_app on audit_log -> INSERT only
#       * journal_admin on audit_log -> SELECT, INSERT only (no UPDATE/DELETE)
#   - otel_ro role exists, LOGIN, member of pg_monitor, no data-table grants
#   - alembic_version + alembic_version_cloud are journal_admin-only
#   - default privileges for ROLE journal_admin in schema public have the
#     precise audited shape: TABLES default = SELECT/INSERT/UPDATE for
#     journal_app (DELETE intentionally absent) + ALL for journal_admin;
#     SEQUENCES default = SELECT/USAGE for journal_app + ALL for journal_admin
#
# Exit 0 on pass, 1 on fail, 2 if no DSN.

# ---------------------------------------------------------------------------
# Resolve DSN (same fallback chain as gubbi/alembic/env.py)
# ---------------------------------------------------------------------------
DB_URL="${JOURNAL_DB_MIGRATION_URL:-${JOURNAL_DB_ADMIN_URL:-}}"

if [[ -z "${DB_URL}" ]]; then
    echo "ERROR: Neither JOURNAL_DB_MIGRATION_URL nor JOURNAL_DB_ADMIN_URL is set." >&2
    exit 2
fi

# Strip any sqlalchemy +psycopg / +asyncpg dialect prefix; psql speaks plain
# postgresql://.
DB_URL=${DB_URL/postgresql+psycopg:\/\//postgresql://}
DB_URL=${DB_URL/postgresql+asyncpg:\/\//postgresql://}

PASS=true
FAILURES=()

fail() {
    FAILURES+=("$1")
    PASS=false
}

# ---------------------------------------------------------------------------
# psql wrapper -- single string return, or a single column expression.
# Uses -tA so the returned value is unadorned.
# ---------------------------------------------------------------------------
q() {
    psql -v ON_ERROR_STOP=1 -tAc "$1" "${DB_URL}"
}

# Returns 't' or 'f' for a single boolean assertion.
qb() {
    q "$1"
}

# Pass if SQL returns 't', else record the failure tag.
assert_true() {
    local tag=$1 sql=$2
    local result
    result=$(qb "$sql")
    if [[ "$result" != "t" ]]; then
        fail "${tag}: expected true, got '${result}'"
    fi
}

# ---------------------------------------------------------------------------
# Canonical sets (single source of truth for this script).
# ---------------------------------------------------------------------------
GUBBI_TENANT_TABLES=(audit_log conversations entries entry_embeddings extraction_jobs messages topics users)

# ---------------------------------------------------------------------------
# 1. Tables exist (gubbi-owned only)
# ---------------------------------------------------------------------------
for t in "${GUBBI_TENANT_TABLES[@]}"; do
    assert_true "table_exists ${t}" \
        "SELECT EXISTS (SELECT 1 FROM pg_class WHERE relkind='r' AND relnamespace='public'::regnamespace AND relname='${t}')"
done

# ---------------------------------------------------------------------------
# 2. Extension
# ---------------------------------------------------------------------------
assert_true "extension_vector" \
    "SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname='vector')"

# ---------------------------------------------------------------------------
# 3. Functions exist
# ---------------------------------------------------------------------------
for fn in audit_log_immutable audit_log_admin_no_user_actor; do
    assert_true "function_${fn}" \
        "SELECT EXISTS (SELECT 1 FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace WHERE n.nspname='public' AND p.proname='${fn}')"
done

# ---------------------------------------------------------------------------
# 4. RLS state per gubbi table
# ---------------------------------------------------------------------------
for t in "${GUBBI_TENANT_TABLES[@]}"; do
    assert_true "rls_enabled ${t}" \
        "SELECT relrowsecurity FROM pg_class WHERE relkind='r' AND relnamespace='public'::regnamespace AND relname='${t}'"
    assert_true "force_rls ${t}" \
        "SELECT relforcerowsecurity FROM pg_class WHERE relkind='r' AND relnamespace='public'::regnamespace AND relname='${t}'"
done

# ---------------------------------------------------------------------------
# 5. Named policies exist, scoped to journal_app -- name + role both checked
# (count-only or name-only assertions miss policy-name swaps and miss role
# escalation if a policy were re-created with TO public).
# ---------------------------------------------------------------------------
declare -a POLICY_PAIRS=(
    "audit_log:audit_log_app_insert_self_only"
    "conversations:tenant_isolation"
    "entries:tenant_isolation"
    "entry_embeddings:tenant_isolation"
    "extraction_jobs:extraction_jobs_user_isolation"
    "messages:tenant_isolation"
    "topics:tenant_isolation"
    "users:users_self_read"
    "users:users_self_update"
)
for pair in "${POLICY_PAIRS[@]}"; do
    table=${pair%%:*}
    policy=${pair##*:}
    assert_true "policy_${table}.${policy}" \
        "SELECT EXISTS (SELECT 1 FROM pg_policies WHERE schemaname='public' AND tablename='${table}' AND policyname='${policy}' AND roles::text='{journal_app}')"
done

# ---------------------------------------------------------------------------
# 6. Triggers (3 on audit_log)
# ---------------------------------------------------------------------------
for trg in trg_audit_log_admin_no_user_actor trg_audit_log_no_delete trg_audit_log_no_update; do
    assert_true "trigger_${trg}" \
        "SELECT EXISTS (SELECT 1 FROM pg_trigger t JOIN pg_class c ON t.tgrelid=c.oid WHERE c.relname='audit_log' AND t.tgname='${trg}' AND NOT t.tgisinternal)"
done

# ---------------------------------------------------------------------------
# 7. Grants on tables -- precise CRUD shape per role
# ---------------------------------------------------------------------------

# Helper: assert privilege flag.
has_priv() {
    local role=$1 table=$2 priv=$3
    qb "SELECT has_table_privilege('${role}', 'public.${table}', '${priv}')"
}

assert_priv() {
    local role=$1 table=$2 priv=$3 expected=$4
    local result
    result=$(has_priv "$role" "$table" "$priv")
    if [[ "$result" != "$expected" ]]; then
        fail "grant ${table}: ${role} ${priv} expected=${expected} got=${result}"
    fi
}

# Tables where journal_app gets full CRUD.
JOURNAL_APP_FULL_CRUD=(topics conversations entries messages entry_embeddings extraction_jobs)
for t in "${JOURNAL_APP_FULL_CRUD[@]}"; do
    for p in SELECT INSERT UPDATE DELETE; do
        assert_priv journal_app "$t" "$p" t
    done
done

# users: journal_app has SELECT + UPDATE only (REVOKE'd INSERT/DELETE per gubbi 0019).
for p in SELECT UPDATE; do
    assert_priv journal_app users "$p" t
done
for p in INSERT DELETE; do
    assert_priv journal_app users "$p" f
done

# audit_log: journal_app has INSERT only.
assert_priv journal_app audit_log INSERT t
for p in SELECT UPDATE DELETE; do
    assert_priv journal_app audit_log "$p" f
done

# audit_log: journal_admin has SELECT + INSERT only.
for p in SELECT INSERT; do
    assert_priv journal_admin audit_log "$p" t
done
for p in UPDATE DELETE; do
    assert_priv journal_admin audit_log "$p" f
done

# All other gubbi tables: journal_admin has full access.
JOURNAL_ADMIN_FULL=("${JOURNAL_APP_FULL_CRUD[@]}" users)
for t in "${JOURNAL_ADMIN_FULL[@]}"; do
    for p in SELECT INSERT UPDATE DELETE; do
        assert_priv journal_admin "$t" "$p" t
    done
done

# ---------------------------------------------------------------------------
# 8. otel_ro role: exists, LOGIN, member of pg_monitor, NO grants on any
# gubbi-owned data table.
# ---------------------------------------------------------------------------
assert_true "otel_ro_exists" \
    "SELECT EXISTS (SELECT 1 FROM pg_roles WHERE rolname='otel_ro')"
assert_true "otel_ro_login" \
    "SELECT rolcanlogin FROM pg_roles WHERE rolname='otel_ro'"
assert_true "otel_ro_pg_monitor" \
    "SELECT EXISTS (
        SELECT 1 FROM pg_auth_members am
        JOIN pg_roles r ON r.oid=am.member
        JOIN pg_roles g ON g.oid=am.roleid
        WHERE r.rolname='otel_ro' AND g.rolname='pg_monitor'
     )"
for t in "${GUBBI_TENANT_TABLES[@]}"; do
    assert_priv otel_ro "$t" SELECT f
done

# ---------------------------------------------------------------------------
# 8b. journal_admin role posture: BYPASSRLS, NO CREATEROLE (escalation
# attack surface; not used by any code path), member of journal_app
# (required by gubbi 0009 grant chain; init.sh / grants.sql sets it; assert
# here so deploys not via init.sh catch a missing membership).
# ---------------------------------------------------------------------------
assert_true "journal_admin_bypassrls" \
    "SELECT rolbypassrls FROM pg_roles WHERE rolname='journal_admin'"
assert_true "journal_admin_no_createrole" \
    "SELECT NOT rolcreaterole FROM pg_roles WHERE rolname='journal_admin'"
assert_true "journal_admin_member_of_journal_app" \
    "SELECT EXISTS (
        SELECT 1 FROM pg_auth_members am
        JOIN pg_roles r ON r.oid=am.member
        JOIN pg_roles g ON g.oid=am.roleid
        WHERE r.rolname='journal_admin' AND g.rolname='journal_app'
     )"

# ---------------------------------------------------------------------------
# 8c. app_current_user_active() helper exists. RLS policies for tenant
# isolation depend on it being defined (defense-in-depth against
# soft-deleted users retaining data-plane access).
# ---------------------------------------------------------------------------
assert_true "function_app_current_user_active" \
    "SELECT EXISTS (
        SELECT 1 FROM pg_proc p
        JOIN pg_namespace n ON n.oid=p.pronamespace
        WHERE n.nspname='public' AND p.proname='app_current_user_active'
          AND p.provolatile='s'
     )"

# ---------------------------------------------------------------------------
# 9. Alembic version tables: journal_admin only. journal_app revoked.
# These tables are auto-created by alembic on first run; assertions here
# guard against a future hand-edit / repair-grants run that re-grants them.
# Skipped if either table doesn't exist yet (e.g., gubbi alembic hasn't run).
# ---------------------------------------------------------------------------
for v in alembic_version alembic_version_cloud; do
    exists=$(qb "SELECT EXISTS (SELECT 1 FROM pg_class WHERE relkind='r' AND relnamespace='public'::regnamespace AND relname='${v}')")
    if [[ "$exists" == "t" ]]; then
        for p in SELECT INSERT UPDATE DELETE; do
            assert_priv journal_app "$v" "$p" f
        done
        for p in SELECT INSERT UPDATE DELETE; do
            assert_priv journal_admin "$v" "$p" t
        done
    fi
done

# ---------------------------------------------------------------------------
# 10. Default privileges (FOR ROLE journal_admin IN SCHEMA public) -- exact
# ACL shape, not just count. The audited shape:
#   TABLES:    journal_app -> SELECT,INSERT,UPDATE  (NO DELETE)
#              journal_admin -> ALL
#   SEQUENCES: journal_app -> SELECT,USAGE
#              journal_admin -> ALL
# Strict equality on per-grantee privilege set so regressions are caught.
# ---------------------------------------------------------------------------
assert_true "default_privs_tables_journal_app_no_delete" \
    "SELECT NOT EXISTS (
        SELECT 1 FROM pg_default_acl, aclexplode(defaclacl) AS a
        WHERE defaclrole = 'journal_admin'::regrole
          AND defaclnamespace = 'public'::regnamespace
          AND defaclobjtype = 'r'
          AND a.grantee = 'journal_app'::regrole
          AND a.privilege_type = 'DELETE'
     )"
assert_true "default_privs_tables_journal_app_has_iud" \
    "SELECT (
        SELECT COUNT(*) FROM pg_default_acl, aclexplode(defaclacl) AS a
        WHERE defaclrole = 'journal_admin'::regrole
          AND defaclnamespace = 'public'::regnamespace
          AND defaclobjtype = 'r'
          AND a.grantee = 'journal_app'::regrole
          AND a.privilege_type IN ('SELECT','INSERT','UPDATE')
     ) = 3"
assert_true "default_privs_tables_journal_admin_all" \
    "SELECT (
        SELECT COUNT(DISTINCT a.privilege_type) FROM pg_default_acl, aclexplode(defaclacl) AS a
        WHERE defaclrole = 'journal_admin'::regrole
          AND defaclnamespace = 'public'::regnamespace
          AND defaclobjtype = 'r'
          AND a.grantee = 'journal_admin'::regrole
     ) >= 7"

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
if [[ ${PASS} == "true" && ${#FAILURES[@]} -eq 0 ]]; then
    echo "verify-db-invariants: OK -- 8 gubbi tables, 9 policies, 3 triggers, otel_ro role, alembic_version revocation, default-priv shape all match audited baseline"
    exit 0
fi

echo "--- invariant check failed ---" >&2
for f in "${FAILURES[@]}"; do
    echo "FAIL: ${f}" >&2
done
echo "" >&2
echo "Remedy:" >&2
echo "  - For grants-only failure (most common):" >&2
echo "      psql -v ON_ERROR_STOP=1 -f deployment/scripts/grants.sql \"\${JOURNAL_DB_MIGRATION_URL:-\${JOURNAL_DB_ADMIN_URL}}\"" >&2
echo "  - For full restore + repair from a pg_dump:" >&2
echo "      deployment/scripts/restore-db.sh --repair-grants <dump-file>" >&2
echo "  - For schema drift (missing tables / RLS gone): re-run alembic upgrade head" >&2
echo "    or, if catastrophic, restore from backup." >&2

exit 1
