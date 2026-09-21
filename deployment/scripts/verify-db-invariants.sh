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
#   - audit_log's SELECT-applicable policy set is EXACTLY one permissive policy
#     named audit_log_app_select_self_only, TO journal_app, whose USING
#     predicate and the INSERT policy's WITH CHECK each match the literal
#     self-only contract and mirror each other. A second permissive SELECT (or
#     FOR ALL) policy would OR-widen the read surface, so existence alone is not
#     enough -- and applicability is evaluated through role membership, so a
#     policy scoped TO PUBLIC or TO a parent role of journal_app is part of the
#     matched set.
#   - role grants per gubbi table match the audited end-state, INCLUDING:
#       * journal_app on users     -> SELECT, UPDATE only (no INSERT/DELETE)
#       * journal_app on audit_log -> INSERT, plus column-level SELECT on
#         EXACTLY actor_id, target_kind, target_id, action, metadata (the
#         five columns the deduped audit INSERT's ON CONFLICT inference
#         clause reads). Table-wide SELECT stays denied; every other column
#         is asserted unreadable, derived from the catalog. EFFECTIVE exposure
#         is checked too: any SELECT granted to PUBLIC or to a role whose
#         privileges journal_app holds is reported with its grantee, since no
#         REVOKE naming journal_app clears it.
#       * journal_admin on audit_log -> SELECT + INSERT (explicit grants);
#         NO explicit UPDATE/DELETE in pg_class.relacl. Ownership-implicit
#         UPDATE/DELETE is NOT blocked at the ACL level (cannot be --
#         journal_admin owns the table per the migration role contract);
#         runtime enforcement is via the BEFORE UPDATE/DELETE triggers
#         verified in section 6. This invariant check keeps the contract
#         VERIFIABLE post-deploy; it does NOT make audit_log tamper-
#         resistant against a compromised journal_admin DSN (out of scope
#         here; would require an external WORM sink).
#   - otel_ro role exists, LOGIN, member of pg_monitor, no data-table grants.
#     Existence gates the rest: has_table_privilege() raises on a missing role,
#     which under `set -e` would abort before the report block, so the dependent
#     checks record a labeled unverified failure instead of aborting.
#   - role posture, each of the four superuser/BYPASSRLS booleans on its own:
#       * journal_app   -> NOSUPERUSER, NOBYPASSRLS (RLS is only enforced for a
#         role that holds neither)
#       * journal_admin -> NOSUPERUSER, BYPASSRLS (the one posture boolean that
#         fails when WEAKENED; the maintenance paths need it)
#     plus NO CREATEROLE on journal_admin and its journal_app membership
#   - REACHABLE posture: journal_app cannot come to execute as ANY role holding
#     SUPERUSER or BYPASSRLS. Role attributes are not inherited, so its own clean
#     row says nothing about what it can SET ROLE into; the walk follows INHERIT,
#     SET and ADMIN edges, where a pg_has_role(USAGE) test sees only INHERIT.
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

# Non-audit gubbi tables, and each role's expected list derived from it
# INDEPENDENTLY. Defining these at canonical-set scope rather than inside a
# role-presence branch is load-bearing: when JOURNAL_APP_FULL_CRUD was declared
# inside the journal_app arm, an absent journal_app left it unset and the
# journal_admin loop below silently iterated an EMPTY list -- so every
# journal_admin grant went unchecked while the report showed no failure for them.
GUBBI_NON_AUDIT_TABLES=(topics conversations entries messages entry_embeddings extraction_jobs)
JOURNAL_APP_FULL_CRUD=("${GUBBI_NON_AUDIT_TABLES[@]}")
JOURNAL_ADMIN_FULL=("${GUBBI_NON_AUDIT_TABLES[@]}" users)

# ---------------------------------------------------------------------------
# Role existence gate.
#
# has_table_privilege(), has_column_privilege() and 'role'::regrole all RAISE
# when the named role does not exist, and under `set -e` that psql error kills
# the script inside a command substitution -- before the remaining sections and
# before the report block. A deployment missing a role would then report an EMPTY
# failure list while leaving every later invariant unchecked, which is strictly
# worse than a named failure.
#
# So role existence is resolved ONCE, up front, and every section whose queries
# name a role consults the result: present roles are checked normally, absent
# ones record a labeled failure per skipped check and execution continues to the
# posture sections and the final aggregation.
# ---------------------------------------------------------------------------
role_exists() {
    [[ "$(qb "SELECT EXISTS (SELECT 1 FROM pg_roles WHERE rolname='$1')")" == "t" ]]
}

JOURNAL_APP_PRESENT=false
JOURNAL_ADMIN_PRESENT=false
OTEL_RO_PRESENT=false
role_exists journal_app && JOURNAL_APP_PRESENT=true
role_exists journal_admin && JOURNAL_ADMIN_PRESENT=true
role_exists otel_ro && OTEL_RO_PRESENT=true

# 'public.audit_log'::regclass RAISES when the table is absent, with the same
# set -e consequence as a missing role. A deployment whose migration chain did not
# complete -- which is exactly what a missing required role causes -- would
# otherwise abort here instead of reporting the missing role it was asked about.
AUDIT_LOG_PRESENT=false
[[ "$(qb "SELECT EXISTS (SELECT 1 FROM pg_class WHERE relkind='r' AND relnamespace='public'::regnamespace AND relname='audit_log')")" == "t" ]] \
    && AUDIT_LOG_PRESENT=true

if [[ "${JOURNAL_APP_PRESENT}" != "true" ]]; then
    fail "role_exists journal_app: expected true, got 'f'"
fi
if [[ "${JOURNAL_ADMIN_PRESENT}" != "true" ]]; then
    fail "role_exists journal_admin: expected true, got 'f'"
fi

# Record one labeled unverified failure per check a missing role forces us to
# skip, so the report says exactly which invariants went unmeasured.
fail_unverified() {
    local role=$1
    shift
    for label in "$@"; do
        fail "${label}: unverified, role ${role} absent"
    done
}

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
    "audit_log:audit_log_app_select_self_only"
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
# 5b. audit_log SELECT policy set -- EXACT, not just present.
#
# A permissive policy is OR-ed with every other permissive policy for the same
# command, so "the expected policy exists" says nothing about how wide the read
# surface actually is: a second permissive SELECT (or FOR ALL) policy on
# audit_log widens it regardless. These checks pin the whole applicable set:
#   * exactly one policy applies to SELECT for journal_app, and it is the
#     expected name
#   * it is PERMISSIVE (polpermissive) -- a RESTRICTIVE policy of the same name
#     would AND instead of OR and silently deny the dedup read
#   * its USING predicate AND the INSERT policy's WITH CHECK each match the
#     literal self-only contract, pinned independently, and mirror each other
#
# APPLICABILITY IS EFFECTIVE, NOT DIRECT. A policy's polroles is matched against
# every role journal_app has the privileges of, not only journal_app itself:
# measured on PostgreSQL 17, a policy scoped TO a parent role of journal_app
# applies to journal_app even when journal_app is NOINHERIT. polroles = {0} is
# TO PUBLIC, which applies to every role; pg_has_role() does not accept oid 0,
# so PUBLIC gets its own arm of the predicate.
# ---------------------------------------------------------------------------
if [[ "${AUDIT_LOG_PRESENT}" == "true" ]]; then
    AUDIT_LOG_SELECT_POLICY=audit_log_app_select_self_only
    AUDIT_LOG_INSERT_POLICY=audit_log_app_insert_self_only

    # The self-only contract as written in the migration and in grants.sql. Pinned
    # literally: comparing the two policies only to EACH OTHER passes a mutation
    # that widens both to `true`, which keeps them mirrored while exposing every row.
    AUDIT_LOG_SELF_ONLY_CONTRACT="actor_id = (SELECT NULLIF(current_setting('app.current_user_id', true), '')) AND actor_id <> '' AND actor_type = 'user'"

    # 'r' = SELECT, '*' = ALL. Both apply to a SELECT statement. A policy reaches
    # journal_app if any of its polroles is PUBLIC (oid 0) or is a role whose
    # privileges journal_app holds -- itself included.
    _audit_log_select_policies="
        SELECT polname
        FROM pg_policy p
        WHERE p.polrelid = 'public.audit_log'::regclass
          AND p.polcmd IN ('r', '*')
          AND EXISTS (
              SELECT 1 FROM unnest(p.polroles) AS r(oid)
              WHERE r.oid = 0 OR pg_has_role('journal_app', r.oid, 'USAGE')
          )
    "

    # pg_has_role() raises on a missing role, so the applicable-set checks are gated.
    if [[ "${JOURNAL_APP_PRESENT}" == "true" ]]; then
        assert_true "policy_set audit_log: exactly one SELECT-applicable policy for journal_app" \
            "SELECT (SELECT COUNT(*) FROM (${_audit_log_select_policies}) s) = 1"

        assert_true "policy_set audit_log: the SELECT-applicable policy is ${AUDIT_LOG_SELECT_POLICY}" \
            "SELECT EXISTS (
                SELECT 1 FROM (${_audit_log_select_policies}) s
                WHERE s.polname = '${AUDIT_LOG_SELECT_POLICY}'
             )"
    else
        fail_unverified journal_app \
            "policy_set audit_log: exactly one SELECT-applicable policy" \
            "policy_set audit_log: the SELECT-applicable policy is ${AUDIT_LOG_SELECT_POLICY}"
    fi

    assert_true "policy_cmd audit_log.${AUDIT_LOG_SELECT_POLICY}: polcmd is SELECT" \
        "SELECT polcmd = 'r' FROM pg_policy
         WHERE polrelid = 'public.audit_log'::regclass
           AND polname = '${AUDIT_LOG_SELECT_POLICY}'"

    assert_true "policy_permissive audit_log.${AUDIT_LOG_SELECT_POLICY}" \
        "SELECT polpermissive FROM pg_policy
         WHERE polrelid = 'public.audit_log'::regclass
           AND polname = '${AUDIT_LOG_SELECT_POLICY}'"

    assert_true "policy_roles audit_log.${AUDIT_LOG_SELECT_POLICY}: journal_app only" \
        "SELECT ARRAY(SELECT rolname::text FROM pg_roles WHERE oid = ANY (polroles) ORDER BY rolname)
                = ARRAY['journal_app']
         FROM pg_policy
         WHERE polrelid = 'public.audit_log'::regclass
           AND polname = '${AUDIT_LOG_SELECT_POLICY}'"

    # pg_get_expr reprints a USING and a WITH CHECK of the same expression in forms
    # that differ cosmetically, so every side is normalized before comparison:
    # whitespace, ::text casts, parens and the subquery output alias pg adds
    # (AS "nullif") are stripped.
    _norm() {
        printf "%s" "regexp_replace(regexp_replace(coalesce(${1}, ''), '\\s|::text|[()]', '', 'g'), 'AS\"nullif\"', '', 'g')"
    }

    _norm_policy_expr() {
        local column=$1 policy=$2
        _norm "(SELECT pg_get_expr(${column}, polrelid) FROM pg_policy
                WHERE polrelid = 'public.audit_log'::regclass AND polname = '${policy}')"
    }

    _norm_contract=$(_norm "'${AUDIT_LOG_SELF_ONLY_CONTRACT//\'/\'\'}'")

    # Each predicate is pinned to the literal contract on its own, so widening BOTH
    # to `true` together -- which keeps the mirror check below green -- still fails.
    assert_true "policy_predicate audit_log.${AUDIT_LOG_SELECT_POLICY}: USING matches the self-only contract" \
        "SELECT $(_norm_policy_expr polqual "${AUDIT_LOG_SELECT_POLICY}") = ${_norm_contract}"

    assert_true "policy_predicate audit_log.${AUDIT_LOG_INSERT_POLICY}: WITH CHECK matches the self-only contract" \
        "SELECT $(_norm_policy_expr polwithcheck "${AUDIT_LOG_INSERT_POLICY}") = ${_norm_contract}"

    assert_true "policy_predicate audit_log.${AUDIT_LOG_SELECT_POLICY}: mirrors ${AUDIT_LOG_INSERT_POLICY}" \
        "SELECT $(_norm_policy_expr polqual "${AUDIT_LOG_SELECT_POLICY}") =
                $(_norm_policy_expr polwithcheck "${AUDIT_LOG_INSERT_POLICY}")"
else
    fail "policy_set audit_log: unverified, table audit_log absent"
    fail "policy_predicate audit_log: unverified, table audit_log absent"
fi

# ---------------------------------------------------------------------------
# 6. Triggers (3 on audit_log)
# ---------------------------------------------------------------------------
for trg in trg_audit_log_admin_no_user_actor trg_audit_log_no_delete trg_audit_log_no_update; do
    assert_true "trigger_${trg}_exists_and_enabled" \
        "SELECT EXISTS (SELECT 1 FROM pg_trigger t JOIN pg_class c ON t.tgrelid=c.oid WHERE c.relname='audit_log' AND t.tgname='${trg}' AND NOT t.tgisinternal AND t.tgenabled = 'O')"
done

# ---------------------------------------------------------------------------
# 7. Grants on tables -- precise CRUD shape per role
# ---------------------------------------------------------------------------

# Helper: assert privilege flag.
#
# has_table_privilege() raises on a missing TABLE just as it does on a missing
# role, so table existence is resolved first and an absent table records a labeled
# failure. Without this a deployment whose migration chain did not complete aborts
# mid-script -- which is precisely the state a missing required role produces.
table_exists() {
    [[ "$(qb "SELECT EXISTS (SELECT 1 FROM pg_class WHERE relkind='r' AND relnamespace='public'::regnamespace AND relname='$1')")" == "t" ]]
}

has_priv() {
    local role=$1 table=$2 priv=$3
    qb "SELECT has_table_privilege('${role}', 'public.${table}', '${priv}')"
}

assert_priv() {
    local role=$1 table=$2 priv=$3 expected=$4
    local result
    if ! table_exists "$table"; then
        fail "grant ${table}: ${role} ${priv} unverified, table ${table} absent"
        return
    fi
    result=$(has_priv "$role" "$table" "$priv")
    if [[ "$result" != "$expected" ]]; then
        fail "grant ${table}: ${role} ${priv} expected=${expected} got=${result}"
    fi
}

# ---------------------------------------------------------------------------
# Helper note on owner-implicit privileges:
# has_table_privilege() returns true for the table owner regardless of any
# explicit REVOKE -- ownership grants implicit ALL, and that cannot be
# revoked. journal_admin OWNS the 8 gubbi tables (per the squashed baseline
# running under JOURNAL_DB_MIGRATION_URL; locked by
# gubbi-testbench/tests/test_alembic_role_contract.py). So when verifying
# "journal_admin should NOT have UPDATE/DELETE on audit_log",
# assert_priv (has_table_privilege) is the WRONG tool: it would silently
# pass because of ownership, masking real ACL drift.
#
# Decision tree:
#   role is NOT the table owner            -> assert_priv (has_table_privilege)
#   role IS the table owner, expecting t   -> assert_priv (passes via ownership)
#   role IS the table owner, expecting f   -> assert_no_explicit_priv (relacl)
#
# Runtime enforcement of audit_log append-only is via the BEFORE UPDATE /
# BEFORE DELETE triggers (verified in section 6 above).
# ---------------------------------------------------------------------------
assert_no_explicit_priv() {
    local role=$1 table=$2 priv=$3
    if ! table_exists "$table"; then
        fail "no_explicit_priv ${table}: ${role} ${priv} unverified, table ${table} absent"
        return
    fi
    assert_true "no_explicit_priv ${table}: ${role} ${priv}" \
        "SELECT NOT EXISTS (
            SELECT 1 FROM pg_class c, aclexplode(c.relacl) AS a
            WHERE c.relname = '${table}'
              AND c.relnamespace = 'public'::regnamespace
              AND c.relkind = 'r'
              AND a.grantee = '${role}'::regrole
              AND a.privilege_type = '${priv}'
         )"
}

# has_table_privilege / has_column_privilege / ::regrole all raise on a missing
# role, so the whole journal_app arm is gated rather than allowed to abort.
if [[ "${JOURNAL_APP_PRESENT}" == "true" ]]; then
    # Tables where journal_app gets full CRUD (canonical list defined above).
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

    # audit_log: journal_app holds column-level SELECT on EXACTLY the five columns
    # the deduped audit INSERT's ON CONFLICT inference clause reads. Table-wide
    # SELECT is asserted absent above; these two checks pin the column set from
    # both directions so neither a missing grant nor a broadened one passes.
    # has_column_privilege is the right tool here: journal_app does not own the
    # table, so no ownership-implicit privilege can mask ACL drift.
    AUDIT_LOG_APP_SELECT_COLUMNS=(actor_id target_kind target_id action metadata)
    if [[ "${AUDIT_LOG_PRESENT}" != "true" ]]; then
        AUDIT_LOG_APP_SELECT_COLUMNS=()
        fail "column_grant audit_log: unverified, table audit_log absent"
        fail "inherited_grant audit_log: unverified, table audit_log absent"
    fi
    for c in "${AUDIT_LOG_APP_SELECT_COLUMNS[@]}"; do
        assert_true "column_grant audit_log.${c}: journal_app SELECT" \
            "SELECT has_column_privilege('journal_app', 'public.audit_log', '${c}', 'SELECT')"
    done
    # Every other audit_log column must be denied. Derived from the catalog rather
    # than hardcoded so a column added by a future migration is caught here.
    audit_log_extra_readable=""
    [[ "${AUDIT_LOG_PRESENT}" == "true" ]] && audit_log_extra_readable=$(q "
        SELECT COALESCE(string_agg(column_name, ',' ORDER BY column_name), '')
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = 'audit_log'
          AND column_name NOT IN ('actor_id','target_kind','target_id','action','metadata')
          AND has_column_privilege('journal_app', 'public.audit_log', column_name, 'SELECT')
    ")
    if [[ -n "${audit_log_extra_readable}" ]]; then
        fail "column_grant audit_log: journal_app can read columns outside the dedup conflict target: ${audit_log_extra_readable}"
    fi

    # EFFECTIVE exposure, attributed to its grantee. has_table_privilege /
    # has_column_privilege above already fold in inherited roles and PUBLIC, so they
    # detect that exposure exists -- but they cannot say WHERE it came from, and a
    # table-level check reporting 'f' while a parent role holds column SELECT leaves
    # the operator with a column list and no grantee. This walks the ACLs directly:
    # grantee 0 is PUBLIC (which every role holds, and which pg_has_role does not
    # accept as an argument), and any other grantee whose privileges journal_app
    # holds reaches journal_app -- measured on PostgreSQL 17, a column SELECT granted
    # to a parent role of journal_app survives every REVOKE naming journal_app.
    audit_log_inherited_select=""
    [[ "${AUDIT_LOG_PRESENT}" == "true" ]] && audit_log_inherited_select=$(q "
        SELECT COALESCE(string_agg(DISTINCT descr, ', ' ORDER BY descr), '')
        FROM (
            SELECT format('table SELECT via %s',
                          CASE WHEN a.grantee = 0 THEN 'PUBLIC'
                               ELSE a.grantee::regrole::text END) AS descr
            FROM pg_class c, aclexplode(c.relacl) AS a
            WHERE c.oid = 'public.audit_log'::regclass
              AND a.privilege_type = 'SELECT'
              AND a.grantee <> 'journal_app'::regrole
              AND (a.grantee = 0 OR pg_has_role('journal_app', a.grantee, 'USAGE'))
            UNION ALL
            SELECT format('column SELECT on %I via %s', at.attname,
                          CASE WHEN a.grantee = 0 THEN 'PUBLIC'
                               ELSE a.grantee::regrole::text END) AS descr
            FROM pg_attribute at, aclexplode(at.attacl) AS a
            WHERE at.attrelid = 'public.audit_log'::regclass
              AND at.attnum > 0
              AND a.privilege_type = 'SELECT'
              AND a.grantee <> 'journal_app'::regrole
              AND (a.grantee = 0 OR pg_has_role('journal_app', a.grantee, 'USAGE'))
        ) s
    ")
    if [[ -n "${audit_log_inherited_select}" ]]; then
        fail "inherited_grant audit_log: journal_app reaches SELECT through PUBLIC or an inherited role: ${audit_log_inherited_select}"
    fi
else
    fail_unverified journal_app \
        "grant tables: journal_app CRUD shape" \
        "grant users: journal_app SELECT/UPDATE only" \
        "grant audit_log: journal_app INSERT only" \
        "column_grant audit_log: journal_app dedup conflict-target columns" \
        "inherited_grant audit_log: journal_app effective SELECT exposure"
fi

if [[ "${JOURNAL_ADMIN_PRESENT}" == "true" ]]; then
    # audit_log: journal_admin has SELECT + INSERT only.
    # SELECT/INSERT pass via ownership AND via explicit grants in grants.sql /
    # baseline. UPDATE/DELETE are NOT explicitly granted -- but ownership
    # returns has_table_privilege=t regardless, so we introspect relacl via
    # assert_no_explicit_priv (see fork-point note above).
    for p in SELECT INSERT; do
        assert_priv journal_admin audit_log "$p" t
    done
    for p in UPDATE DELETE; do
        assert_no_explicit_priv journal_admin audit_log "$p"
    done

    # All other gubbi tables: journal_admin has full access.
    for t in "${JOURNAL_ADMIN_FULL[@]}"; do
        for p in SELECT INSERT UPDATE DELETE; do
            assert_priv journal_admin "$t" "$p" t
        done
    done
else
    fail_unverified journal_admin \
        "grant audit_log: journal_admin SELECT/INSERT" \
        "no_explicit_priv audit_log: journal_admin UPDATE/DELETE" \
        "grant tables: journal_admin full access"
fi

# ---------------------------------------------------------------------------
# 8. otel_ro role: exists, LOGIN, member of pg_monitor, NO grants on any
# gubbi-owned data table. Gated on the shared existence probe resolved up front,
# for the same reason sections 5b and 7 are.
# ---------------------------------------------------------------------------
if [[ "${OTEL_RO_PRESENT}" != "true" ]]; then
    fail "otel_ro_exists: expected true, got 'f'"
    fail_unverified otel_ro "otel_ro_login" "otel_ro_pg_monitor"
    for t in "${GUBBI_TENANT_TABLES[@]}"; do
        fail_unverified otel_ro "grant ${t}: otel_ro SELECT"
    done
else
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
fi

# ---------------------------------------------------------------------------
# 8b. Role posture: the four superuser / BYPASSRLS booleans, plus journal_admin's
# CREATEROLE and journal_app membership.
#
# rolsuper and rolbypassrls are INDEPENDENT catalog columns -- a superuser role
# still reports rolbypassrls = false while bypassing RLS anyway -- so each is
# asserted in its own right rather than inferred from the other.
#
#   journal_app    NOSUPERUSER, NOBYPASSRLS
#     Every user-facing connection authenticates as this role. Either boolean
#     turning true makes all the RLS and grant assertions above advisory: the
#     policies stay in the catalog and stop filtering rows.
#   journal_admin  NOSUPERUSER, BYPASSRLS
#     BYPASSRLS is REQUIRED (cross-tenant maintenance paths), so this is the one
#     posture boolean that fails when WEAKENED rather than broadened. SUPERUSER
#     is not required by any code path and would void every ACL check here.
#
# NO CREATEROLE on journal_admin removes an escalation surface no code path uses.
# Membership in journal_app is required by the gubbi 0009 grant chain; init.sh /
# grants.sql establishes it, and asserting it here catches a deploy that did not
# run init.sh.
# ---------------------------------------------------------------------------
if [[ "${JOURNAL_APP_PRESENT}" == "true" ]]; then
    assert_true "journal_app_not_superuser" \
        "SELECT NOT rolsuper FROM pg_roles WHERE rolname='journal_app'"
    assert_true "journal_app_no_bypassrls" \
        "SELECT NOT rolbypassrls FROM pg_roles WHERE rolname='journal_app'"
else
    fail_unverified journal_app "journal_app_not_superuser" "journal_app_no_bypassrls"
fi

if [[ "${JOURNAL_ADMIN_PRESENT}" == "true" ]]; then
    assert_true "journal_admin_not_superuser" \
        "SELECT NOT rolsuper FROM pg_roles WHERE rolname='journal_admin'"
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
else
    fail_unverified journal_admin \
        "journal_admin_not_superuser" \
        "journal_admin_bypassrls" \
        "journal_admin_no_createrole" \
        "journal_admin_member_of_journal_app"
fi

# ---------------------------------------------------------------------------
# 8b-ii. REACHABLE posture. The four booleans above describe journal_app's own
# catalog row, and role attributes are NOT inherited -- SUPERUSER and BYPASSRLS
# apply only to the role that holds them. So a journal_app whose own row is clean
# still defeats every RLS assertion above if it can come to EXECUTE AS a role
# that holds either attribute: one SET ROLE and the self-only policies stop
# constraining it.
#
# Reachability is actor-rooted and follows three distinct edge kinds, matching the
# semantics the cloud-side audit capability probe settled on:
#   * INHERIT  -- privileges apply without SET ROLE. Kept in the walk because
#                 ADMIN authority held by an INHERITED role is itself reachable
#                 (journal_app --INHERIT--> mid --ADMIN--> target is walkable),
#                 but a role reached by inheritance ALONE is not assumable: role
#                 ATTRIBUTES do not flow across an inherit edge.
#   * SET      -- an assumable role's SET edges are assumable, so SET chains walk.
#   * ADMIN    -- ADMIN authority on a role lets journal_app GRANT itself SET on
#                 it, so an ADMIN-reached role is assumable, and ADMIN results are
#                 themselves assumable so ADMIN chains compose.
# A plain pg_has_role(...,'USAGE') test sees only the INHERIT closure and reports
# clean for both the SET-only and the ADMIN-only escalation.
#
# Only ASSUMABLE roles are checked for a defeating attribute, for the same reason
# inheritance alone does not carry one.
#
# DIAGNOSTIC SHAPE. A role name is operator-supplied text: it may contain a
# newline, a quote, or a control character, and this string lands in a deploy log
# that is parsed line by line. So the diagnostic is BOUNDED and SINGLE-LINE:
#   * at most REACHABLE_POSTURE_SAMPLE_LIMIT names, ordered, so a cluster with
#     hundreds of reachable roles cannot emit an unbounded line;
#   * the exact TOTAL and the OMITTED count are always preserved, so truncation
#     never hides the scale of the finding;
#   * every C0 control byte (U+0000..U+001F), DEL (U+007F) and every C1 control
#     (U+0080..U+009F) is encoded as its EXACT \xHH value -- not a placeholder --
#     so the diagnostic is reversible: a reader can tell a newline (\x0A) from a
#     tab (\x09). A literal backslash is doubled first, so the encoding is
#     unambiguous.
#     The C1 range matters specifically for U+0085 NEXT LINE: Python's
#     str.splitlines() and many log readers treat it as a line break, and
#     ascii() reports 133 for it, so a `< 32 OR = 127` test lets it through.
#   * the two Unicode characters that also break lines -- LINE SEPARATOR (U+2028)
#     and PARAGRAPH SEPARATOR (U+2029) -- get their own explicit \u2028 / \u2029
#     encodings; they are outside the control ranges above and would otherwise
#     pass through and split the line.
# No role text can therefore forge a second log line or a FAIL: prefix.
# ---------------------------------------------------------------------------
REACHABLE_POSTURE_SAMPLE_LIMIT=5

_act_as_cte="
    WITH RECURSIVE _act_as(roleid, assumable) AS (
        SELECT oid, true FROM pg_roles WHERE rolname = 'journal_app'
      UNION
        SELECT m.roleid,
               (m.set_option AND a.assumable) OR m.admin_option
        FROM pg_auth_members m
        JOIN _act_as a ON m.member = a.roleid
        WHERE (m.set_option AND a.assumable)
           OR m.admin_option
           OR m.inherit_option
    )
"

if [[ "${JOURNAL_APP_PRESENT}" == "true" ]]; then
    # Returns "<total>|<shown>|<sample>".
    #
    # Encoding is EXACT and per character, so it is reversible: \x0A and \x09 stay
    # distinguishable rather than collapsing to one placeholder. The string is split
    # into characters, each mapped independently, then reassembled -- which is what
    # makes the hex value the ACTUAL byte rather than a literal backreference.
    # Branch order matters: the backslash case runs first so an encoded value can be
    # read back unambiguously, and U+2028 / U+2029 are matched before the control
    # range because they break log lines for readers while not being [[:cntrl:]].
    app_reachable_posture=$(q "${_act_as_cte},
        offenders AS (
            SELECT r.rolname AS rolname,
                   CASE WHEN r.rolsuper AND r.rolbypassrls THEN 'SUPERUSER + BYPASSRLS'
                        WHEN r.rolsuper THEN 'SUPERUSER'
                        ELSE 'BYPASSRLS' END AS capability
            FROM _act_as a
            JOIN pg_roles r ON r.oid = a.roleid
            WHERE a.assumable
              AND r.rolname <> 'journal_app'
              AND (r.rolsuper OR r.rolbypassrls)
        ),
        sampled AS (
            SELECT format('%s holds %s', encoded.safe_name, o.capability) AS descr
            FROM (
                SELECT rolname, capability FROM offenders ORDER BY rolname
                LIMIT ${REACHABLE_POSTURE_SAMPLE_LIMIT}
            ) o
            CROSS JOIN LATERAL (
                SELECT COALESCE(string_agg(
                           CASE
                               WHEN ch = '\\' THEN '\\\\'
                               WHEN ascii(ch) = 8232 THEN '\\u2028'
                               WHEN ascii(ch) = 8233 THEN '\\u2029'
                               WHEN ascii(ch) < 32
                                    OR (ascii(ch) >= 127 AND ascii(ch) <= 159)
                                   THEN '\\x' || lpad(upper(to_hex(ascii(ch))), 2, '0')
                               ELSE ch
                           END, '' ORDER BY ord), '') AS safe_name
                FROM regexp_split_to_table(o.rolname, '') WITH ORDINALITY AS s(ch, ord)
            ) encoded
        )
        SELECT format('%s|%s|%s',
                      (SELECT COUNT(*) FROM offenders),
                      (SELECT COUNT(*) FROM sampled),
                      (SELECT COALESCE(string_agg(descr, ', '), '') FROM sampled))
    ")
    reachable_total=${app_reachable_posture%%|*}
    _rest=${app_reachable_posture#*|}
    reachable_shown=${_rest%%|*}
    reachable_sample=${_rest#*|}
    if [[ "${reachable_total}" != "0" ]]; then
        fail "reachable_posture journal_app: can assume $(printf '%s' "${reachable_total}") role(s) holding SUPERUSER or BYPASSRLS (showing ${reachable_shown}, omitted $((reachable_total - reachable_shown))): ${reachable_sample}"
    fi
else
    fail_unverified journal_app "reachable_posture journal_app"
fi

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
if [[ "${JOURNAL_APP_PRESENT}" == "true" && "${JOURNAL_ADMIN_PRESENT}" == "true" ]]; then
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
else
    fail_unverified "journal_app or journal_admin" \
        "grant alembic_version: role revocation shape"
fi

# ---------------------------------------------------------------------------
# 10. Default privileges (FOR ROLE journal_admin IN SCHEMA public) -- exact
# ACL shape, not just count. The audited shape:
#   TABLES:    journal_app -> SELECT,INSERT,UPDATE  (NO DELETE)
#              journal_admin -> ALL
#   SEQUENCES: journal_app -> SELECT,USAGE
#              journal_admin -> ALL
# Strict equality on per-grantee privilege set so regressions are caught.
# ---------------------------------------------------------------------------
# ::regrole raises on a missing role, so the default-priv shape is gated too.
if [[ "${JOURNAL_APP_PRESENT}" == "true" && "${JOURNAL_ADMIN_PRESENT}" == "true" ]]; then
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
else
    fail_unverified "journal_app or journal_admin" \
        "default_privs_tables: audited default-privilege shape"
fi

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
if [[ ${PASS} == "true" && ${#FAILURES[@]} -eq 0 ]]; then
    echo "verify-db-invariants: OK -- 8 gubbi tables, 10 policies, exact audit_log SELECT policy set (effective through PUBLIC + inherited roles), 3 triggers, audit_log dedup column grants, otel_ro role, app + admin superuser/BYPASSRLS posture including reachable-role escalation, alembic_version revocation, default-priv shape all match audited baseline"
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
