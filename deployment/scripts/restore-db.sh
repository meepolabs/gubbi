#!/usr/bin/env bash
set -euo pipefail

# restore-db.sh -- Restore a PostgreSQL dump and verify invariants.
#
# Usage: restore-db.sh [--repair-grants] [--no-verify] <dump-file>
#
# Flags:
#   --repair-grants    Replay the canonical GRANT block from deployment/scripts/grants.sql
#                      and re-verify. Use after restoring a dump taken with
#                      --no-acl, which strips those grants.
#   --no-verify        Skip the post-restore invariant check. Discouraged: only
#                      use when you have verified separately outside this tool.
#
# Required environment:
#   JOURNAL_DB_SUPERUSER_URL  PostgreSQL superuser DSN used for pg_restore and
#                             GRANT replay (e.g. postgresql://journal:pass@host:5432/journal).
#
# Exit codes:
#   0  Success (restore + verify pass, or restore + repair-grants + verify pass)
#   1  Explicit failure (verify failed, verify-after-repair failed)
#   2  Usage error (bad flag, missing env var, missing dump file)
#   N  pg_restore or psql exit code propagated via set -e (connection errors, etc.)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
REPAIR_GRANTS=false
NO_VERIFY=false
DUMP_FILE=""

for arg in "$@"; do
    case "${arg}" in
        --no-acl)
            echo "ERROR: --no-acl is forbidden -- it strips GRANTs from migration 0002 (journal_app/journal_admin) and produces a server that 500s on first read. Run without --no-acl, or use --repair-grants after the fact." >&2
            exit 2
            ;;
        --repair-grants) REPAIR_GRANTS=true ;;
        --no-verify)     NO_VERIFY=true ;;
        -*)
            echo "ERROR: unknown flag '${arg}'" >&2
            echo "Usage: restore-db.sh [--repair-grants] [--no-verify] <dump-file>" >&2
            exit 2
            ;;
        *)
            if [[ -n "${DUMP_FILE}" ]]; then
                echo "ERROR: multiple dump files specified" >&2
                exit 2
            fi
            DUMP_FILE="${arg}"
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Validate required inputs
# ---------------------------------------------------------------------------
if [[ -z "${DUMP_FILE}" ]]; then
    echo "ERROR: no dump file specified" >&2
    echo "Usage: restore-db.sh [--repair-grants] [--no-verify] <dump-file>" >&2
    exit 2
fi

if [[ ! -f "${DUMP_FILE}" ]]; then
    echo "ERROR: dump file does not exist: ${DUMP_FILE}" >&2
    exit 2
fi

if [[ -z "${JOURNAL_DB_SUPERUSER_URL:-}" ]]; then
    echo "ERROR: JOURNAL_DB_SUPERUSER_URL is not set" >&2
    echo "Set it to the superuser DSN (e.g. postgresql://journal:pass@host:5432/journal)" >&2
    exit 2
fi

# ---------------------------------------------------------------------------
# Restore
# ---------------------------------------------------------------------------
echo "--- pg_restore: ${DUMP_FILE} ---"
pg_restore --clean --if-exists \
    --no-owner \
    -d "${JOURNAL_DB_SUPERUSER_URL}" \
    "${DUMP_FILE}"

echo "pg_restore completed."

# ---------------------------------------------------------------------------
# Post-restore: verify invariants (unless skipped)
# ---------------------------------------------------------------------------
GRANTS_REPAIRED="no"

run_verify() {
    # verify-db-invariants.sh resolves its DSN from JOURNAL_DB_MIGRATION_URL or
    # JOURNAL_DB_ADMIN_URL. If neither is set but JOURNAL_DB_SUPERUSER_URL is,
    # bridge the gap so the verify step does not exit 2.
    local db_url="${JOURNAL_DB_MIGRATION_URL:-${JOURNAL_DB_ADMIN_URL:-${JOURNAL_DB_SUPERUSER_URL:-}}}"
    JOURNAL_DB_MIGRATION_URL="${db_url}" bash "${SCRIPT_DIR}/verify-db-invariants.sh"
}

if [[ "${NO_VERIFY}" == "true" ]]; then
    echo "(skipped: --no-verify)"
else
    if run_verify; then
        echo "--- verify-db-invariants: pass ---"
    else
        # Verify failed. Check whether --repair-grants can fix it.
        if [[ "${REPAIR_GRANTS}" == "true" ]]; then
            echo "--- verify failed, replaying grants (--repair-grants) ---"

            psql -v ON_ERROR_STOP=1 -f "${SCRIPT_DIR}/grants.sql" "${JOURNAL_DB_SUPERUSER_URL}"

            GRANTS_REPAIRED="yes"
            echo "--- grants replayed, re-verifying ---"

            if ! run_verify; then
                echo "ERROR: verify still fails after grant repair. Inspect output above." >&2
                exit 1
            fi
        else
            echo "ERROR: post-restore verification failed." >&2
            echo "If your dump was taken with --no-acl, re-run with --repair-grants to replay grants from deployment/scripts/grants.sql." >&2
            exit 1
        fi
    fi
fi

echo "restore-db: OK -- restored ${DUMP_FILE}, verify=pass, grants_repaired=${GRANTS_REPAIRED}"
exit 0
