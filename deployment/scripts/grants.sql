-- grants.sql -- Canonical GRANT/REVOKE state for the journal database.
--
-- Single source of truth for privilege state at the squashed-baseline head
-- (gubbi 0001_squashed_baseline + gubbi-cloud 0001_squashed_baseline) PLUS
-- the post-audit tightening (outbox_events / stripe_events / alembic_version
-- / users_self_update / default-priv DELETE) -- so a `--repair-grants` run
-- against a deployed instance settles on the intended state, not the
-- pg_dump-captured chain state.
--
-- Used by:
--   deployment/restore-db.sh --repair-grants (psql -f grants.sql)
--   tests/integration/test_grants_contract.py (contract assertions)
--
-- Run against a superuser DSN:
--   psql -v ON_ERROR_STOP=1 -f grants.sql <JOURNAL_DB_SUPERUSER_URL>
--
-- This file ASSUMES tables already exist (it is post-migration repair, not
-- bootstrap). It does NOT create roles -- those are pre-created by
-- testbench config/postgres/init.sh and prod gubbi-stack/postgres-init.sh.

BEGIN;

-- ---------------------------------------------------------------------------
-- Schema access
-- ---------------------------------------------------------------------------

GRANT ALL ON SCHEMA public TO journal_admin;
GRANT USAGE ON SCHEMA public TO journal_app;

GRANT journal_app TO journal_admin WITH ADMIN OPTION;

-- ---------------------------------------------------------------------------
-- Default privileges for FUTURE tables / sequences created by the journal
-- superuser. Tightened from the chain capture: DELETE removed from the
-- journal_app default. Migrations that need DELETE for journal_app must
-- now opt in with an explicit GRANT DELETE. Eliminates the historical
-- "forgot to REVOKE DELETE" failure mode on admin-only tables.
-- ---------------------------------------------------------------------------

ALTER DEFAULT PRIVILEGES FOR ROLE journal_admin IN SCHEMA public
    GRANT SELECT, INSERT, UPDATE ON TABLES TO journal_app;
ALTER DEFAULT PRIVILEGES FOR ROLE journal_admin IN SCHEMA public
    GRANT ALL ON TABLES TO journal_admin;
ALTER DEFAULT PRIVILEGES FOR ROLE journal_admin IN SCHEMA public
    GRANT SELECT, USAGE ON SEQUENCES TO journal_app;
ALTER DEFAULT PRIVILEGES FOR ROLE journal_admin IN SCHEMA public
    GRANT ALL ON SEQUENCES TO journal_admin;

-- ---------------------------------------------------------------------------
-- Reset every existing table / sequence back to a known baseline so the
-- explicit GRANTs below are the final word (immune to whatever stale
-- privileges may have accreted from prior repair runs).
-- ---------------------------------------------------------------------------

REVOKE ALL ON ALL TABLES IN SCHEMA public FROM journal_app, journal_admin;
REVOKE ALL ON ALL SEQUENCES IN SCHEMA public FROM journal_app, journal_admin;

-- ---------------------------------------------------------------------------
-- gubbi tables: full CRUD for journal_app, ALL for journal_admin
-- (audit_log is the append-only exception, narrowed at the end.)
-- ---------------------------------------------------------------------------

GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE public.topics            TO journal_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE public.entries           TO journal_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE public.conversations     TO journal_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE public.messages          TO journal_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE public.entry_embeddings  TO journal_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE public.extraction_jobs   TO journal_app;

GRANT ALL ON TABLE public.topics            TO journal_admin;
GRANT ALL ON TABLE public.entries           TO journal_admin;
GRANT ALL ON TABLE public.conversations     TO journal_admin;
GRANT ALL ON TABLE public.messages          TO journal_admin;
GRANT ALL ON TABLE public.entry_embeddings  TO journal_admin;
GRANT ALL ON TABLE public.extraction_jobs   TO journal_admin;

-- users: narrowed -- journal_app gets SELECT + UPDATE only (no INSERT, no DELETE).
-- Original chain narrowed in gubbi migration 0019 (rls_users) to enforce that
-- user rows are written via journal_admin / Hydra subject flow only.
GRANT SELECT, UPDATE ON TABLE public.users TO journal_app;
GRANT ALL              ON TABLE public.users TO journal_admin;

-- ---------------------------------------------------------------------------
-- gubbi-cloud user-data tables: narrower set -- INSERT/SELECT/UPDATE only
-- for journal_app (no DELETE on billing surfaces; deletion is audit-retained
-- by policy).
-- ---------------------------------------------------------------------------

GRANT SELECT, INSERT, UPDATE ON TABLE public.tenants        TO journal_app;
GRANT SELECT, INSERT, UPDATE ON TABLE public.subscriptions  TO journal_app;
GRANT SELECT, INSERT, UPDATE ON TABLE public.llm_budgets    TO journal_app;

GRANT ALL ON TABLE public.tenants        TO journal_admin;
GRANT ALL ON TABLE public.subscriptions  TO journal_admin;
GRANT ALL ON TABLE public.llm_budgets    TO journal_admin;

-- ---------------------------------------------------------------------------
-- gubbi-cloud admin-only tables: NO journal_app access.
-- outbox_events: producers (Kratos + Stripe webhooks) and the relay worker
-- both run under db_pool (journal_admin); no user-facing path touches it.
-- stripe_events: webhook idempotency markers; webhook dispatcher only.
-- ---------------------------------------------------------------------------

GRANT ALL ON TABLE public.outbox_events  TO journal_admin;
GRANT ALL ON TABLE public.stripe_events  TO journal_admin;
-- explicit REVOKE so journal_app has nothing here, regardless of whether
-- a prior run had granted via the older default-priv DELETE-included shape:
REVOKE ALL ON TABLE public.outbox_events FROM journal_app;
REVOKE ALL ON TABLE public.stripe_events FROM journal_app;
REVOKE ALL ON SEQUENCE public.outbox_events_id_seq FROM journal_app;

-- ---------------------------------------------------------------------------
-- Alembic version tables: alembic itself runs under journal_admin only;
-- journal_app has no operational need to read or write these. REVOKE so
-- prior --repair-grants runs that pre-dated this audit are cleaned up.
-- ---------------------------------------------------------------------------

GRANT ALL ON TABLE public.alembic_version        TO journal_admin;
GRANT ALL ON TABLE public.alembic_version_cloud  TO journal_admin;
REVOKE ALL ON TABLE public.alembic_version       FROM journal_app;
REVOKE ALL ON TABLE public.alembic_version_cloud FROM journal_app;

-- ---------------------------------------------------------------------------
-- Sequences (one per identity column + outbox_events_id_seq, which is
-- already revoked above)
-- ---------------------------------------------------------------------------

GRANT SELECT, USAGE ON ALL SEQUENCES IN SCHEMA public TO journal_app;
GRANT ALL           ON ALL SEQUENCES IN SCHEMA public TO journal_admin;
-- ALL SEQUENCES grant above re-includes outbox_events_id_seq for journal_app;
-- restore the admin-only posture explicitly:
REVOKE ALL ON SEQUENCE public.outbox_events_id_seq FROM journal_app;

-- ---------------------------------------------------------------------------
-- audit_log: append-only least-privilege (gubbi migration 0010).
--   journal_app: INSERT + column-level SELECT on the five dedup
--                conflict-target columns (no table-wide SELECT, no
--                UPDATE / DELETE)
--   journal_admin: SELECT + INSERT only (no UPDATE / DELETE)
-- This section MUST come AFTER the broad gubbi-table grants -- it narrows
-- the broad grants for audit_log specifically.
-- ---------------------------------------------------------------------------

REVOKE ALL ON TABLE public.audit_log FROM journal_app;
GRANT INSERT ON TABLE public.audit_log TO journal_app;

-- PUBLIC is implicitly held by journal_app, so a table- or column-level grant
-- to PUBLIC widens the effective read surface while every journal_app-scoped
-- ACL check still reports the narrow contract. A REVOKE naming journal_app
-- does not touch a PUBLIC grant, so revoke it in its own right.
REVOKE ALL ON TABLE public.audit_log FROM PUBLIC;

-- The canonical deduped audit INSERT infers its conflict target from the
-- partial unique index audit_log_content_hash_uidx, so Postgres reads these
-- five columns while evaluating ON CONFLICT. Both the column grant and the
-- self-only SELECT policy below are required -- neither works alone, and
-- the REVOKE ALL above clears column-level grants too, so a repair run
-- without this block silently removes the capability and every
-- user-attributed deduped write starts failing with insufficient_privilege.
-- Table-wide SELECT stays denied: actor_type, occurred_at, reason,
-- ip_address and user_agent remain unreadable through journal_app.
--
-- Column-level SELECT is cleared per column before the regrant rather than
-- relying on the table-level REVOKE ALL above to do it. This file's job is to
-- make the end state the final word regardless of what accreted before it, and
-- the column loop is derived from the catalog so a column added by a later
-- migration -- or hand-granted during an incident -- is narrowed back out
-- without editing this file.
--
-- PUBLIC is cleared per column alongside journal_app: measured on PostgreSQL 17,
-- a column-level GRANT ... TO PUBLIC makes has_column_privilege('journal_app',
-- ...) true and survives every REVOKE that names journal_app.
DO $$
DECLARE
    col text;
BEGIN
    FOR col IN
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = 'audit_log'
    LOOP
        EXECUTE format(
            'REVOKE SELECT (%I) ON TABLE public.audit_log FROM journal_app', col
        );
        EXECUTE format(
            'REVOKE SELECT (%I) ON TABLE public.audit_log FROM PUBLIC', col
        );
    END LOOP;
END $$;

GRANT SELECT (actor_id, target_kind, target_id, action, metadata)
    ON TABLE public.audit_log TO journal_app;

-- Mirrors the audit_log_app_insert_self_only WITH CHECK predicate exactly, so
-- an author's read surface is precisely the set of rows it may write.
-- Idempotent: repair runs replay this file against a live instance.
DROP POLICY IF EXISTS audit_log_app_select_self_only ON public.audit_log;
CREATE POLICY audit_log_app_select_self_only ON public.audit_log
    FOR SELECT TO journal_app
    USING (
        actor_id = (SELECT NULLIF(current_setting('app.current_user_id', true), ''))
        AND actor_id <> ''
        AND actor_type = 'user'
    );

REVOKE ALL ON TABLE public.audit_log FROM journal_admin;
GRANT SELECT, INSERT ON TABLE public.audit_log TO journal_admin;

-- ---------------------------------------------------------------------------
-- Fail closed on audit_log read exposure this file cannot safely remove.
--
-- Everything above narrows what journal_app and PUBLIC hold DIRECTLY. Two
-- widenings survive that, because journal_app's effective privileges include
-- everything held by every role it is a member of:
--
--   * a table- or column-level SELECT grant on audit_log to a PARENT role of
--     journal_app -- has_column_privilege('journal_app', ...) reports true and
--     no REVOKE naming journal_app or PUBLIC clears it (measured on PG 17);
--   * a SELECT-applicable policy scoped TO a parent role or TO PUBLIC -- it
--     applies to journal_app regardless of journal_app's own rolinherit
--     setting (measured on PG 17 with journal_app NOINHERIT).
--
-- Repairing either would mean mutating a role or dropping a policy this file
-- does not own -- a parent role may exist for a legitimate reason elsewhere in
-- the cluster, and silently stripping it is a wider blast radius than the
-- exposure. So this block RAISES instead: the whole repair transaction rolls
-- back, and the operator gets the exact grantee or policy to resolve. Reporting
-- a successful repair while journal_app can read every audit row is the one
-- outcome that must be impossible.
--
-- The expected policies are pinned to the literal self-only predicate, not only
-- to equality with each other: a mutation widening BOTH to `true` keeps them
-- mirrored while exposing every row.
-- ---------------------------------------------------------------------------

DO $$
DECLARE
    _expected_select_policy CONSTANT text := 'audit_log_app_select_self_only';
    _expected_insert_policy CONSTANT text := 'audit_log_app_insert_self_only';
    -- The self-only contract, as written in this file's CREATE POLICY above and
    -- in the migration that creates the INSERT policy.
    _contract CONSTANT text :=
        $c$actor_id = (SELECT NULLIF(current_setting('app.current_user_id', true), ''))
           AND actor_id <> '' AND actor_type = 'user'$c$;
    _offenders text;
BEGIN
    -- Effective SELECT grants reaching journal_app other than its own.
    -- grantee 0 is PUBLIC; pg_has_role() does not accept it, so it gets its own
    -- arm. journal_app itself is excluded -- its direct grants are this file's
    -- own end state.
    SELECT string_agg(DISTINCT descr, ', ' ORDER BY descr) INTO _offenders
    FROM (
        SELECT format('table SELECT granted to %s',
                      CASE WHEN a.grantee = 0 THEN 'PUBLIC'
                           ELSE a.grantee::regrole::text END) AS descr
        FROM pg_class c, aclexplode(c.relacl) AS a
        WHERE c.oid = 'public.audit_log'::regclass
          AND a.privilege_type = 'SELECT'
          AND a.grantee <> 'journal_app'::regrole
          AND (a.grantee = 0 OR pg_has_role('journal_app', a.grantee, 'USAGE'))
        UNION ALL
        SELECT format('column SELECT on %I granted to %s', at.attname,
                      CASE WHEN a.grantee = 0 THEN 'PUBLIC'
                           ELSE a.grantee::regrole::text END) AS descr
        FROM pg_attribute at, aclexplode(at.attacl) AS a
        WHERE at.attrelid = 'public.audit_log'::regclass
          AND at.attnum > 0
          AND a.privilege_type = 'SELECT'
          AND a.grantee <> 'journal_app'::regrole
          AND (a.grantee = 0 OR pg_has_role('journal_app', a.grantee, 'USAGE'))
    ) s;

    IF _offenders IS NOT NULL THEN
        RAISE EXCEPTION
            'audit_log read exposure reaches journal_app through a grantee this '
            'repair cannot safely revoke: %. journal_app inherits every '
            'privilege of every role it belongs to, so this widens its audit '
            'read surface beyond the five dedup conflict-target columns. '
            'Resolve the grant at its source (REVOKE it from that grantee, or '
            'remove the role membership) and re-run this file.', _offenders;
    END IF;

    -- SELECT-applicable policies ('r' = SELECT, '*' = ALL) that reach
    -- journal_app directly, through PUBLIC, or through a parent role. Anything
    -- other than the one expected policy OR-widens the read surface.
    SELECT string_agg(polname, ', ' ORDER BY polname) INTO _offenders
    FROM pg_policy p
    WHERE p.polrelid = 'public.audit_log'::regclass
      AND p.polcmd IN ('r', '*')
      AND p.polname <> _expected_select_policy
      AND EXISTS (
          SELECT 1 FROM unnest(p.polroles) AS r(oid)
          WHERE r.oid = 0 OR pg_has_role('journal_app', r.oid, 'USAGE')
      );

    IF _offenders IS NOT NULL THEN
        RAISE EXCEPTION
            'audit_log carries SELECT-applicable policies beyond %, reaching '
            'journal_app directly or through PUBLIC/an inherited role: %. '
            'Permissive policies for the same command are OR-ed, so these '
            'widen journal_app''s audit read surface. Drop them at their '
            'source and re-run this file.', _expected_select_policy, _offenders;
    END IF;

    -- Both expected predicates, pinned independently to the literal contract.
    -- pg_get_expr reprints in its own canonical form, so compare normalized:
    -- whitespace, ::text casts, parens and the subquery output alias removed.
    SELECT string_agg(format('%s (%s)', which, coalesce(got, '<missing>')), ', ' ORDER BY which)
      INTO _offenders
    FROM (
        SELECT 'USING of ' || _expected_select_policy AS which,
               (SELECT pg_get_expr(polqual, polrelid) FROM pg_policy
                WHERE polrelid = 'public.audit_log'::regclass
                  AND polname = _expected_select_policy) AS got
        UNION ALL
        SELECT 'WITH CHECK of ' || _expected_insert_policy AS which,
               (SELECT pg_get_expr(polwithcheck, polrelid) FROM pg_policy
                WHERE polrelid = 'public.audit_log'::regclass
                  AND polname = _expected_insert_policy) AS got
    ) s
    WHERE regexp_replace(regexp_replace(coalesce(got, ''), '\s|::text|[()]', '', 'g'),
                         'AS"nullif"', '', 'g')
       <> regexp_replace(regexp_replace(_contract, '\s|::text|[()]', '', 'g'),
                         'AS"nullif"', '', 'g');

    IF _offenders IS NOT NULL THEN
        RAISE EXCEPTION
            'audit_log self-only predicate does not match the contract: %. The '
            'contract is: %. Both the SELECT USING and the INSERT WITH CHECK '
            'must be exactly this -- mirroring each other is not enough, since '
            'widening both to true keeps them mirrored while exposing every '
            'row. Re-run `alembic upgrade head` to restore the migration-owned '
            'INSERT policy.', _offenders, _contract;
    END IF;
END $$;

COMMIT;
