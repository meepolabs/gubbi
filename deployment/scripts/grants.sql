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

ALTER DEFAULT PRIVILEGES FOR ROLE journal IN SCHEMA public
    GRANT SELECT, INSERT, UPDATE ON TABLES TO journal_app;
ALTER DEFAULT PRIVILEGES FOR ROLE journal IN SCHEMA public
    GRANT ALL ON TABLES TO journal_admin;
ALTER DEFAULT PRIVILEGES FOR ROLE journal IN SCHEMA public
    GRANT SELECT, USAGE ON SEQUENCES TO journal_app;
ALTER DEFAULT PRIVILEGES FOR ROLE journal IN SCHEMA public
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
--   journal_app: INSERT only (no SELECT / UPDATE / DELETE)
--   journal_admin: SELECT + INSERT only (no UPDATE / DELETE)
-- This section MUST come AFTER the broad gubbi-table grants -- it narrows
-- the broad grants for audit_log specifically.
-- ---------------------------------------------------------------------------

REVOKE ALL ON TABLE public.audit_log FROM journal_app;
GRANT INSERT ON TABLE public.audit_log TO journal_app;

REVOKE ALL ON TABLE public.audit_log FROM journal_admin;
GRANT SELECT, INSERT ON TABLE public.audit_log TO journal_admin;

COMMIT;
