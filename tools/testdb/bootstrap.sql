-- Role, extension and schema-grant bootstrap for a fresh test cluster.
--
-- Run as the cluster superuser, against the database the migrations will
-- target, BEFORE `alembic upgrade head`: the 0001 baseline grants to
-- journal_app and journal_admin and assumes both already exist. Every
-- statement is idempotent; running this file twice leaves the same state.
--
--   psql -v ON_ERROR_STOP=1 [-v name=value ...] -d <db> -f tools/testdb/bootstrap.sql
--
-- psql variables (the contract every consumer relies on; do not rename):
--
--   admin_createrole  default: true. Set by: -v admin_createrole=false.
--       Whether journal_admin holds CREATEROLE. The deploy posture (and
--       deployment/scripts/verify-db-invariants.sh) requires NOCREATEROLE;
--       the default keeps the posture the gubbi test suite has always run on.
--       Applied on every run, so an existing role converges to the value.
--
--   grant_app_to_admin  default: false. Set by: -v grant_app_to_admin=true.
--       Whether to run GRANT journal_app TO journal_admin WITH ADMIN OPTION,
--       which the deploy posture expects (PostgreSQL 16+ requires ADMIN
--       OPTION for journal_admin to re-grant or manage journal_app). Only
--       adds the grant; false never revokes one already present.
--
--   app_password  default: unset (password left unchanged).
--       Set by: -v app_password=..., else env JOURNAL_DB_APP_PASSWORD.
--   admin_password  default: unset (password left unchanged).
--       Set by: -v admin_password=..., else env JOURNAL_DB_ADMIN_PASSWORD.
--       Prefer the env form for real secrets: -v puts the value on the psql
--       argv, which other local users can read from the process table.
--
--   with_otel_ro  default: false. Set by: -v with_otel_ro=true.
--       Create the read-only telemetry role otel_ro (LOGIN, pg_monitor, no
--       data grants). The baseline migration does not create it.
--   otel_ro_password  default: unset (password left unchanged).
--       Set by: -v otel_ro_password=..., else env PG_OTEL_RO_PASSWORD.
--       Only read when with_otel_ro is true.
--
-- Boolean variables accept any PostgreSQL boolean spelling (true/false,
-- on/off, yes/no, 1/0); any other value aborts the run.

\set ON_ERROR_STOP on

\if :{?admin_createrole}
\else
    \set admin_createrole true
\endif
\if :{?grant_app_to_admin}
\else
    \set grant_app_to_admin false
\endif
\if :{?with_otel_ro}
\else
    \set with_otel_ro false
\endif
-- An unparseable value in a psql \if is reported but NOT stopped on by
-- ON_ERROR_STOP; it silently takes the \else branch. Casting through SQL
-- makes a typo fatal, and \gset normalizes each value to t/f.
SELECT :'admin_createrole'::boolean AS admin_createrole,
       :'grant_app_to_admin'::boolean AS grant_app_to_admin,
       :'with_otel_ro'::boolean AS with_otel_ro
\gset

\if :{?app_password}
\else
    \getenv app_password JOURNAL_DB_APP_PASSWORD
\endif
\if :{?admin_password}
\else
    \getenv admin_password JOURNAL_DB_ADMIN_PASSWORD
\endif

CREATE EXTENSION IF NOT EXISTS vector;

-- psql does not interpolate variables inside dollar-quoted bodies, so the DO
-- block only creates the roles; every variable-driven attribute is applied
-- by the ALTER ROLE statements after it.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'journal_app') THEN
        CREATE ROLE journal_app LOGIN;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'journal_admin') THEN
        CREATE ROLE journal_admin LOGIN BYPASSRLS;
    END IF;
END $$;

ALTER ROLE journal_app WITH LOGIN NOSUPERUSER NOBYPASSRLS;
ALTER ROLE journal_admin WITH LOGIN NOSUPERUSER BYPASSRLS;

\if :admin_createrole
    ALTER ROLE journal_admin WITH CREATEROLE;
\else
    ALTER ROLE journal_admin WITH NOCREATEROLE;
\endif

\if :{?app_password}
    ALTER ROLE journal_app WITH PASSWORD :'app_password';
\endif
\if :{?admin_password}
    ALTER ROLE journal_admin WITH PASSWORD :'admin_password';
\endif

-- PostgreSQL 15+ makes schema public writable by its owner only, so without
-- this the first migration cannot create its bookkeeping table.
GRANT ALL PRIVILEGES ON SCHEMA public TO journal_admin;
GRANT USAGE ON SCHEMA public TO journal_app;

\if :grant_app_to_admin
    GRANT journal_app TO journal_admin WITH ADMIN OPTION;
\endif

\if :with_otel_ro
    \if :{?otel_ro_password}
    \else
        \getenv otel_ro_password PG_OTEL_RO_PASSWORD
    \endif
    DO $$
    BEGIN
        IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'otel_ro') THEN
            CREATE ROLE otel_ro LOGIN;
        END IF;
    END $$;
    ALTER ROLE otel_ro WITH LOGIN NOSUPERUSER NOBYPASSRLS;
    \if :{?otel_ro_password}
        ALTER ROLE otel_ro WITH PASSWORD :'otel_ro_password';
    \endif
    GRANT pg_monitor TO otel_ro;
\endif
