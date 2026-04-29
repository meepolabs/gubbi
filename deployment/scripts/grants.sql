-- grants.sql -- Canonical GRANT/REVOKE state for the journal database.
--
-- Single source of truth for privilege state after alembic upgrade head.
-- Replays migrations 0002, 0009, and 0010 privilege sections exactly.
-- Used by:
--   deployment/restore-db.sh --repair-grants (psql -f grants.sql)
--   tests/integration/test_grants_contract.py (contract assertions)
--
-- Run against a superuser DSN:
--   psql -v ON_ERROR_STOP=1 -f grants.sql <JOURNAL_DB_SUPERUSER_URL>

BEGIN;

-- ---------------------------------------------------------------------------
-- Migration 0002: journal_app runtime grants
-- ---------------------------------------------------------------------------

-- Schema access
GRANT USAGE ON SCHEMA public TO journal_app;

-- Per-table grants for tenant tables and users (SELECT, INSERT, UPDATE, DELETE)
GRANT SELECT, INSERT, UPDATE, DELETE ON topics TO journal_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON entries TO journal_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON conversations TO journal_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON messages TO journal_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON entry_embeddings TO journal_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON users TO journal_app;

-- Sequence grants (USAGE, SELECT) for all sequences in public schema
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO journal_app;

-- Default privileges for future tables/sequences
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO journal_app;
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT USAGE, SELECT ON SEQUENCES TO journal_app;

-- ---------------------------------------------------------------------------
-- Migration 0002: journal_admin migration/admin grants
-- ---------------------------------------------------------------------------

-- Schema access
GRANT ALL PRIVILEGES ON SCHEMA public TO journal_admin;

-- Per-table grants for tenant tables and users (ALL PRIVILEGES)
GRANT ALL PRIVILEGES ON topics TO journal_admin;
GRANT ALL PRIVILEGES ON entries TO journal_admin;
GRANT ALL PRIVILEGES ON conversations TO journal_admin;
GRANT ALL PRIVILEGES ON messages TO journal_admin;
GRANT ALL PRIVILEGES ON entry_embeddings TO journal_admin;
GRANT ALL PRIVILEGES ON users TO journal_admin;
GRANT ALL PRIVILEGES ON alembic_version TO journal_admin;

-- Sequence grants (ALL PRIVILEGES)
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO journal_admin;

-- Default privileges for future tables/sequences
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT ALL PRIVILEGES ON TABLES TO journal_admin;
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT ALL PRIVILEGES ON SEQUENCES TO journal_admin;

-- ---------------------------------------------------------------------------
-- Migration 0009: grant journal_admin ADMIN OPTION on journal_app
-- ---------------------------------------------------------------------------

GRANT journal_app TO journal_admin WITH ADMIN OPTION;

-- ---------------------------------------------------------------------------
-- Migration 0010: audit_log least-privilege (REVOKE then targeted GRANT)
--
-- audit_log is append-only. journal_app gets INSERT only.
-- journal_admin gets SELECT + INSERT only. Neither role gets UPDATE or DELETE.
-- This section MUST come last -- it narrows the broad grants above for audit_log.
-- ---------------------------------------------------------------------------

-- journal_app: INSERT only
REVOKE ALL ON audit_log FROM journal_app;
GRANT INSERT ON audit_log TO journal_app;

-- journal_admin: SELECT + INSERT only
REVOKE ALL ON audit_log FROM journal_admin;
GRANT SELECT, INSERT ON audit_log TO journal_admin;

COMMIT;
