"""Alembic environment configuration -- sync mode with raw SQL migrations.

The migration DSN resolves via a fallback chain that prefers an explicit
admin/migration role over the runtime app role. We do not use SQLAlchemy
model reflection (autogenerate) because our schema source of truth is
the raw SQL file at gubbi/storage/schema.sql.
"""

import logging
import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# Alembic Config object
config = context.config

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = None

logger = logging.getLogger(__name__)


def get_database_url() -> str:
    """Resolve the migration DSN with a role-aware fallback chain.

    Preference order:
      1. ``JOURNAL_DB_MIGRATION_URL`` -- explicit migration DSN (preferred)
      2. ``JOURNAL_DB_ADMIN_URL``     -- admin role, equivalent for migrations
      3. ``JOURNAL_DB_APP_URL``       -- app role; lacks DDL privileges (deprecated)

    Falling back to ``JOURNAL_DB_APP_URL`` emits a deprecation warning;
    that role is least-privilege by design and migrations will fail with
    ``permission denied`` on schema-changing DDL. Returns the resolved
    URL rewritten to the psycopg driver scheme.
    """
    migration_url = os.environ.get("JOURNAL_DB_MIGRATION_URL")
    admin_url = os.environ.get("JOURNAL_DB_ADMIN_URL")
    app_url = os.environ.get("JOURNAL_DB_APP_URL")

    if migration_url:
        url, source = migration_url, "JOURNAL_DB_MIGRATION_URL"
    elif admin_url:
        url, source = admin_url, "JOURNAL_DB_ADMIN_URL"
    elif app_url:
        url, source = app_url, "JOURNAL_DB_APP_URL"
        logger.warning(
            "Resolved migration DSN from JOURNAL_DB_APP_URL. "
            "The app role is least-privilege and lacks DDL rights; "
            "migrations will fail on schema-changing operations. "
            "Set JOURNAL_DB_MIGRATION_URL (preferred) or "
            "JOURNAL_DB_ADMIN_URL to point at the admin role."
        )
    else:
        raise RuntimeError(
            "No migration DSN found. Set JOURNAL_DB_MIGRATION_URL "
            "(preferred) or JOURNAL_DB_ADMIN_URL to your PostgreSQL "
            "connection string, e.g.: "
            "postgresql+psycopg://journal_admin:pass@host:5432/journal"
        )

    # Normalize to the psycopg sync driver SQLAlchemy expects
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    elif url.startswith("postgresql+asyncpg://"):
        url = url.replace("postgresql+asyncpg://", "postgresql+psycopg://", 1)

    logger.info("Alembic using migration DSN from %s", source)
    return url


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    Generates SQL scripts without connecting to the database.
    """
    url = get_database_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    Connects to the database and executes migrations via SQLAlchemy.
    """
    config_section = config.get_section(config.config_ini_section, {}) or {}
    config_section["sqlalchemy.url"] = get_database_url()
    connectable = engine_from_config(
        config_section,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
