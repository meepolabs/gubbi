"""Alembic environment configuration — sync mode with raw SQL migrations.

The database URL is read from the JOURNAL_DB_APP_URL environment variable.
We do not use SQLAlchemy model reflection (autogenerate) because our schema
source of truth is the raw SQL file at journalctl/storage/schema.sql.
"""

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


def get_database_url() -> str:
    """Read the database URL from JOURNAL_DB_APP_URL environment variable.

    Returns the URL with the psycopg driver scheme for SQLAlchemy.
    """
    url = os.environ.get("JOURNAL_DB_APP_URL")
    if not url:
        raise RuntimeError(
            "JOURNAL_DB_APP_URL is not set. "
            "Set it to your PostgreSQL connection string, e.g.: "
            "postgresql+psycopg://user:pass@host:5432/dbname"
        )
    # Ensure we use the psycopg (async-friendly sync driver)
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    elif url.startswith("postgresql+asyncpg://"):
        url = url.replace("postgresql+asyncpg://", "postgresql+psycopg://", 1)
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
