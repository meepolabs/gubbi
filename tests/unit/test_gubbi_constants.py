from gubbi.constants import (
    ADMIN_POOL_SIZE_MAX,
    ADMIN_POOL_SIZE_MIN,
    ANTHROPIC_MAX_RETRIES,
    APP_POOL_SIZE_MAX,
    APP_POOL_SIZE_MIN,
    ARQ_JOB_TIMEOUT_SECS,
    DB_COMMAND_TIMEOUT_SECS,
    DB_HEALTH_TIMEOUT_SECS,
    DEFAULT_PAGE_SIZE,
    EMBEDDING_REQUEST_TIMEOUT_SECS,
    HTTPX_CONNECT_TIMEOUT_SECS,
)


def test_constants_import_smoke() -> None:
    constants = (
        HTTPX_CONNECT_TIMEOUT_SECS,
        DB_COMMAND_TIMEOUT_SECS,
        DB_HEALTH_TIMEOUT_SECS,
        ARQ_JOB_TIMEOUT_SECS,
        EMBEDDING_REQUEST_TIMEOUT_SECS,
        ANTHROPIC_MAX_RETRIES,
        DEFAULT_PAGE_SIZE,
        APP_POOL_SIZE_MIN,
        APP_POOL_SIZE_MAX,
        ADMIN_POOL_SIZE_MIN,
        ADMIN_POOL_SIZE_MAX,
    )
    assert len(constants) == 11


def test_replaced_inline_literal_values() -> None:
    assert DB_COMMAND_TIMEOUT_SECS == 30
    assert APP_POOL_SIZE_MIN == 2
    assert APP_POOL_SIZE_MAX == 5
    assert ARQ_JOB_TIMEOUT_SECS == 600
    assert ANTHROPIC_MAX_RETRIES == 5


def test_all_constants_are_positive_numbers() -> None:
    constants = (
        HTTPX_CONNECT_TIMEOUT_SECS,
        DB_COMMAND_TIMEOUT_SECS,
        DB_HEALTH_TIMEOUT_SECS,
        ARQ_JOB_TIMEOUT_SECS,
        EMBEDDING_REQUEST_TIMEOUT_SECS,
        ANTHROPIC_MAX_RETRIES,
        DEFAULT_PAGE_SIZE,
        APP_POOL_SIZE_MIN,
        APP_POOL_SIZE_MAX,
        ADMIN_POOL_SIZE_MIN,
        ADMIN_POOL_SIZE_MAX,
    )
    assert all(isinstance(value, int | float) for value in constants)
    assert all(value > 0 for value in constants)
