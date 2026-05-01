import os
import threading

from arq.connections import RedisSettings

from journalctl.extraction.health import app as health_app


def _build_redis_settings() -> RedisSettings:
    url = os.environ.get("JOURNAL_REDIS_URL")
    if url:
        return RedisSettings.from_dsn(url)
    return RedisSettings(host="localhost", port=6379)


async def startup(ctx: dict) -> None:
    health_thread = threading.Thread(
        target=_run_health_server,
        daemon=True,
    )
    health_thread.start()
    ctx["health_thread"] = health_thread


def _run_health_server() -> None:
    import uvicorn  # noqa: PLC0415

    uvicorn.run(health_app, host="0.0.0.0", port=8201, log_level="info")  # noqa: S104


async def placeholder_job(ctx: dict) -> None:
    """Placeholder job -- will be replaced with real extraction jobs."""
    pass


class WorkerSettings:
    redis_settings = _build_redis_settings()
    functions = [placeholder_job]
    on_startup = startup
    max_jobs = 10
    job_timeout = 600
    keep_result = 86400
    poll_delay = 0.5
