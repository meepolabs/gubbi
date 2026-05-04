FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    POETRY_VERSION=2.0.1 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    PYSETUP_PATH="/opt/pysetup" \
    VENV_PATH="/opt/pysetup/.venv"

ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y curl build-essential gosu && \
    rm -rf /var/lib/apt/lists/*

# Install poetry
RUN curl -sSL https://install.python-poetry.org | python -

# Copy dependency files
WORKDIR $PYSETUP_PATH
COPY poetry.lock pyproject.toml ./

# Install runtime deps — versions and hashes locked via poetry.lock
RUN poetry install --only main


RUN adduser --disabled-password --gecos "" appuser && \
    mkdir -p /app/logs /app/journal

# Copy source
COPY . /src
WORKDIR /src

RUN chown -R appuser:appuser /src /app /home/appuser

# Entrypoint handles permission fixing + privilege drop
# ONNX model (~80MB) downloads to /home/appuser/.cache on first container start,
# persisted via the onnx-model-cache Docker volume — survives all future rebuilds.
COPY deployment/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

EXPOSE 8100

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8100/health || exit 1

# No --preload: asyncpg pools must not be shared across os.fork().
# Each worker creates its own pool independently (2 workers × min_size=2 = 4 connections).
# ONNX model is pre-downloaded by entrypoint.sh; each worker loads from disk cache.
CMD ["gunicorn", "gubbi.main:server", \
    "--workers", "2", \
    "--worker-class", "uvicorn.workers.UvicornWorker", \
    "--bind", "0.0.0.0:8100", \
    "--timeout", "300", \
    "--keep-alive", "120", \
    "--log-level", "info", \
    "--access-logfile", "-", \
    "--access-logformat", "%(m)s %(U)s → %(s)s (%(D)sμs)"]
