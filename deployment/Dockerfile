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

# Install system deps (git for daily_sync.sh if run inside container)
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y curl build-essential git && \
    rm -rf /var/lib/apt/lists/*

# Install poetry
RUN curl -sSL https://install.python-poetry.org | python -

# Copy dependency files
WORKDIR $PYSETUP_PATH
COPY poetry.lock pyproject.toml ./

# Install runtime deps
RUN poetry install --only main

# Copy source
COPY . /src
WORKDIR /src

# Create non-root user and log directory
RUN adduser --disabled-password --gecos "" appuser && \
    mkdir -p /app/logs && \
    chown -R appuser:appuser /src /app/logs
USER appuser

EXPOSE 8100

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8100/health || exit 1

CMD ["gunicorn", "journalctl.main:server", \
    "--workers", "4", \
    "--worker-class", "uvicorn.workers.UvicornWorker", \
    "--bind", "0.0.0.0:8100", \
    "--timeout", "300", \
    "--keep-alive", "120", \
    "--log-level", "info"]
