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

# Install mcp-memory-service from PyPI, exact version + sha256 hash verified.
# --no-deps: skips torch/sentence-transformers (2.7GB); lightweight deps
# are provided by the Poetry install above (onnxruntime, tokenizers, sqlite-vec).
# To upgrade: bump version, update hash from https://pypi.org/pypi/mcp-memory-service/<ver>/json
COPY deployment/requirements-memory.txt ./
RUN pip install --no-deps --require-hashes -r requirements-memory.txt

# Pre-download ONNX embedding model at build time (~80MB).
# Cached in image layer — no network calls at container startup.
RUN python -c "from mcp_memory_service.embeddings.onnx_embeddings import ONNXEmbeddingModel; ONNXEmbeddingModel()"

# Copy source
COPY . /src
WORKDIR /src

# Create non-root user and writable directories.
RUN adduser --disabled-password --gecos "" appuser && \
    mkdir -p /app/logs /app/journal && \
    chown -R appuser:appuser /src /app

# Entrypoint handles permission fixing + privilege drop
COPY deployment/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

EXPOSE 8100

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8100/health || exit 1

# --preload loads app once in master, workers share model memory via fork (copy-on-write).
CMD ["gunicorn", "journalctl.main:server", \
    "--workers", "2", \
    "--preload", \
    "--worker-class", "uvicorn.workers.UvicornWorker", \
    "--bind", "0.0.0.0:8100", \
    "--timeout", "300", \
    "--keep-alive", "120", \
    "--log-level", "info", \
    "--access-logfile", "-", \
    "--access-logformat", "%(m)s %(U)s → %(s)s (%(D)sμs)"]
