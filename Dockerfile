
# syntax=docker/dockerfile:1.7

# --------------------------------------------------
# Base
# --------------------------------------------------
FROM python:3.12-slim-bookworm AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/home/app

WORKDIR /app

# ONNX Runtime / FastEmbed may require OpenMP runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------
# Dependencies
# --------------------------------------------------
FROM base AS dependencies

COPY requirements-runtime.txt .

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip \
    && pip install -r requirements-runtime.txt

# --------------------------------------------------
# Model download
# --------------------------------------------------
FROM dependencies AS models

RUN useradd --create-home --shell /usr/sbin/nologin app \
    && mkdir -p /home/app/.cache \
    && chown -R app:app /home/app

USER app

# Download both models into the default FastEmbed cache.
# RUN python -c "\
#     from fastembed import TextEmbedding; \
#     from fastembed.rerank.cross_encoder import TextCrossEncoder; \
#     TextEmbedding(model_name='BAAI/bge-small-en-v1.5'); \
#     TextCrossEncoder(model_name='Xenova/ms-marco-MiniLM-L-6-v2'); \
#     print('FastEmbed models downloaded successfully')"
RUN python - <<'PY'
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder

TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
TextCrossEncoder(model_name="Xenova/ms-marco-MiniLM-L-6-v2")

print("FastEmbed models downloaded successfully")
PY
# --------------------------------------------------
# Production runtime
# --------------------------------------------------
FROM base AS runtime

# Create an unprivileged application user.
RUN useradd --create-home --shell /usr/sbin/nologin app

COPY --from=dependencies /usr/local /usr/local
COPY --from=models --chown=app:app /home/app/.cache /home/app/.cache

WORKDIR /app

# Application source and migration files.
COPY --chown=app:app app/ ./app/
COPY --chown=app:app alembic/ ./alembic/
COPY --chown=app:app alembic.ini ./

ENV APP_ENV=prod \
    AGENT_MODE=prod

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
