# Бэкенд на FastAPI

FROM python:3.11-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git

RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --no-cache --link-mode=copy -r requirements.txt

FROM python:3.11-slim

WORKDIR /app

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends curl

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

ENV HF_HOME=/root/.cache/huggingface \
    HF_DATASETS_CACHE=/root/.cache/huggingface/datasets \
    TRANSFORMERS_CACHE=/root/.cache/huggingface/transformers

COPY pyproject.toml .
COPY src/ ./src/
COPY tests/ ./tests/

EXPOSE 8000 8080

CMD ["uvicorn", "src.presentation.main:app", "--host", "0.0.0.0", "--port", "8000"]