# syntax=docker/dockerfile:1
FROM python:3.10.12-slim AS builder

# 1. Install uv from the official binary
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 2. System deps (only needed for building)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    pkg-config \
    g++ \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 3. Cache and install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

#Final Production Stage
FROM python:3.10.12-slim

WORKDIR /app

# 4. Create non-root user early
RUN useradd -m appuser && chown -R appuser /app

# 5. Copy the virtual environment from the builder
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv

# 6. Copy application code
COPY --chown=appuser:appuser . .

# 7. Setup Environment
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH=/app
USER appuser

EXPOSE 8501

# 8. Entrypoint
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]