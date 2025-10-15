# syntax=docker/dockerfile:1.4

# Base image (Python 3.11 compatible with CUDA wheels)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (git for certain HF repos)
RUN apt-get update && apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Requirements (dev and prod variants)
COPY requirements-dev.txt requirements-dev.txt
COPY requirements-prod.txt requirements-prod.txt

# Build arg chooses which requirements to install (dev by default)
ARG BUILD_PROFILE=dev
ENV BUILD_PROFILE=${BUILD_PROFILE}

# Install dependencies with basic caching
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "$BUILD_PROFILE" = "prod" ]; then \
        pip install --upgrade pip && pip install -r requirements-prod.txt ; \
    else \
        pip install --upgrade pip && pip install -r requirements-dev.txt ; \
    fi

# Copy project files
COPY . .

# Default runtime env
ENV MODEL_PROFILE=dev
ENV MODEL_CACHE=/app/hf_models/cache

EXPOSE 8000

# Start API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]