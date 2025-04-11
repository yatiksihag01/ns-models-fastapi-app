# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.13
FROM python:${PYTHON_VERSION} as base

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set workdir
WORKDIR /

# Pre-install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Leverage Docker cache for Python deps
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

# Copy app source
COPY . .

# Use non-root user
USER root

# Expose FastAPI port
EXPOSE 8080

# Launch API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
