# syntax=docker/dockerfile:1
# LLM Companion Discord Bot Dockerfile
# Based on uv's official Alpine image for fast dependency management

FROM ghcr.io/astral-sh/uv:alpine3.22

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Copy dependency files first for better caching
COPY pyproject.toml uv.lock ./

# Install dependencies (without the project itself)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

# Copy source code
COPY src/ ./src/
COPY main.py ./
COPY README.md ./

# Install the project
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

# Create logs directory
RUN mkdir -p /app/logs

# Run the bot
CMD ["uv", "run", "python", "-m", "llmcompanioncord"]
