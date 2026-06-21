# =============================================================================
# StreamSentinel — Dockerfile
# =============================================================================
# Multi-stage build:
#   Stage 1 (builder): install Python dependencies into a venv
#   Stage 2 (runtime): copy only the venv + source — no build tools
#
# This keeps the final image small and avoids shipping compilers.
#
# Build:  docker build -t streamsentinel:latest -f deployment/Dockerfile .
# Run:    docker run -p 8000:8000 --env-file .env streamsentinel:latest
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Builder — install dependencies
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /app

# System build dependencies (needed to compile some Python extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create an isolated virtual environment inside the image
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install CPU-only PyTorch first (smaller than CUDA build; swap for GPU)
RUN pip install --no-cache-dir \
    torch==2.3.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 2: Runtime — lean final image
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

WORKDIR /app

# Runtime system libs only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Copy the pre-built venv from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy project source
COPY . .

# Create non-root user for security
RUN useradd -m -u 1000 streamsentinel && chown -R streamsentinel:streamsentinel /app
USER streamsentinel

# Expose FastAPI port
EXPOSE 8000

# Health check — pings the /health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

# Default command: run the FastAPI inference service
CMD ["uvicorn", "api.fastapi_service:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
