FROM python:3.10-slim

LABEL maintainer="BEAR-RAEL Team"
LABEL description="Bayesian Embodied Autonomous Robotics Lab — OpenEnv Environment"

RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy pyproject.toml and lockfile
COPY pyproject.toml .
COPY uv.lock .

# Install dependencies using pup (or standard pip)
RUN pip install --no-cache-dir .

# Copy application code
COPY env/       ./env/
COPY tasks/     ./tasks/
COPY graders/   ./graders/
COPY server/    ./server/
COPY models.py  .
COPY openenv.yaml .
COPY inference.py .

# Ensure packages are importable
RUN touch env/__init__.py tasks/__init__.py graders/__init__.py server/__init__.py

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Start server using the entry point defined in pyproject.toml or direct uvicorn
CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
