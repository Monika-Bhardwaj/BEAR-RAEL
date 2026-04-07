FROM python:3.10-slim

# Metadata
LABEL maintainer="BEAR-RAEL Team"
LABEL description="Bayesian Embodied Autonomous Robotics Lab — OpenEnv Environment"

# Non-root user for HF Spaces compatibility
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install system deps (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY env/       ./env/
COPY tasks/     ./tasks/
COPY graders/   ./graders/
COPY api/       ./api/
COPY openenv.yaml .
COPY inference.py .

# Ensure packages are importable from /app
RUN touch env/__init__.py tasks/__init__.py graders/__init__.py api/__init__.py

# Switch to non-root
RUN chown -R appuser:appuser /app
USER appuser

# Expose FastAPI port (HF Spaces default)
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Start server
CMD ["python", "-m", "uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
