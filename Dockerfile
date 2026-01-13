FROM python:3.12-slim

# Environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libgomp1 \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user early so we can use --chown on COPY
RUN useradd --create-home --shell /bin/bash appuser

# Copy requirements and install (cache busting when requirements.txt changes)
COPY --chown=appuser:appuser requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

# Copy app files as non-root user (photos and other large files should be excluded by .dockerignore)
COPY --chown=appuser:appuser . /app

# Switch to non-root user
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s CMD curl -f http://localhost:${PORT}/health || exit 1

# Run with gunicorn + uvicorn workers
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "2", "api:app", "-b", "0.0.0.0:8000"]