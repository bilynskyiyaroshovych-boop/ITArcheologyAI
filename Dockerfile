FROM python:3.12-slim


ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libgomp1 \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home --shell /bin/bash appuser

COPY --chown=appuser:appuser requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

COPY --chown=appuser:appuser . /app
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD curl -f http://localhost:${PORT}/health || exit 1

CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "2", "api:app", "-b", "0.0.0.0:8000"]