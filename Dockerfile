FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Add small set of system libraries commonly needed for ML/Wheels
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libgomp1 \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Use the server-specific requirements (smaller, no GUI libs)
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
