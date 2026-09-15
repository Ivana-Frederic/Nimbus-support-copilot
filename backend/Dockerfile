FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Installing CPU-only torch from the official CPU wheel index first.
# Keeps the image several hundred MB smaller than the default PyPI wheel.
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.5.1" && \
    pip install -r requirements.txt

COPY app ./app
COPY scripts ./scripts
COPY data ./data

RUN mkdir -p /data

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
