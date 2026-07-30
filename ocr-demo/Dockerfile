FROM python:3.11-slim

# OpenCV (pulled in by easyocr) needs these shared libs.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OCR_LANGS=en \
    OCR_MIN_CONF=0.2

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app/ ./app/

# Pre-download detection + recognition weights at build time to avoid slow cold starts.
RUN python -c "import easyocr; easyocr.Reader(['en'], gpu=False, verbose=False)"

EXPOSE 8000
# Single worker: the model is loaded per-process; scale out via ACA replicas, not workers.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
