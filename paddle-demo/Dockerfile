FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libgomp1 ccache \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PADDLE_DEVICE=cpu \
    PADDLE_LANG=en \
    PADDLE_LAYOUT_MODEL=PP-DocLayout_plus-L \
    PADDLE_ENABLE_MKLDNN=1 \
    INFERENCE_WORKERS=2 \
    CPU_THREADS_PER_WORKER=1

WORKDIR /app

COPY requirements-cpu.txt .
RUN pip install -r requirements-cpu.txt

COPY app/ ./app/

# Pre-download the pipeline weights at build time so replica cold starts do not
# fetch ~1 GB of models over the network.
COPY download_models.py .
RUN python download_models.py

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
