FROM python:3.11-slim

# OpenCV (pulled in by ultralytics) needs these shared libs.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    YOLO_MODEL=yolov8n.pt \
    YOLO_CONF=0.25 \
    YOLO_IMGSZ=640 \
    # Keep Ultralytics from writing config/telemetry to a non-writable home.
    YOLO_CONFIG_DIR=/tmp/Ultralytics

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app/ ./app/

# Pre-download the weights at build time so cold starts don't fetch over the network.
RUN python -c "from ultralytics import YOLO; YOLO('${YOLO_MODEL}')"

EXPOSE 8000
# Single worker: the model is loaded per-process; scale out via ACA replicas, not workers.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
