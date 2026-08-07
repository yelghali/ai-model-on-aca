# Multi-stage: the exporter needs torch + ultralytics (~500 MB), the runtime does
# not. Only the resulting .onnx graph crosses the stage boundary, so the final
# image ships onnxruntime alone.

# --- stage 1: export the graph ---------------------------------------------
FROM python:3.11-slim AS export

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    YOLO_MODEL=yolov8n.pt \
    YOLO_IMGSZ=640

WORKDIR /export

RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu \
        torch==2.5.1 torchvision==0.20.1 ultralytics==8.3.40 onnx==1.17.0

COPY export_model.py .
RUN python export_model.py

# --- stage 2: runtime -------------------------------------------------------
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_PATH=/app/model.onnx \
    YOLO_CONF=0.25 \
    YOLO_IOU=0.45 \
    YOLO_IMGSZ=640

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app/ ./app/
# Only the graph crosses over - none of torch/ultralytics.
COPY --from=export /export/yolov8n.onnx /app/model.onnx

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
