"""Export yolov8n to ONNX at image build time.

Runs in a throwaway builder stage that has torch + ultralytics. The final image
keeps only onnxruntime, so the ~500 MB training framework never ships.

dynamic=True is what allows a variable batch dimension, which is required for
the batched /predict endpoint.
"""
import os

from ultralytics import YOLO

MODEL = os.getenv("YOLO_MODEL", "yolov8n.pt")
IMGSZ = int(os.getenv("YOLO_IMGSZ", "640"))

model = YOLO(MODEL)
path = model.export(format="onnx", imgsz=IMGSZ, dynamic=True, simplify=False, opset=12)
print("exported:", path)
