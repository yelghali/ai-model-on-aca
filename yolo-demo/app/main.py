"""
YOLO object-detection API — an OSS model wrapped in FastAPI.

Concurrency design (the interesting part for the ACA scaling demo)
------------------------------------------------------------------
Two things commonly make a YOLO API serve exactly ONE image at a time:

1. Blocking the event loop.
   `async def` + a blocking CPU call (`model.predict(...)`) runs inference *on the
   asyncio event loop thread*. While it runs, the process cannot accept, parse or
   answer anything else — not even /healthz. Throughput collapses to 1 concurrent
   request no matter how many CPU cores the container has.
   Fix: never run blocking inference inside `async def`; offload it to a thread.

2. Ultralytics YOLO instances are NOT thread-safe.
   A single `YOLO` object keeps mutable internal state (`model.predictor` and its
   batch/result buffers) that is reused on every call. Sharing one instance across
   threads causes races and cross-contaminated results.
   Fix: one model instance PER worker thread, borrowed from a pool.

So: a bounded ThreadPoolExecutor of N workers plus a Queue of N model instances.
Each in-flight request borrows a dedicated model, so N inferences genuinely run in
parallel. Torch intra-op threads are capped per worker so the workers don't
oversubscribe the CPU and slow each other down.

Endpoints:
  GET  /         -> service info
  GET  /healthz  -> liveness/readiness probe (used by ACA)
  GET  /stats    -> live concurrency counters (useful when narrating a load test)
  POST /predict  -> run detection on an uploaded image
"""
import asyncio
import io
import os
import queue
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import torch
import psutil
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from ultralytics import YOLO

MODEL_NAME = os.getenv("YOLO_MODEL", "yolov8n.pt")
CONF = float(os.getenv("YOLO_CONF", "0.25"))
IMGSZ = int(os.getenv("YOLO_IMGSZ", "640"))

# How many inferences may run in parallel inside ONE replica.
INFERENCE_WORKERS = int(os.getenv("INFERENCE_WORKERS", str(os.cpu_count() or 2)))
# Torch intra-op threads per worker. workers * threads should be <= container CPUs,
# otherwise workers fight over cores and every request gets slower.
TORCH_THREADS_PER_WORKER = int(os.getenv("TORCH_THREADS_PER_WORKER", "1"))

REPLICA = os.getenv("CONTAINER_APP_REPLICA_NAME") or socket.gethostname()

state: dict = {"pool": None, "executor": None, "ready": False, "rss_baseline_mb": None}

# Live counters so the demo can show queueing vs. real parallelism.
#   in_flight = accepted requests (queued OR executing)
#   running   = requests actually inside model.predict() right now
# peak_running is the honest proof of parallelism: it can never exceed the pool size.
_lock = threading.Lock()
_counters = {
    "in_flight": 0, "peak_in_flight": 0,
    "running": 0, "peak_running": 0,
    "total": 0,
}


def _acquire():
    with _lock:
        _counters["in_flight"] += 1
        _counters["total"] += 1
        if _counters["in_flight"] > _counters["peak_in_flight"]:
            _counters["peak_in_flight"] = _counters["in_flight"]


def _release():
    with _lock:
        _counters["in_flight"] -= 1


def _start_running():
    with _lock:
        _counters["running"] += 1
        if _counters["running"] > _counters["peak_running"]:
            _counters["peak_running"] = _counters["running"]


def _stop_running():
    with _lock:
        _counters["running"] -= 1


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Cap intra-op parallelism BEFORE building the models.
    torch.set_num_threads(TORCH_THREADS_PER_WORKER)

    pool: "queue.Queue[YOLO]" = queue.Queue()
    warm = Image.new("RGB", (IMGSZ, IMGSZ), (128, 128, 128))
    for _ in range(INFERENCE_WORKERS):
        model = YOLO(MODEL_NAME)
        # Warm up so the first real request doesn't pay lazy-init cost and so this
        # instance's predictor buffers are already allocated.
        model.predict(warm, imgsz=IMGSZ, conf=CONF, verbose=False)
        pool.put(model)

    state["pool"] = pool
    state["executor"] = ThreadPoolExecutor(
        max_workers=INFERENCE_WORKERS, thread_name_prefix="infer"
    )
    # Memory after runtime + all pooled weights are loaded, before serving traffic.
    state["rss_baseline_mb"] = round(
        psutil.Process().memory_info().rss / 1024 / 1024, 1
    )
    state["ready"] = True
    yield
    state["ready"] = False
    state["executor"].shutdown(wait=True)
    state["executor"] = None
    state["pool"] = None


app = FastAPI(
    title="YOLO Object Detection API",
    description="Ultralytics YOLO wrapped in FastAPI, with a thread-safe model pool.",
    version="2.0.0",
    lifespan=lifespan,
)


def _infer(image: Image.Image) -> dict:
    """Runs on a worker thread with its own dedicated model instance."""
    pool: "queue.Queue[YOLO]" = state["pool"]
    model = pool.get()  # borrow — guarantees no two threads share a model
    _start_running()
    try:
        t0 = time.perf_counter()
        results = model.predict(image, imgsz=IMGSZ, conf=CONF, verbose=False)
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    finally:
        _stop_running()
        pool.put(model)  # always return it, even on failure

    r = results[0]
    names = r.names
    detections = [
        {
            "label": names.get(int(b.cls[0]), str(int(b.cls[0]))),
            "confidence": round(float(b.conf[0]), 4),
            "box_xyxy": [round(float(v), 1) for v in b.xyxy[0].tolist()],
        }
        for b in r.boxes
    ]
    return {"detections": detections, "inference_ms": elapsed_ms}


@app.get("/")
def info():
    return {
        "service": "yolo-object-detection",
        "model": MODEL_NAME,
        "conf": CONF,
        "imgsz": IMGSZ,
        "replica": REPLICA,
        "inference_workers": INFERENCE_WORKERS,
        "torch_threads_per_worker": TORCH_THREADS_PER_WORKER,
        "endpoints": {
            "predict": "POST /predict",
            "health": "GET /healthz",
            "stats": "GET /stats",
        },
    }


@app.get("/healthz")
def healthz():
    # Stays responsive under full load because inference never runs on this thread.
    return {"status": "ok" if state["ready"] else "loading", "ready": state["ready"]}


@app.get("/stats")
def stats():
    with _lock:
        snapshot = dict(_counters)
    return {
        "replica": REPLICA,
        "inference_workers": INFERENCE_WORKERS,
        # queued OR executing
        "in_flight": snapshot["in_flight"],
        "peak_in_flight": snapshot["peak_in_flight"],
        # actually inside model.predict() -> proof of real parallelism
        "running": snapshot["running"],
        "peak_running": snapshot["peak_running"],
        "total_requests": snapshot["total"],
        # Resident set size: what ACA actually counts against the memory limit.
        # Compare rss_mb at rest vs. under load to separate the fixed cost
        # (runtime + pooled weights) from per-inference activation memory.
        "rss_mb": round(psutil.Process().memory_info().rss / 1024 / 1024, 1),
        "rss_baseline_mb": state["rss_baseline_mb"],
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not state["ready"]:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image file")

    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image")

    _acquire()
    t0 = time.perf_counter()
    try:
        loop = asyncio.get_running_loop()
        # Offload to the pool: the event loop stays free to accept more requests.
        result = await loop.run_in_executor(state["executor"], _infer, image)
    finally:
        _release()
    total_ms = round((time.perf_counter() - t0) * 1000, 1)

    return {
        "count": len(result["detections"]),
        "detections": result["detections"],
        "inference_ms": result["inference_ms"],
        # total_ms - inference_ms = time spent waiting for a free worker.
        "total_ms": total_ms,
        "queue_wait_ms": round(total_ms - result["inference_ms"], 1),
        "replica": REPLICA,
    }
