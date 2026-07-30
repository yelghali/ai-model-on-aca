"""
EasyOCR text-recognition API — an OSS OCR model wrapped in FastAPI.

Same concurrency design as the YOLO demo (see yolo-demo/app/main.py for the full
rationale): blocking inference is offloaded off the asyncio event loop, and each
worker thread borrows its own `easyocr.Reader` instance from a pool, because a
Reader holds mutable per-call state and is not safe to share across threads.

Endpoints:
  GET  /         -> service info
  GET  /healthz  -> liveness/readiness probe (used by ACA)
  GET  /stats    -> live concurrency counters
  POST /ocr      -> read text from an uploaded image
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

import easyocr
import numpy as np
import psutil
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

LANGS = [s.strip() for s in os.getenv("OCR_LANGS", "en").split(",") if s.strip()]
MIN_CONF = float(os.getenv("OCR_MIN_CONF", "0.2"))

# Each Reader holds its own copy of the detection + recognition weights, so this
# also multiplies memory. Keep it modest (2 is a good default for 4Gi).
INFERENCE_WORKERS = int(os.getenv("INFERENCE_WORKERS", "2"))
TORCH_THREADS_PER_WORKER = int(os.getenv("TORCH_THREADS_PER_WORKER", "1"))

REPLICA = os.getenv("CONTAINER_APP_REPLICA_NAME") or socket.gethostname()

state: dict = {"pool": None, "executor": None, "ready": False, "rss_baseline_mb": None}

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
    torch.set_num_threads(TORCH_THREADS_PER_WORKER)

    pool: "queue.Queue[easyocr.Reader]" = queue.Queue()
    warm = np.zeros((64, 64, 3), dtype=np.uint8)
    for _ in range(INFERENCE_WORKERS):
        reader = easyocr.Reader(LANGS, gpu=False, verbose=False)
        reader.readtext(warm)  # warm up this instance
        pool.put(reader)

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
    title="EasyOCR API",
    description="EasyOCR wrapped in FastAPI, with a thread-safe reader pool.",
    version="2.0.0",
    lifespan=lifespan,
)


def _infer(image: np.ndarray) -> dict:
    """Runs on a worker thread with its own dedicated Reader instance."""
    pool: "queue.Queue[easyocr.Reader]" = state["pool"]
    reader = pool.get()
    _start_running()
    try:
        t0 = time.perf_counter()
        results = reader.readtext(image)
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    finally:
        _stop_running()
        pool.put(reader)

    blocks = [
        {
            "text": text,
            "confidence": round(float(conf), 4),
            "box": [[round(float(x), 1), round(float(y), 1)] for x, y in box],
        }
        for box, text, conf in results
        if float(conf) >= MIN_CONF
    ]
    return {"blocks": blocks, "inference_ms": elapsed_ms}


@app.get("/")
def info():
    return {
        "service": "easyocr-text-recognition",
        "languages": LANGS,
        "min_confidence": MIN_CONF,
        "replica": REPLICA,
        "inference_workers": INFERENCE_WORKERS,
        "torch_threads_per_worker": TORCH_THREADS_PER_WORKER,
        "endpoints": {
            "ocr": "POST /ocr",
            "health": "GET /healthz",
            "stats": "GET /stats",
        },
    }


@app.get("/healthz")
def healthz():
    return {"status": "ok" if state["ready"] else "loading", "ready": state["ready"]}


@app.get("/stats")
def stats():
    with _lock:
        snapshot = dict(_counters)
    return {
        "replica": REPLICA,
        "inference_workers": INFERENCE_WORKERS,
        "in_flight": snapshot["in_flight"],
        "peak_in_flight": snapshot["peak_in_flight"],
        "running": snapshot["running"],
        "peak_running": snapshot["peak_running"],
        "total_requests": snapshot["total"],
        "rss_mb": round(psutil.Process().memory_info().rss / 1024 / 1024, 1),
        "rss_baseline_mb": state["rss_baseline_mb"],
    }


@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    if not state["ready"]:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image file")

    raw = await file.read()
    try:
        image = np.array(Image.open(io.BytesIO(raw)).convert("RGB"))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image")

    _acquire()
    t0 = time.perf_counter()
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(state["executor"], _infer, image)
    finally:
        _release()
    total_ms = round((time.perf_counter() - t0) * 1000, 1)

    return {
        "count": len(result["blocks"]),
        "text": " ".join(b["text"] for b in result["blocks"]),
        "blocks": result["blocks"],
        "inference_ms": result["inference_ms"],
        "total_ms": total_ms,
        "queue_wait_ms": round(total_ms - result["inference_ms"], 1),
        "replica": REPLICA,
    }
