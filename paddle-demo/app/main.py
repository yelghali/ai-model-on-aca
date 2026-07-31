"""
PaddleOCR PP-StructureV3 document-parsing API — wrapped in FastAPI.

Same concurrency architecture as the YOLO and EasyOCR demos (see
yolo-demo/app/main.py for the full rationale):
  * blocking inference is offloaded off the asyncio event loop
  * each worker thread borrows its own pipeline instance from a pool, because
    PaddleOCR pipeline objects hold mutable per-call state and are not safe to
    share across threads

Runs on CPU or GPU from the same image - set PADDLE_DEVICE=cpu|gpu.

Batching
--------
The workflow sends MANY images per call, so /parse accepts a batch of files and
hands the whole list to `pipeline.predict(...)` in one go. That is PaddleOCR's
native batching path: sub-models (layout detection, text detection, text
recognition) each process the batch together, which is markedly more efficient
than N separate HTTP calls - especially on GPU, where a batch fills the device
while a single 1-page request leaves it mostly idle.

  POST /parse  -f file=@a.png -f file=@b.png -f file=@c.png

A single file still works; it is just a batch of one.

Concurrency on top of batching
------------------------------
Batching handles parallelism WITHIN one request. The pool below handles
parallelism ACROSS simultaneous requests, and follows PaddleOCR's own guidance:
its reference script exposes `--instances_per_device` and, given multiple
devices, "creates an instance of the underlying pipeline class on each device".
A single pipeline object is not meant to serve concurrent callers.
https://www.paddleocr.ai/latest/en/version3.x/inference_deployment/local_inference/parallel_inference.html

Upstream uses multiprocessing rather than threads. Threads are cheaper (one
runtime, N model copies) but only parallelise if Paddle releases the GIL during
inference. Watch `peak_running` in /stats: if it never exceeds 1 while
`inference_workers` is 2, the GIL is the bottleneck and the answer is processes
instead - `uvicorn --workers N`, or INFERENCE_WORKERS=1 plus more ACA replicas.

IMPORTANT sizing note for GPU:
  Pooling duplicates model weights. On CPU that is cheap. On a single GPU it
  consumes N x VRAM while the device serialises kernels anyway, so the default
  for GPU is a pool of 1, a LARGER batch, and scale-out via replicas.

Endpoints:
  GET  /         -> service info
  GET  /healthz  -> liveness/readiness probe (used by ACA)
  GET  /stats    -> live concurrency + memory counters
  POST /parse    -> parse a BATCH of document images (repeat the `file` field)
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
from typing import List

import psutil
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

DEVICE = os.getenv("PADDLE_DEVICE", "cpu").lower()
LANG = os.getenv("PADDLE_LANG", "en")
LAYOUT_MODEL = os.getenv("PADDLE_LAYOUT_MODEL", "PP-DocLayout_plus-L")
LAYOUT_THRESHOLD = float(os.getenv("PADDLE_LAYOUT_THRESHOLD", "0.3"))
# MKL-DNN meaningfully speeds up CPU inference; keep it on unless told otherwise.
ENABLE_MKLDNN = os.getenv("PADDLE_ENABLE_MKLDNN", "1") not in ("0", "false", "False")

# On GPU, extra pipeline copies cost VRAM without buying parallelism.
_default_workers = "1" if DEVICE == "gpu" else "2"
INFERENCE_WORKERS = int(os.getenv("INFERENCE_WORKERS", _default_workers))
CPU_THREADS_PER_WORKER = int(os.getenv("CPU_THREADS_PER_WORKER", "1"))
# Guard rail: reject absurd batches rather than OOM the replica.
MAX_BATCH = int(os.getenv("MAX_BATCH", "32"))

REPLICA = os.getenv("CONTAINER_APP_REPLICA_NAME") or socket.gethostname()

# Wall-clock at import. The gap between this and the moment the pool is ready is
# this replica's own start-up cost - i.e. "how long does a new instance take".
# Reported per replica in /stats as startup_ms, so a soak test can capture the
# real cost of every scale-out event without guessing from the outside.
PROCESS_START = time.time()

state: dict = {
    "pool": None, "executor": None, "ready": False,
    "rss_baseline_mb": None, "startup_ms": None, "ready_at": None,
}

_lock = threading.Lock()
_counters = {
    "in_flight": 0, "peak_in_flight": 0,
    "running": 0, "peak_running": 0,
    "batches": 0, "images": 0, "failures": 0,
}


def _acquire(n_images: int):
    with _lock:
        _counters["in_flight"] += 1
        _counters["batches"] += 1
        _counters["images"] += n_images
        if _counters["in_flight"] > _counters["peak_in_flight"]:
            _counters["peak_in_flight"] = _counters["in_flight"]


def _release():
    with _lock:
        _counters["in_flight"] -= 1


def _fail():
    with _lock:
        _counters["failures"] += 1


def _start_running():
    with _lock:
        _counters["running"] += 1
        if _counters["running"] > _counters["peak_running"]:
            _counters["peak_running"] = _counters["running"]


def _stop_running():
    with _lock:
        _counters["running"] -= 1


def _build_pipeline():
    """One PP-StructureV3 pipeline, configured for layout + text only."""
    from paddleocr import PPStructureV3

    return PPStructureV3(
        lang=LANG,
        layout_detection_model_name=LAYOUT_MODEL,
        # Keep the pipeline lean: layout + OCR only. Each extra sub-pipeline
        # adds another model to load and another pass over the page.
        use_table_recognition=False,
        use_chart_recognition=False,
        use_formula_recognition=False,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        layout_threshold=LAYOUT_THRESHOLD,
        layout_merge_bboxes_mode="union",
        enable_mkldnn=ENABLE_MKLDNN,
        cpu_threads=CPU_THREADS_PER_WORKER,
        device=DEVICE,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    import numpy as np

    pool: "queue.Queue" = queue.Queue()
    warm = np.full((1024, 768, 3), 255, dtype=np.uint8)
    for _ in range(INFERENCE_WORKERS):
        pipe = _build_pipeline()
        # Warm up so the first real request doesn't pay lazy model loading.
        try:
            list(pipe.predict(warm))
        except Exception:
            # A blank page can legitimately yield nothing; the point is that the
            # weights are now loaded and the graph is built.
            pass
        pool.put(pipe)

    state["pool"] = pool
    state["executor"] = ThreadPoolExecutor(
        max_workers=INFERENCE_WORKERS, thread_name_prefix="infer"
    )
    state["rss_baseline_mb"] = round(psutil.Process().memory_info().rss / 1024 / 1024, 1)
    state["ready_at"] = time.time()
    state["startup_ms"] = round((state["ready_at"] - PROCESS_START) * 1000, 1)
    state["ready"] = True
    yield
    state["ready"] = False
    state["executor"].shutdown(wait=True)
    state["executor"] = None
    state["pool"] = None


app = FastAPI(
    title="PP-StructureV3 Document Parsing API",
    description="PaddleOCR PP-StructureV3 wrapped in FastAPI, with a thread-safe pipeline pool.",
    version="1.0.0",
    lifespan=lifespan,
)


def _summarise(res) -> dict:
    """Pull a compact, JSON-safe summary out of a PP-StructureV3 result."""
    try:
        data = res.json if isinstance(getattr(res, "json", None), dict) else None
    except Exception:
        data = None
    if data is None:
        try:
            data = dict(res)
        except Exception:
            data = {}

    payload = data.get("res", data) if isinstance(data, dict) else {}

    blocks = []
    for key in ("parsing_res_list", "layout_det_res", "overall_ocr_res"):
        val = payload.get(key)
        if isinstance(val, dict):
            val = val.get("boxes") or val.get("rec_texts") or []
        if isinstance(val, list) and val:
            for item in val:
                if isinstance(item, dict):
                    blocks.append({
                        "label": item.get("label") or item.get("block_label"),
                        "score": round(float(item["score"]), 4) if isinstance(item.get("score"), (int, float)) else None,
                        "text": (item.get("block_content") or item.get("text") or "")[:400],
                    })
            if blocks:
                break

    texts = payload.get("overall_ocr_res", {})
    rec_texts = texts.get("rec_texts", []) if isinstance(texts, dict) else []

    return {
        "blocks": blocks[:80],
        "block_count": len(blocks),
        "text": " ".join(t for t in rec_texts)[:4000],
        "text_item_count": len(rec_texts),
    }


def _infer_batch(images: list) -> dict:
    """Run a whole batch through ONE pipeline instance, on a worker thread.

    `pipeline.predict(list)` is PaddleOCR's native batching path: each sub-model
    processes the batch together instead of once per image.
    """
    pool: "queue.Queue" = state["pool"]
    pipe = pool.get()  # borrow - guarantees no two threads share a pipeline
    _start_running()
    try:
        t0 = time.perf_counter()
        results = list(pipe.predict(images))
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    finally:
        _stop_running()
        pool.put(pipe)  # always return it, even on failure

    pages = [_summarise(r) for r in results]
    n = max(len(images), 1)
    return {
        "pages": pages,
        "batch_size": len(images),
        "returned": len(pages),
        "inference_ms": elapsed_ms,
        "inference_ms_per_image": round(elapsed_ms / n, 1),
    }


@app.get("/")
def info():
    return {
        "service": "paddle-ppstructurev3",
        "device": DEVICE,
        "layout_model": LAYOUT_MODEL,
        "lang": LANG,
        "replica": REPLICA,
        "inference_workers": INFERENCE_WORKERS,
        "cpu_threads_per_worker": CPU_THREADS_PER_WORKER,
        "enable_mkldnn": ENABLE_MKLDNN,
        "max_batch": MAX_BATCH,
        "endpoints": {
            "parse": "POST /parse  (repeat the `file` field to send a batch)",
            "health": "GET /healthz",
            "stats": "GET /stats",
        },
    }


@app.get("/healthz")
def healthz():
    return {"status": "ok" if state["ready"] else "loading", "ready": state["ready"]}


@app.get("/stats")
def stats():
    """Deliberately small: the numbers a soak test actually needs.

    startup_ms is this replica's own cold-start cost, measured from process
    import to the moment all pipelines were loaded and warmed. Collecting it
    across replicas gives the real "time to create a new instance".
    """
    with _lock:
        c = dict(_counters)
    now = time.time()
    return {
        "replica": REPLICA,
        "device": DEVICE,
        "workers": INFERENCE_WORKERS,
        # this instance
        "startup_ms": state["startup_ms"],
        "uptime_s": round(now - PROCESS_START, 1),
        "rss_mb": round(psutil.Process().memory_info().rss / 1024 / 1024, 1),
        # work done by this replica
        "batches": c["batches"],
        "images": c["images"],
        "failures": c["failures"],
        # concurrency right now / high-water mark
        "in_flight": c["in_flight"],
        "peak_running": c["peak_running"],
    }


@app.post("/parse")
async def parse(file: List[UploadFile] = File(...)):
    """Parse a batch of document images in a single call.

    Send the `file` field once per image:
        curl -X POST .../parse -F "file=@p1.png" -F "file=@p2.png"
    """
    if not state["ready"]:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    if not file:
        raise HTTPException(status_code=400, detail="No files uploaded")
    if len(file) > MAX_BATCH:
        raise HTTPException(
            status_code=413,
            detail=f"Batch of {len(file)} exceeds MAX_BATCH={MAX_BATCH}",
        )

    import numpy as np

    images, names = [], []
    for f in file:
        if not (f.content_type or "").startswith("image/"):
            raise HTTPException(status_code=400, detail=f"{f.filename}: not an image")
        raw = await f.read()
        try:
            images.append(np.array(Image.open(io.BytesIO(raw)).convert("RGB")))
        except Exception:
            raise HTTPException(status_code=400, detail=f"{f.filename}: could not decode")
        names.append(f.filename)

    _acquire(len(images))
    t0 = time.perf_counter()
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(state["executor"], _infer_batch, images)
    except Exception:
        _fail()
        raise
    finally:
        _release()
    total_ms = round((time.perf_counter() - t0) * 1000, 1)

    # Attach the original filename to each page so callers can correlate.
    for name, page in zip(names, result["pages"]):
        page["filename"] = name

    result["total_ms"] = total_ms
    result["total_ms_per_image"] = round(total_ms / max(len(images), 1), 1)
    result["queue_wait_ms"] = round(total_ms - result["inference_ms"], 1)
    result["replica"] = REPLICA
    result["device"] = DEVICE
    result["startup_ms"] = state["startup_ms"]
    return result
