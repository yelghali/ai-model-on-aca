"""YOLO on ONNX Runtime — the same API as yolo-demo, without the model pool.

Why this variant exists
-----------------------
yolo-demo runs Ultralytics/PyTorch and therefore needs a POOL of N model objects:
a `YOLO` instance carries mutable predictor state, so two threads cannot share
one. That works (torch releases the GIL) but it costs N copies of the weights
and ships a ~500 MB training framework to run a 3.2 M-parameter model.

ONNX Runtime is different: **an InferenceSession is thread-safe**. `session.run()`
may be called concurrently from many threads against ONE session. So the pool
disappears entirely:

    yolo-demo     : N threads -> queue.Queue of N YOLO objects  -> N x weights
    yolo-onnx-demo: N threads -> ONE InferenceSession (shared)  -> 1 x weights

Fewer moving parts, less memory, and no borrow/return discipline to get wrong.

The trade-off is that Ultralytics' pre/post-processing no longer comes for free -
letterboxing and NMS are implemented here in numpy. That is the honest cost of
dropping the framework.

Threading
---------
Two independent knobs, and they must not be multiplied blindly:
  INFERENCE_WORKERS - concurrent requests in flight (thread pool size)
  ORT_INTRA_THREADS - threads ORT uses INSIDE one run()
`workers x intra` should stay <= container vCPUs or the workers fight for cores.

Endpoints match yolo-demo exactly so the same load tests drive both:
  GET  /         -> service info
  GET  /healthz  -> liveness/readiness probe
  GET  /stats    -> live concurrency counters + cold-start cost
  POST /predict  -> detection on one OR MANY images (batch)
"""
import ast
import asyncio
import io
import os
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import List

import numpy as np
import onnxruntime as ort
import psutil
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

# Wall-clock at import: everything before `ready` is this replica's cold start.
PROCESS_START = time.time()

MODEL_PATH = os.getenv("MODEL_PATH", "/app/model.onnx")
CONF = float(os.getenv("YOLO_CONF", "0.25"))
IOU = float(os.getenv("YOLO_IOU", "0.45"))
IMGSZ = int(os.getenv("YOLO_IMGSZ", "640"))
MAX_BATCH = int(os.getenv("MAX_BATCH", "32"))

INFERENCE_WORKERS = int(os.getenv("INFERENCE_WORKERS", "2"))
# ORT threads INSIDE a single run(). workers * intra should be <= container vCPUs.
ORT_INTRA_THREADS = int(os.getenv("ORT_INTRA_THREADS", "1"))

REPLICA = os.getenv("CONTAINER_APP_REPLICA_NAME") or socket.gethostname()

state: dict = {
    "session": None, "executor": None, "ready": False,
    "names": {}, "input_name": None,
    "rss_baseline_mb": None, "startup_ms": None, "providers": [],
}

# Live counters. Same contract as yolo-demo so the two are directly comparable.
#   in_flight = accepted requests (queued OR executing)
#   running   = requests actually inside session.run() right now
# peak_running proves real parallelism: if it never exceeds 1 while
# INFERENCE_WORKERS is 2, something is serialising us.
_lock = threading.Lock()
_counters = {
    "in_flight": 0, "peak_in_flight": 0,
    "running": 0, "peak_running": 0,
    "total": 0, "images": 0, "failures": 0,
}


def _acquire(n: int):
    """Request accepted (may still be queued). n = images in this batch."""
    with _lock:
        _counters["in_flight"] += 1
        _counters["total"] += 1
        _counters["images"] += n
        if _counters["in_flight"] > _counters["peak_in_flight"]:
            _counters["peak_in_flight"] = _counters["in_flight"]


def _release():
    """Request has left the system, successfully or not."""
    with _lock:
        _counters["in_flight"] -= 1


def _fail():
    """Inference raised. Counted separately from HTTP-level rejections."""
    with _lock:
        _counters["failures"] += 1


def _start_running():
    """Inference actually beginning. Gap from _acquire to here = queue wait."""
    with _lock:
        _counters["running"] += 1
        if _counters["running"] > _counters["peak_running"]:
            _counters["peak_running"] = _counters["running"]


def _stop_running():
    with _lock:
        _counters["running"] -= 1


# --- pre / post processing (what Ultralytics used to do for us) -------------
#
# A raw ONNX graph is ONLY the neural network. Everything Ultralytics wrapped
# around it has to be reimplemented here. The full round trip is:
#
#   JPEG bytes
#     -> _letterbox()  resize to 640x640 keeping aspect ratio, pad the rest
#     -> /255, HWC->CHW, stack           (the tensor the graph expects)
#     -> session.run()                   -> raw (84, 8400) predictions
#     -> _decode()     filter by confidence, cxcywh->xyxy, undo the letterbox,
#                      then _nms() to collapse duplicate boxes
#     -> [{label, confidence, box_xyxy}] in ORIGINAL pixel coordinates
#
# "84" = 4 box values + 80 COCO class scores. "8400" = candidate boxes the model
# emits per image (across its three detection scales). Almost all are junk; the
# confidence filter and NMS are what turn them into a handful of real detections.

def _letterbox(im: Image.Image, size: int):
    """Resize preserving aspect ratio, pad to square.

    The network needs a fixed 640x640 input, but stretching a 1280x720 photo to
    a square would distort it and hurt accuracy. So we scale by the SAME factor
    on both axes and pad the leftover with grey.

    Returns (padded_image, r, dw, dh) where r is the scale factor and dw/dh are
    the padding offsets - _decode needs all three to map boxes back to the
    original image.
    """
    w0, h0 = im.size
    r = min(size / w0, size / h0)          # scale so the LONGER side fits
    nw, nh = int(round(w0 * r)), int(round(h0 * r))
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    dw, dh = (size - nw) // 2, (size - nh) // 2   # centre it
    canvas.paste(im.resize((nw, nh), Image.BILINEAR), (dw, dh))
    return canvas, r, dw, dh


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list:
    """Non-Maximum Suppression: collapse overlapping boxes for the same object.

    The model fires several times around one object. Greedy NMS keeps the
    highest-scoring box, deletes everything overlapping it by more than
    `iou_thr`, and repeats with what remains.

    IoU (Intersection over Union) = overlap area / combined area, so 0 means
    disjoint and 1 means identical. boxes are xyxy.
    """
    if boxes.size == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]          # strongest first
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))                 # winner survives
        if order.size == 1:
            break
        rest = order[1:]
        # intersection rectangle between the winner and every remaining box
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_thr]        # drop everything it suppressed
    return keep


def _decode(pred: np.ndarray, r: float, dw: int, dh: int, w0: int, h0: int) -> list:
    """Turn ONE image's raw model output into usable detections.

    pred is (84, 8400): 84 values for each of 8400 candidate boxes.
      rows 0-3  = cx, cy, w, h   (box centre + size, in 640x640 letterboxed space)
      rows 4-83 = one score per COCO class

    Note there is no separate "objectness" score in YOLOv8 - a box is only as
    good as its best class score.

    r/dw/dh come from _letterbox and are needed to undo it; w0/h0 clamp the
    result to the original image bounds.
    """
    p = pred.T                      # (8400, 84) - one row per candidate box

    # Best class per box, and how confident it is.
    cls_scores = p[:, 4:]
    cls_ids = cls_scores.argmax(1)
    confs = cls_scores[np.arange(cls_scores.shape[0]), cls_ids]

    # Throw away the ~99% of candidates that are noise.
    m = confs >= CONF
    if not m.any():
        return []
    boxes, cls_ids, confs = p[m, :4], cls_ids[m], confs[m]

    # centre+size -> corners, still in letterboxed 640x640 space
    xy = np.empty_like(boxes)
    xy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2      # x1
    xy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2      # y1
    xy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2      # x2
    xy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2      # y2

    # Undo _letterbox: subtract the padding, then divide by the scale factor.
    # Without this every box would be wrong for any non-square input image.
    xy[:, [0, 2]] = (xy[:, [0, 2]] - dw) / r
    xy[:, [1, 3]] = (xy[:, [1, 3]] - dh) / r
    xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, w0)     # a box may spill past the edge
    xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, h0)

    # Class-aware NMS. Shifting each class into its own coordinate band means
    # boxes of DIFFERENT classes can never overlap, so a person standing in
    # front of a car cannot suppress the car. One NMS pass, no per-class loop.
    offset = cls_ids.astype(np.float32) * 8192.0
    keep = _nms(xy + offset[:, None], confs, IOU)

    names = state["names"]
    return [
        {
            "label": names.get(int(cls_ids[i]), str(int(cls_ids[i]))),
            "confidence": round(float(confs[i]), 4),
            "box_xyxy": [round(float(v), 1) for v in xy[i]],
        }
        for i in keep
    ]


def _infer_batch(payload: tuple) -> dict:
    """Run one batch. Executes on a worker thread, NOT the event loop.

    Contrast with yolo-demo, which must borrow a model from a pool here:
    onnxruntime's InferenceSession.run() is thread-safe, so every worker shares
    the SAME session. No pool, no borrow/return, one copy of the weights.

    run() releases the GIL for its whole duration (it is a C++ call), which is
    what lets N Python threads genuinely execute in parallel.
    """
    tensor, metas = payload
    sess = state["session"]
    _start_running()
    try:
        t0 = time.perf_counter()
        # None = "give me every output"; the graph has exactly one.
        out = sess.run(None, {state["input_name"]: tensor})[0]
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    finally:
        _stop_running()

    # out is (batch, 84, 8400) - decode each image with ITS OWN letterbox params.
    images = [_decode(out[i], *metas[i]) for i in range(len(metas))]
    return {"images": images, "inference_ms": elapsed_ms}


@asynccontextmanager
async def lifespan(app: FastAPI):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = ORT_INTRA_THREADS
    # Requests are parallelised by our own thread pool, so ORT does not need to
    # run separate branches of one graph concurrently.
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    sess = ort.InferenceSession(MODEL_PATH, opts, providers=["CPUExecutionProvider"])
    state["session"] = sess
    state["input_name"] = sess.get_inputs()[0].name
    state["providers"] = sess.get_providers()

    # Ultralytics stores the class names in the ONNX metadata.
    meta = sess.get_modelmeta().custom_metadata_map or {}
    try:
        state["names"] = {int(k): v for k, v in ast.literal_eval(meta.get("names", "{}")).items()}
    except Exception:
        state["names"] = {}

    # Warm up: first run() builds the execution plan and allocates arenas.
    sess.run(None, {state["input_name"]: np.zeros((1, 3, IMGSZ, IMGSZ), dtype=np.float32)})

    state["executor"] = ThreadPoolExecutor(
        max_workers=INFERENCE_WORKERS, thread_name_prefix="infer"
    )
    state["rss_baseline_mb"] = round(psutil.Process().memory_info().rss / 1024 / 1024, 1)
    state["startup_ms"] = round((time.time() - PROCESS_START) * 1000, 1)
    state["ready"] = True
    yield
    state["ready"] = False
    state["executor"].shutdown(wait=True)
    state["executor"] = None
    state["session"] = None


app = FastAPI(
    title="YOLO ONNX Runtime Detection API",
    description="yolov8n as a static ONNX graph. One shared thread-safe session, no model pool.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def info():
    return {
        "service": "yolo-onnx-detection",
        "runtime": "onnxruntime",
        "providers": state["providers"],
        "model": os.path.basename(MODEL_PATH),
        "conf": CONF,
        "iou": IOU,
        "imgsz": IMGSZ,
        "replica": REPLICA,
        "inference_workers": INFERENCE_WORKERS,
        "ort_intra_threads": ORT_INTRA_THREADS,
        "max_batch": MAX_BATCH,
        "model_pool": "none - InferenceSession is thread-safe",
        "endpoints": {
            "predict": "POST /predict (one or many 'file' parts)",
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
        c = dict(_counters)
    return {
        "replica": REPLICA,
        "runtime": "onnxruntime",
        "inference_workers": INFERENCE_WORKERS,
        "ort_intra_threads": ORT_INTRA_THREADS,
        "startup_ms": state["startup_ms"],
        "uptime_s": round(time.time() - PROCESS_START, 1),
        "in_flight": c["in_flight"],
        "peak_in_flight": c["peak_in_flight"],
        # actually inside session.run() -> proof of real parallelism
        "running": c["running"],
        "peak_running": c["peak_running"],
        "total_requests": c["total"],
        "images": c["images"],
        "failures": c["failures"],
        "rss_mb": round(psutil.Process().memory_info().rss / 1024 / 1024, 1),
        "rss_baseline_mb": state["rss_baseline_mb"],
    }


@app.post("/predict")
async def predict(file: List[UploadFile] = File(...)):
    """Detect on one or many images. Repeat the `file` part to send a batch."""
    if not state["ready"]:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    if not file:
        raise HTTPException(status_code=400, detail="Upload at least one image")
    if len(file) > MAX_BATCH:
        raise HTTPException(
            status_code=413, detail=f"Batch too large: max {MAX_BATCH} images"
        )

    tensors, metas = [], []
    for f in file:
        if not (f.content_type or "").startswith("image/"):
            raise HTTPException(status_code=400, detail="Upload image files only")
        raw = await f.read()
        try:
            im = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            raise HTTPException(status_code=400, detail=f"Could not decode {f.filename}")
        w0, h0 = im.size
        canvas, r, dw, dh = _letterbox(im, IMGSZ)
        # Model wants float32 in 0..1, channels-first. PIL gives uint8 HWC.
        arr = np.asarray(canvas, dtype=np.float32) / 255.0     # HWC
        tensors.append(arr.transpose(2, 0, 1))                  # CHW
        # Keep each image's letterbox params - _decode needs them to map the
        # boxes back, and they differ per image when sizes differ.
        metas.append((r, dw, dh, w0, h0))

    # (N, 3, 640, 640). ascontiguousarray because ORT wants a contiguous buffer
    # and would otherwise copy it itself.
    batch = np.ascontiguousarray(np.stack(tensors))             # NCHW

    _acquire(len(metas))
    t0 = time.perf_counter()
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            state["executor"], _infer_batch, (batch, metas)
        )
    except Exception:
        _fail()
        raise
    finally:
        _release()
    total_ms = round((time.perf_counter() - t0) * 1000, 1)

    n = len(metas)
    per_image = [{"count": len(d), "detections": d} for d in result["images"]]
    body = {
        "batch_size": n,
        "images": per_image,
        "inference_ms": result["inference_ms"],
        "inference_ms_per_image": round(result["inference_ms"] / n, 1),
        "total_ms": total_ms,
        "total_ms_per_image": round(total_ms / n, 1),
        "queue_wait_ms": round(total_ms - result["inference_ms"], 1),
        "replica": REPLICA,
        "startup_ms": state["startup_ms"],
    }
    if n == 1:
        # Backwards compatible with the single-image callers.
        body["count"] = per_image[0]["count"]
        body["detections"] = per_image[0]["detections"]
    return body
