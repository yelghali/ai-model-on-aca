"""Durable Functions document pipeline — fan-out / fan-in.

Flow
----
    App uploads PDFs to  uploads/
        -> blob trigger starts ONE orchestration per file (deterministic id)
        -> Document orchestrator:
             PHASE 1  fan out per PAGE   (Content Understanding + Azure OpenAI)
                      each page also reports the FIGURES it contains
             PHASE 2  fan out per CHUNK of figures  -> YOLO, batched
             PHASE 3  merge -> ONE result.json for the file
        -> notify the caller (webhook), results readable from results/

Why the phases are sequential: you cannot fan out YOLO until extraction has told
you which figures exist. That dependency is inherent to the data, not a design
choice. Within each phase everything runs in parallel.

Why per-file orchestrations rather than one big one: a 500-page file failing must
not roll back the other four, each file emits its own JSON, and each gets its own
history (which keeps replay cheap).

Status
------
Two ways to poll:
  GET /api/status/{instanceId}   -> progress, phase, per-page counts (custom status)
  the standard Durable statusQueryGetUri returned by the HTTP starter

Progress is published with set_custom_status so a UI can show "page 143 of 300"
while the run is still going.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import timedelta

import azure.durable_functions as df
import azure.functions as func

from shared import storage
from shared import docai
from shared.http_policy import (
    SCALE_SENSITIVE,
    THROTTLE_SENSITIVE,
    PermanentError,
    run_with_policy,
)
from shared.pdfsplit import extract_figures, page_count, render_page
from shared.simulators import content_understanding, openai_page, yolo_batch

app = df.DFApp(http_auth_level=func.AuthLevel.ANONYMOUS)

# Figures per YOLO call. 8 matches the measured sweet spot (ms/image is flat from
# 1..8) and keeps a call around a second - far below the ACA 240 s request limit.
YOLO_CHUNK = int(os.getenv("YOLO_CHUNK", "8"))
PAGE_DPI = int(os.getenv("PAGE_DPI", "150"))
NOTIFY_URL = os.getenv("NOTIFY_URL", "")

# Where page text and FIGURES come from:
#   content-understanding : Azure AI CU for text + PyMuPDF for figures (2 steps,
#                           and the figure half is heuristic - see pdfsplit)
#   mistral-docai         : Mistral Document AI returns markdown AND figure
#                           bounding boxes in ONE call, model-based rather than
#                           guessed from PDF internals
TEXT_SERVICE = os.getenv("TEXT_SERVICE", "content-understanding").lower()

# Durable-level retry. This is the OUTER net: it covers host restarts, activity
# timeouts and total outages. The INNER, cause-aware retry (429 vs 503 vs 400)
# lives in shared/http_policy.py, because RetryOptions cannot read Retry-After.
OUTER_RETRY = df.RetryOptions(first_retry_interval_in_milliseconds=2000, max_number_of_attempts=3)


# ---------------------------------------------------------------- triggers ---
@app.blob_trigger(arg_name="blob", path="uploads/{name}", connection="AzureWebJobsStorage")
@app.durable_client_input(client_name="client")
async def on_upload(blob: func.InputStream, client):
    """Start one orchestration per uploaded file.

    The instance id is derived from the blob path + size, so a duplicate trigger
    (blob triggers can fire more than once) resolves to the SAME instance and is
    a no-op instead of processing - and paying for - the document twice.
    """
    name = blob.name.split("/", 1)[-1]
    instance_id = "doc-" + hashlib.sha256(f"{blob.name}:{blob.length}".encode()).hexdigest()[:24]

    existing = await client.get_status(instance_id)
    if existing and existing.runtime_status and existing.runtime_status.name in (
        "Running", "Pending", "Completed"
    ):
        logging.info("duplicate trigger for %s -> instance %s already %s",
                     name, instance_id, existing.runtime_status.name)
        return

    await client.start_new("document_orchestrator", instance_id,
                           {"file": name, "run_id": instance_id})
    logging.info("started %s for %s", instance_id, name)


@app.route(route="start", methods=["POST"])
@app.durable_client_input(client_name="client")
async def http_start(req: func.HttpRequest, client) -> func.HttpResponse:
    """Manual kick-off: POST {"files": ["a.pdf", "b.pdf"]} (already in uploads/).

    Returns one instance id per file. Useful for demos and for re-running a
    batch without re-uploading.
    """
    body = req.get_json()
    files = body.get("files") or []
    if not files:
        return func.HttpResponse(json.dumps({"error": "no files"}), status_code=400,
                                 mimetype="application/json")

    started = []
    for name in files:
        instance_id = "doc-" + hashlib.sha256(f"uploads/{name}".encode()).hexdigest()[:24]
        await client.start_new("document_orchestrator", instance_id,
                               {"file": name, "run_id": instance_id})
        started.append({"file": name, "instanceId": instance_id,
                        "statusUrl": f"/api/status/{instance_id}"})
    return func.HttpResponse(json.dumps({"started": started}, indent=2),
                             mimetype="application/json")


@app.route(route="status/{instance_id}", methods=["GET"])
@app.durable_client_input(client_name="client")
async def http_status(req: func.HttpRequest, client) -> func.HttpResponse:
    """Progress for one run. Safe to poll while the workflow is still running."""
    instance_id = req.route_params.get("instance_id")
    status = await client.get_status(instance_id)
    if status is None:
        return func.HttpResponse(json.dumps({"error": "unknown instance"}),
                                 status_code=404, mimetype="application/json")

    payload = {
        "instanceId": instance_id,
        "runtimeStatus": status.runtime_status.name if status.runtime_status else None,
        "createdTime": str(status.created_time),
        "lastUpdatedTime": str(status.last_updated_time),
        # progress published by the orchestrator while it runs
        "progress": status.custom_status,
        "output": status.output,
    }
    return func.HttpResponse(json.dumps(payload, indent=2, default=str),
                             mimetype="application/json")


# ----------------------------------------------------------- orchestrator ---
@app.orchestration_trigger(context_name="context")
def document_orchestrator(context: df.DurableOrchestrationContext):
    args = context.get_input()
    file_name, run_id = args["file"], args["run_id"]

    # PHASE 0 - how many pages? (cheap, and needed to size the fan-out)
    meta = yield context.call_activity_with_retry("prepare_document", OUTER_RETRY,
                                                  {"file": file_name, "run_id": run_id})
    pages = meta["pages"]
    context.set_custom_status({"phase": "pages", "pages": pages, "pages_done": 0})

    # PHASE 1 - fan out per PAGE.
    # Content Understanding and OpenAI are SEPARATE activities on purpose: a 429
    # from one must not re-run the other (double cost, double latency).
    page_tasks = [
        context.call_activity_with_retry(
            "process_page", OUTER_RETRY,
            {"file": file_name, "run_id": run_id, "page": p},
        )
        for p in range(pages)
    ]
    page_results = yield context.task_all(page_tasks)

    # Collect figures across ALL pages, then chunk. This is what lets YOLO batch:
    # figure count is decoupled from page count.
    figures = [f for r in page_results for f in r.get("figures", [])]
    chunks = [figures[i:i + YOLO_CHUNK] for i in range(0, len(figures), YOLO_CHUNK)]
    context.set_custom_status({
        "phase": "figures", "pages": pages, "pages_done": len(page_results),
        "figures": len(figures), "yolo_batches": len(chunks),
    })

    # PHASE 2 - fan out per CHUNK of figures.
    yolo_tasks = [
        context.call_activity_with_retry(
            "detect_figures", OUTER_RETRY,
            {"run_id": run_id, "file": file_name, "figures": chunk},
        )
        for chunk in chunks
    ]
    yolo_results = yield context.task_all(yolo_tasks) if yolo_tasks else []

    # PHASE 3 - merge into ONE json for this file.
    result = yield context.call_activity_with_retry(
        "merge_results", OUTER_RETRY,
        {"file": file_name, "run_id": run_id,
         "pages": page_results, "detections": yolo_results},
    )

    context.set_custom_status({
        "phase": "done", "pages": pages, "pages_done": len(page_results),
        "figures": len(figures), "status": result["status"],
    })

    if NOTIFY_URL:
        yield context.call_activity_with_retry("notify", OUTER_RETRY, result)

    return result


# -------------------------------------------------------------- activities ---
@app.activity_trigger(input_name="payload")
def prepare_document(payload: dict) -> dict:
    storage.ensure_containers()
    pdf = storage.read_blob(storage.UPLOAD_CONTAINER, payload["file"])
    return {"file": payload["file"], "pages": page_count(pdf)}


@app.activity_trigger(input_name="payload")
def process_page(payload: dict) -> dict:
    """One page: render, Content Understanding, OpenAI, extract figures.

    Returns page data plus the figures found (as blob paths - never bytes).

    A PERMANENT failure is RETURNED as data, not raised. Raising would make
    Durable retry something that can never succeed; returning lets the document
    complete as `partial` with an honest per-page error.
    """
    file_name, run_id, page = payload["file"], payload["run_id"], payload["page"]
    base = f"{run_id}/{file_name}"

    pdf = storage.read_blob(storage.UPLOAD_CONTAINER, file_name)
    png = render_page(pdf, page, dpi=PAGE_DPI)
    page_path = f"{base}/pages/page-{page + 1:04d}.png"
    storage.write_blob(storage.WORK_CONTAINER, page_path, png, "image/png")

    out: dict = {"page": page + 1, "status": "ok", "page_blob": page_path,
                 "figures": [], "errors": []}

    # ---- text extraction (and, for Document AI, the figures too) -----------
    doc_figures: list = []
    if TEXT_SERVICE == "mistral-docai":
        # ONE call returns markdown AND figure regions, so no separate
        # extraction step and no proximity heuristic.
        try:
            res = run_with_policy(lambda: docai.analyze_page(png, page), THROTTLE_SENSITIVE)
            doc_figures = res.pop("figures", [])
            out["text"] = res
        except PermanentError as exc:
            out["status"] = "partial"
            out["errors"].append({"service": "mistral-document-ai", "error": str(exc)})
    else:
        try:
            out["text"] = run_with_policy(
                lambda: content_understanding(png, page), THROTTLE_SENSITIVE)
        except PermanentError as exc:
            out["status"] = "partial"
            out["errors"].append({"service": "content-understanding", "error": str(exc)})

    # ---- summarisation: a SEPARATE activity-level call on purpose ----------
    # A 429 from OpenAI must not re-run the text extraction above.
    try:
        out["openai"] = run_with_policy(
            lambda: openai_page(png, page), THROTTLE_SENSITIVE)
    except PermanentError as exc:
        out["status"] = "partial"
        out["errors"].append({"service": "azure-openai", "error": str(exc)})

    # ---- figures ----------------------------------------------------------
    # Document AI already returned them; otherwise fall back to PDF extraction.
    # (page, figure) travels with each one so YOLO results map back explicitly
    # after batching - figure counts vary per page, so position is not enough.
    figures = doc_figures or extract_figures(pdf, page)
    for fig in figures:
        fig_path = f"{base}/figures/p{page + 1:04d}-f{fig['figure']:02d}.png"
        storage.write_blob(storage.WORK_CONTAINER, fig_path, fig["png"], "image/png")
        out["figures"].append({
            "page": page + 1, "figure": fig["figure"], "blob": fig_path,
            "kind": fig.get("kind"), "width": fig["width"], "height": fig["height"],
            **({"bbox": fig["bbox"]} if fig.get("bbox") else {}),
        })

    return out


@app.activity_trigger(input_name="payload")
def detect_figures(payload: dict) -> list:
    """YOLO on a BATCH of figures. Fetches bytes from blob by reference."""
    figures = payload["figures"]
    loaded = [
        {"page": f["page"], "figure": f["figure"],
         "png": storage.read_blob(storage.WORK_CONTAINER, f["blob"])}
        for f in figures
    ]
    try:
        # SCALE_SENSITIVE: 503/507 here means ACA is adding a replica. Short,
        # jittered backoff - the capacity arrives in seconds.
        return run_with_policy(lambda: yolo_batch(loaded), SCALE_SENSITIVE)
    except PermanentError as exc:
        return [{"page": f["page"], "figure": f["figure"], "count": 0,
                 "detections": [], "error": str(exc)} for f in figures]


@app.activity_trigger(input_name="payload")
def merge_results(payload: dict) -> dict:
    """Fan-in: ONE json per file, one entry per page."""
    file_name, run_id = payload["file"], payload["run_id"]

    # Explicit correlation by (page, figure) - never positional.
    det_by_key: dict[tuple, dict] = {}
    for chunk in payload["detections"]:
        for d in chunk:
            det_by_key[(d["page"], d["figure"])] = d

    pages_out, failed = [], 0
    for p in sorted(payload["pages"], key=lambda r: r["page"]):
        figs = []
        for f in p.get("figures", []):
            d = det_by_key.get((f["page"], f["figure"]), {})
            figs.append({
                "figure": f["figure"], "blob": f["blob"], "kind": f.get("kind"),
                "width": f["width"], "height": f["height"],
                "count": d.get("count", 0),
                "detections": d.get("detections", []),
                **({"error": d["error"]} if d.get("error") else {}),
            })
        if p["status"] != "ok":
            failed += 1
        pages_out.append({
            "page": p["page"],
            "status": p["status"],
            "text": p.get("text"),
            "openai": p.get("openai"),
            "figures": figs,
            **({"errors": p["errors"]} if p.get("errors") else {}),
        })

    doc = {
        "file": file_name,
        "run_id": run_id,
        "pages": len(pages_out),
        "figures": sum(len(p["figures"]) for p in pages_out),
        "status": "ok" if failed == 0 else "partial",
        "pages_with_errors": failed,
        "results": pages_out,
    }

    out_path = f"{file_name}.json"
    storage.write_blob(storage.RESULT_CONTAINER, out_path,
                       json.dumps(doc, indent=2).encode(), "application/json")

    # Return a SUMMARY, not the full document: orchestration output is persisted
    # to Storage and a 300-page result set with text would hit the size limits.
    return {"file": file_name, "run_id": run_id, "status": doc["status"],
            "pages": doc["pages"], "figures": doc["figures"],
            "pages_with_errors": failed,
            "result_blob": f"{storage.RESULT_CONTAINER}/{out_path}"}


@app.activity_trigger(input_name="payload")
def notify(payload: dict) -> dict:
    """Tell the calling app the run is done. Best-effort, never fails the run."""
    import requests

    try:
        resp = requests.post(NOTIFY_URL, json=payload, timeout=15)
        return {"notified": True, "status_code": resp.status_code}
    except Exception as exc:
        logging.warning("notify failed: %s", exc)
        return {"notified": False, "error": str(exc)}
