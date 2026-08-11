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
import time
import uuid
from datetime import timedelta

import azure.durable_functions as df
import azure.functions as func

from shared import docai, notify as notifier, storage
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
# Bound the upload: an unbounded multipart body is a trivial memory DoS, and the
# function holds the only path to storage.
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(40 * 1024 * 1024)))

# Where page text and FIGURES come from:
#   content-understanding : Azure AI CU for text + PyMuPDF for figures (2 steps,
#                           and the figure half is heuristic - see pdfsplit)
#   mistral-docai         : Mistral Document AI returns markdown AND figure
#                           bounding boxes in ONE call, model-based rather than
#                           guessed from PDF internals
TEXT_SERVICE = os.getenv("TEXT_SERVICE", "content-understanding").lower()


def _instance_id_for(blob_name: str, size: int, metadata: dict | None) -> str:
    """Resolve the orchestration id for an uploaded blob.

    THE CALLER OWNS THE IDENTIFIER. If the app stamped a `correlationid` on the
    blob at upload time, that is used verbatim - so the app knows the id BEFORE
    the upload completes and never has to reverse-engineer a hashing scheme.

    Falling back to a hash of path+size keeps the "just drop a file in the
    container" case working, and keeps it idempotent: the same bytes at the same
    path always resolve to the same instance, so a duplicate blob trigger is a
    no-op rather than a second (billed) run.
    """
    corr = (metadata or {}).get("correlationid") or (metadata or {}).get("correlationId")
    if corr:
        # Durable instance ids must not contain path or query separators.
        safe = "".join(c for c in corr if c.isalnum() or c in "-_")[:64]
        if safe:
            return safe
    return "doc-" + hashlib.sha256(f"{blob_name}:{size}".encode()).hexdigest()[:24]

# Durable-level retry. This is the OUTER net: it covers host restarts, activity
# timeouts and total outages. The INNER, cause-aware retry (429 vs 503 vs 400)
# lives in shared/http_policy.py, because RetryOptions cannot read Retry-After.
OUTER_RETRY = df.RetryOptions(first_retry_interval_in_milliseconds=2000, max_number_of_attempts=3)

# Callbacks get more attempts and a longer first interval: the receiving app may
# simply be restarting, and losing the completion notice is worse than a slow one.
NOTIFY_RETRY = df.RetryOptions(first_retry_interval_in_milliseconds=5000, max_number_of_attempts=5)


# ---------------------------------------------------------------- triggers ---
@app.blob_trigger(arg_name="blob", path="uploads/{name}", connection="AzureWebJobsStorage")
@app.durable_client_input(client_name="client")
async def on_upload(blob: func.InputStream, client):
    """Start one orchestration per uploaded file.

    The id comes from the blob's `correlationid` metadata when the app supplied
    one, otherwise from a hash of path+size. Either way the SAME upload always
    maps to the SAME instance, so a duplicate trigger (blob triggers can fire
    more than once) is a no-op instead of processing - and billing - the
    document twice.
    """
    name = blob.name.split("/", 1)[-1]
    metadata = dict(getattr(blob, "metadata", None) or {})
    instance_id = _instance_id_for(blob.name, blob.length or 0, metadata)

    existing = await client.get_status(instance_id)
    if existing and existing.runtime_status and existing.runtime_status.name in (
        "Running", "Pending", "Completed"
    ):
        logging.info("duplicate trigger for %s -> %s already %s",
                     name, instance_id, existing.runtime_status.name)
        return

    await client.start_new("document_orchestrator", instance_id, {
        "file": name, "run_id": instance_id,
        "batch_id": metadata.get("batchid") or metadata.get("batchId"),
    })
    logging.info("started %s for %s", instance_id, name)


@app.route(route="upload", methods=["POST"])
@app.durable_client_input(client_name="client")
async def http_upload(req: func.HttpRequest, client) -> func.HttpResponse:
    """Public upload gateway: multipart PDFs in, orchestrations started.

    This exists because storage has `publicNetworkAccess: Disabled` - the
    browser has NO network path to blob, and SAS upload is impossible with
    `allowSharedKeyAccess: false`. The function is VNet-integrated and holds the
    only route, so it accepts the bytes and writes them with managed identity.

    That is also the more secure shape: the client never receives a storage
    credential of any kind.
    """
    files = req.files.getlist("file") if hasattr(req, "files") else []
    if not files:
        return _json({"error": "no files (multipart field 'file')"}, 400)

    form_batch = None
    try:
        form_batch = req.form.get("batchId")
    except Exception:
        pass
    batch_id = form_batch or ("batch-" + uuid.uuid4().hex[:16])

    storage.ensure_containers()
    items = []
    for f in files:
        data = f.read()
        if not data[:5] == b"%PDF-":
            return _json({"error": f"{f.filename}: not a PDF"}, 400)
        if len(data) > MAX_UPLOAD_BYTES:
            return _json({"error": f"{f.filename}: exceeds {MAX_UPLOAD_BYTES} bytes"}, 413)

        # The CALLER owns the id. Generated here and returned immediately, so
        # the app can poll without reverse-engineering any hashing scheme.
        corr = "doc-" + uuid.uuid4().hex[:20]
        safe_name = os.path.basename(f.filename or "upload.pdf").replace("\\", "_")
        blob_name = f"{corr}-{safe_name}"

        storage.write_blob(
            storage.UPLOAD_CONTAINER, blob_name, data, "application/pdf",
            metadata={"correlationid": corr, "batchid": batch_id},
        )
        items.append({"file": blob_name, "display": safe_name, "run_id": corr})

    await client.start_new("batch_orchestrator", batch_id,
                           {"batch_id": batch_id, "items": items})

    return _json({
        "batchId": batch_id,
        "statusUrl": f"api/batch/{batch_id}",
        "files": [{"file": i["display"], "blob": i["file"],
                   "instanceId": i["run_id"],
                   "statusUrl": f"api/status/{i['run_id']}"} for i in items],
    })


@app.route(route="ui", methods=["GET"])
def http_ui(req: func.HttpRequest) -> func.HttpResponse:
    """Serve the demo page. Static file, no templating, no secrets."""
    path = os.path.join(os.path.dirname(__file__), "ui", "index.html")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return func.HttpResponse(fh.read(), mimetype="text/html")
    except FileNotFoundError:
        return func.HttpResponse("UI not deployed", status_code=404)


def _json(payload: dict, status: int = 200) -> func.HttpResponse:
    return func.HttpResponse(json.dumps(payload, indent=2, default=str),
                             status_code=status, mimetype="application/json")


@app.route(route="start", methods=["POST"])
@app.durable_client_input(client_name="client")
async def http_start(req: func.HttpRequest, client) -> func.HttpResponse:
    """Start a BATCH the app already uploaded.

        POST /api/start
        { "batchId": "optional-caller-id",
          "files": ["a.pdf", {"file": "b.pdf", "correlationId": "my-id"}] }

    Returns a batchId plus one instance id per file, so the app never has to
    guess an identifier. One callback is sent when the WHOLE batch finishes -
    which is what "user uploaded 5 documents" actually needs, rather than five
    unrelated webhooks.
    """
    try:
        body = req.get_json()
    except ValueError:
        return func.HttpResponse(json.dumps({"error": "invalid json"}),
                                 status_code=400, mimetype="application/json")

    files = body.get("files") or []
    if not files:
        return func.HttpResponse(json.dumps({"error": "no files"}),
                                 status_code=400, mimetype="application/json")

    batch_id = body.get("batchId") or ("batch-" + uuid.uuid4().hex[:16])
    items = []
    for entry in files:
        if isinstance(entry, str):
            name, corr = entry, None
        else:
            name, corr = entry.get("file"), entry.get("correlationId")
        if not name:
            continue
        inst = _instance_id_for(f"uploads/{name}", 0, {"correlationid": corr} if corr else None)
        items.append({"file": name, "run_id": inst})

    await client.start_new("batch_orchestrator", batch_id,
                           {"batch_id": batch_id, "items": items})

    return func.HttpResponse(json.dumps({
        "batchId": batch_id,
        "statusUrl": f"/api/batch/{batch_id}",
        "files": [{"file": i["file"], "instanceId": i["run_id"],
                   "statusUrl": f"/api/status/{i['run_id']}"} for i in items],
    }, indent=2), mimetype="application/json")


@app.route(route="status/{instance_id}", methods=["GET"])
@app.durable_client_input(client_name="client")
async def http_status(req: func.HttpRequest, client) -> func.HttpResponse:
    """Progress for one run. Safe to poll while the workflow is still running."""
    return await _status_response(client, req.route_params.get("instance_id"))


@app.route(route="status/by-file/{name}", methods=["GET"])
@app.durable_client_input(client_name="client")
async def http_status_by_file(req: func.HttpRequest, client) -> func.HttpResponse:
    """Escape hatch: resolve a run from the FILE NAME.

    For apps that uploaded without stamping a correlation id and cannot
    reproduce the deterministic hash (it includes the blob size).
    """
    name = req.route_params.get("name")
    try:
        props = storage.blob_properties(storage.UPLOAD_CONTAINER, name)
    except Exception:
        return func.HttpResponse(json.dumps({"error": f"no such upload: {name}"}),
                                 status_code=404, mimetype="application/json")
    instance_id = _instance_id_for(f"uploads/{name}", props["size"], props["metadata"])
    return await _status_response(client, instance_id)


@app.route(route="batch/{batch_id}", methods=["GET"])
@app.durable_client_input(client_name="client")
async def http_batch_status(req: func.HttpRequest, client) -> func.HttpResponse:
    """Progress for a whole upload set: per-file status plus an aggregate."""
    return await _status_response(client, req.route_params.get("batch_id"))


async def _status_response(client, instance_id: str) -> func.HttpResponse:
    status = await client.get_status(instance_id)
    if status is None:
        return func.HttpResponse(json.dumps({"error": "unknown instance", "instanceId": instance_id}),
                                 status_code=404, mimetype="application/json")
    payload = {
        "instanceId": instance_id,
        "runtimeStatus": status.runtime_status.name if status.runtime_status else None,
        "createdTime": str(status.created_time),
        "lastUpdatedTime": str(status.last_updated_time),
        "progress": status.custom_status,     # published DURING the run
        "output": status.output,
    }
    return func.HttpResponse(json.dumps(payload, indent=2, default=str),
                             mimetype="application/json")


# ----------------------------------------------------------- orchestrators ---
@app.orchestration_trigger(context_name="context")
def batch_orchestrator(context: df.DurableOrchestrationContext):
    """Fan out over FILES, then send ONE callback for the whole set.

    Sub-orchestrations rather than activities: each file gets its own history
    and its own failure boundary, so a 500-page document failing does not roll
    back the other four.
    """
    args = context.get_input()
    batch_id, items = args["batch_id"], args["items"]

    context.set_custom_status({"phase": "running", "files": len(items), "files_done": 0})

    tasks = [
        context.call_sub_orchestrator(
            "document_orchestrator",
            {"file": i["file"], "run_id": i["run_id"], "batch_id": batch_id},
            i["run_id"],
        )
        for i in items
    ]
    results = yield context.task_all(tasks)

    summary = {
        "batch_id": batch_id,
        "files": len(results),
        "ok": sum(1 for r in results if r.get("status") == "ok"),
        "partial": sum(1 for r in results if r.get("status") == "partial"),
        "pages": sum(r.get("pages", 0) for r in results),
        "figures": sum(r.get("figures", 0) for r in results),
        "results": results,
        "completed_utc": context.current_utc_datetime.isoformat(),
    }
    context.set_custom_status({"phase": "done", "files": len(results),
                               "files_done": len(results),
                               "ok": summary["ok"], "partial": summary["partial"]})

    # ONE signed callback for the whole upload set.
    if notifier.is_configured():
        yield context.call_activity_with_retry("notify", NOTIFY_RETRY,
                                               {"event": "batch.completed", **summary})
    return summary


# ----------------------------------------------------------- orchestrator ---
@app.orchestration_trigger(context_name="context")
def document_orchestrator(context: df.DurableOrchestrationContext):
    args = context.get_input()
    file_name, run_id = args["file"], args["run_id"]
    batch_id = args.get("batch_id")

    # current_utc_datetime, NOT datetime.now(): orchestrator code is REPLAYED,
    # so every wall-clock read must come from the deterministic context or the
    # timings would change on each replay.
    started = context.current_utc_datetime

    # PHASE 0 - how many pages? (cheap, and needed to size the fan-out)
    meta = yield context.call_activity_with_retry("prepare_document", OUTER_RETRY,
                                                  {"file": file_name, "run_id": run_id})
    pages = meta["pages"]
    context.set_custom_status({"phase": "pages", "pages": pages, "pages_done": 0,
                               "batch_id": batch_id})

    # PHASE 1 - fan out per PAGE.
    # Text extraction and OpenAI are SEPARATE calls on purpose: a 429 from one
    # must not re-run the other (double cost, double latency).
    page_tasks = [
        context.call_activity_with_retry(
            "process_page", OUTER_RETRY,
            {"file": file_name, "run_id": run_id, "page": p},
        )
        for p in range(pages)
    ]
    page_results = yield context.task_all(page_tasks)
    t_pages = context.current_utc_datetime

    # Collect figures across ALL pages, then chunk. This is what lets YOLO batch:
    # figure count is decoupled from page count.
    figures = [f for r in page_results for f in r.get("figures", [])]
    chunks = [figures[i:i + YOLO_CHUNK] for i in range(0, len(figures), YOLO_CHUNK)]
    context.set_custom_status({
        "phase": "figures", "pages": pages, "pages_done": len(page_results),
        "figures": len(figures), "yolo_batches": len(chunks), "batch_id": batch_id,
    })

    # PHASE 2 - fan out per CHUNK of figures.
    yolo_tasks = [
        context.call_activity_with_retry(
            "detect_figures", OUTER_RETRY,
            {"run_id": run_id, "file": file_name, "figures": chunk},
        )
        for chunk in chunks
    ]
    # NOTE: this must NOT be written as
    #     yield context.task_all(tasks) if tasks else []
    # Python binds that as `yield (task_all(...) if tasks else [])`, so a
    # document with NO figures would yield a plain list to the Durable runtime,
    # which expects a Task. Documents with zero figures are common (a paper
    # whose diagrams are all vector, a text-only report), so this path is hit
    # in practice.
    if yolo_tasks:
        yolo_results = yield context.task_all(yolo_tasks)
    else:
        yolo_results = []
    t_yolo = context.current_utc_datetime

    # PHASE 3 - merge into ONE json for this file.
    result = yield context.call_activity_with_retry(
        "merge_results", OUTER_RETRY,
        {"file": file_name, "run_id": run_id, "batch_id": batch_id,
         "pages": page_results, "detections": yolo_results},
    )

    finished = context.current_utc_datetime
    total_s = max((finished - started).total_seconds(), 0.001)
    result["batch_id"] = batch_id
    result["timings"] = {
        "pages_s": round((t_pages - started).total_seconds(), 2),
        "yolo_s": round((t_yolo - t_pages).total_seconds(), 2),
        "total_s": round(total_s, 2),
        "pages_per_s": round(pages / total_s, 2),
    }

    context.set_custom_status({
        "phase": "done", "pages": pages, "pages_done": len(page_results),
        "figures": len(figures), "status": result["status"],
        "batch_id": batch_id, "timings": result["timings"],
    })

    # Notify per file ONLY when this document is standalone. Inside a batch the
    # parent sends a single callback for the whole upload set - five files must
    # not produce five unrelated webhooks.
    if notifier.is_configured() and not batch_id:
        yield context.call_activity_with_retry(
            "notify", NOTIFY_RETRY, {"event": "document.completed", **result})

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

    def _retry_logger(service: str):
        """Make retries VISIBLE.

        Without this the policy still works, but silently - there is no way to
        tell a run that sailed through from one that absorbed twenty 429s, and
        no way to prove the backoff is doing anything. The prefix is fixed so
        it can be queried:

            traces | where message startswith "RETRY "
        """
        def _log(attempt: int, delay: float, exc: Exception) -> None:
            logging.warning("RETRY %s attempt=%d delay=%.2fs page=%d cause=%s: %s",
                            service, attempt, delay, page + 1,
                            type(exc).__name__, exc)
        return _log

    # ---- text extraction (and, for Document AI, the figures too) -----------
    doc_figures: list = []
    if TEXT_SERVICE == "mistral-docai":
        # ONE call returns markdown AND figure regions, so no separate
        # extraction step and no proximity heuristic.
        try:
            res = run_with_policy(lambda: docai.analyze_page(png, page), THROTTLE_SENSITIVE,
                                  on_retry=_retry_logger("mistral-document-ai"))
            doc_figures = res.pop("figures", [])
            out["text"] = res
        except PermanentError as exc:
            out["status"] = "partial"
            out["errors"].append({"service": "mistral-document-ai", "error": str(exc)})
    else:
        try:
            out["text"] = run_with_policy(
                lambda: content_understanding(png, page), THROTTLE_SENSITIVE,
                on_retry=_retry_logger("content-understanding"))
        except PermanentError as exc:
            out["status"] = "partial"
            out["errors"].append({"service": "content-understanding", "error": str(exc)})

    # ---- summarisation: a SEPARATE activity-level call on purpose ----------
    # A 429 from OpenAI must not re-run the text extraction above.
    try:
        out["openai"] = run_with_policy(
            lambda: openai_page(png, page), THROTTLE_SENSITIVE,
            on_retry=_retry_logger("azure-openai"))
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
        def _log(attempt: int, delay: float, exc: Exception) -> None:
            logging.warning("RETRY yolo attempt=%d delay=%.2fs figures=%d cause=%s: %s",
                            attempt, delay, len(figures), type(exc).__name__, exc)

        return run_with_policy(lambda: yolo_batch(loaded), SCALE_SENSITIVE,
                               on_retry=_log)
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
        "batch_id": payload.get("batch_id"),
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
    """Deliver the completion callback.

    Deliberately does NOT swallow transport failures: shared/notify.py raises on
    5xx and connection errors so Durable's retry actually fires. The earlier
    version caught everything and returned, which silently disabled the retry.
    A 4xx raises PermanentError, which we convert to a result - the app rejected
    the payload and will keep rejecting it.
    """
    try:
        return notifier.post(payload)
    except PermanentError as exc:
        logging.error("notify rejected permanently: %s", exc)
        return {"notified": False, "permanent": True, "error": str(exc)}
