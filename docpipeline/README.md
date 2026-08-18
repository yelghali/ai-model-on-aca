# Document pipeline — Durable Functions fan-out / fan-in

Upload → per-page AI extraction → batched YOLO on figures → one JSON per file,
with status polling and a completion webhook.

```
POST /api/upload  (writes the blob AND starts the run, returns ids)
                    │
                    ▼   one orchestration PER FILE
        ┌──────────────────────────────────────────────────┐
        │ PHASE 1 — fan out per PAGE                       │
        │   text + figure regions:                         │
        │     default: CU + PyMuPDF                        │
        │     optional: Mistral Document AI (one call)     │
        │   Azure OpenAI              (1 call / page)      │
        ├──────────────────────────────────────────────────┤
        │ collect figures across ALL pages → chunk by 8    │
        ├──────────────────────────────────────────────────┤
        │ PHASE 2 — fan out per CHUNK → YOLO (batched)     │
        ├──────────────────────────────────────────────────┤
        │ PHASE 3 — merge → results/<file>.json            │
        └──────────────────────────────────────────────────┘
                    │
                    ▼  webhook (NOTIFY_URL) + GET /api/status/{id}
```

**There is no blob trigger, on purpose.** Storage has public network access and
shared keys disabled, so nothing outside the VNet can write to `uploads/` — not
even with a SAS, because there is no key to sign one with. The only writer is
`/api/upload`, which starts the orchestration itself. A blob trigger would be a
slower second path to start work that already started, racing it on the same
instance id. It also does not work on Flex Consumption without Event Grid; see
[Known limitations](#known-limitations).

**Diagrams** (open with [diagrams.net](https://app.diagrams.net)):
[docs/architecture.drawio](docs/architecture.drawio) — Azure hosting, private
networking and the notification path back to the uploading app.
[docs/pipeline.drawio](docs/pipeline.drawio) — the orchestration phases.

## Functions shown in the Azure portal

Azure discovers 13 functions from [`function_app.py`](function_app.py), split
across two Flex Consumption Function Apps by the `FUNCTION_ROLE` setting. The
public API app indexes six HTTP/client functions. The worker app indexes two
orchestrators and five activities. `Enabled` means that host loaded the function;
it does not describe the outcome of an individual document run.

Both apps deploy the same package, but role-gated blueprints keep runtime
ownership separate:

| Function App role | Indexed functions | Scaling signal |
|---|---|---|
| `api` | UI, upload, start, status, batch status | HTTP load |
| `worker` | orchestrators and activities | Durable task-hub backlog |

The API's Durable client explicitly targets the worker-owned `docpipeline` task
hub through an identity-based `WorkerStorage` connection. Status does not require
co-location because orchestration state lives in the task hub. Only the worker
app consumes that hub; running two different worker apps against one hub can
compete for messages and leave orchestrations stuck.

The Durable extension itself also uses `WorkerStorage`; do not let it fall back
to Flex's generated `AzureWebJobsStorage` connection string. Shared keys are
disabled, and that generated value contains an empty `AccountKey`. The worker
keeps one `durable` instance always ready, as recommended for Durable Functions
on Flex, and scales out from task-hub backlog.

**HTTP entry points** are called by the browser or another application:

| Portal name | Route | Responsibility |
|---|---|---|
| `http_ui` | `GET /api/ui` | Serves the upload demo page. |
| `http_upload` | `POST /api/upload` | Accepts PDFs, writes them to the private `uploads` container, and starts a batch. |
| `http_start` | `POST /api/start` | Starts a batch for files that are already in `uploads`; the browser normally uses `http_upload`. |
| `http_status` | `GET /api/status/{instanceId}` | Returns Durable runtime state, live phase/progress, and the final result summary for one file. |
| `http_status_by_file` | `GET /api/status/by-file/{name}` | Resolves an orchestration from an uploaded blob name. |
| `http_batch_status` | `GET /api/batch/{batchId}` | Returns aggregate status for all files in one upload batch. |

**Orchestrations** coordinate work. Durable Functions persists and replays them,
so they schedule activities but do not perform network or file I/O themselves:

| Portal name | Responsibility |
|---|---|
| `batch_orchestrator` | Parent workflow: starts one document sub-orchestration per file, waits for all files, aggregates their summaries, and sends one batch callback. |
| `document_orchestrator` | Per-file workflow: count pages, fan out page processing, batch figures for YOLO, merge the result, and publish progress. |

**Activities** perform the actual I/O and computation scheduled by the
orchestrators:

| Portal name | Responsibility |
|---|---|
| `prepare_document` | Reads the uploaded PDF and counts its pages. |
| `process_page` | Renders one page, extracts text, runs summarisation, extracts figures, and writes page/figure images to `work`. |
| `detect_figures` | Sends one chunk of figures to the private YOLO Container App and correlates detections by page and figure number. |
| `merge_results` | Combines page and YOLO outputs and writes one JSON document to `results/<uploaded-file>.json`. |
| `notify` | Sends the optional signed completion webhook after the workflow finishes. |

The normal call chain crosses the Function App boundary through the task hub:

```text
API app:    http_ui -> http_upload -> worker task hub
            http_status / http_batch_status -> worker task hub

Worker app: batch_orchestrator -> document_orchestrator
                                   -> prepare_document
                                   -> process_page (one per page)
                                   -> detect_figures (one per chunk) -> ACA YOLO API
                                   -> merge_results
            -> notify
```

The three status fields answer different questions:

| Field | Example | Meaning |
|---|---|---|
| Durable runtime status | `Completed` | The orchestration executed successfully. |
| Progress phase | `done` | Page processing, figure detection, merge, and result storage all finished. |
| Document result status | `partial` | The workflow finished, but one or more pages recorded a permanent service error; details are preserved in the result JSON. |

Therefore `Completed` + `done` + `partial` is valid: execution completed, but
the document result is incomplete. Retryable 429/503 failures do not cause
`partial` when they recover; permanent 400-class page failures do.

## Why the shape is what it is

**Durable Functions is the control plane; the YOLO API is the compute plane.**
The document is already a Durable job. `/predict` does not replace job control;
it is a short, stateless invocation that keeps the model resident and reusable
across batches. This is the preferred shape while a batched inference remains
comfortably below the configured 120-second client timeout and ACA's 240-second
default ingress limit.

If one inference becomes long or unpredictable, keep the Durable orchestration
but replace the held HTTP call with external queue-backed work: write input
references, enqueue an idempotent job, let an ACA worker write the result, then
resume the orchestration from a completion event or durable polling activity.
A queue-scaled Container App avoids reloading the model for every message. ACA
Jobs are better for coarse one-off or scheduled batches, not one Job per small
figure chunk.

See [the concurrency guide](../docs/CONCURRENCY.md#how-aca-http-scaling-behaves-over-time)
for the target-2 scale calculation, 15-second metric window, five-minute
scale-down behavior, request timeouts, and the API-versus-job recommendation.

**Pages and figures are different units.** A 300-page document may hold 60 diagrams;
some pages have none. Measured on the sample papers:

| Document | Pages | Figures |
|---|---:|---:|
| bert.pdf | 16 | 39 |
| yolo-unified-detection.pdf | 10 | 13 |
| resnet-deep-residual-learning.pdf | 12 | 7 |
| attention-is-all-you-need.pdf | 15 | 6 |

That decoupling is what lets YOLO batch: figures are pooled across the whole document
and chunked, instead of being stuck at one image per page. The text APIs cannot batch —
they are one call per page — so the two fan-outs have different granularity.

**The phases are sequential because of a data dependency**, not by choice: you cannot
fan out YOLO until extraction has reported which figures exist. Inside each phase,
everything runs in parallel.

**Vector figures matter.** `page.get_images()` only sees *embedded raster* images.
Diagrams from TikZ, matplotlib PDF export, Illustrator or Visio are vector paths and are
invisible to it. With rasters only, `resnet.pdf` yielded **0 figures** — a paper full of
diagrams. Clustering vector paths and rendering the regions took the corpus from
**45 → 65 figures**.

## Two ways to get text and figures

`TEXT_SERVICE` selects the source:

| | `content-understanding` (default) | `mistral-docai` |
|---|---|---|
| Page text | Azure AI Content Understanding | Mistral Document AI (`mistral-ocr-latest`) |
| Figures | PyMuPDF: rasters + **clustered vector paths** | returned by the **same call**, with bounding boxes |
| Calls per page | 2 (text, then local extraction) | **1** |
| Figure detection | heuristic — proximity clustering of drawing paths | model-based layout detection |

Document AI is the better figure source: it returns the regions it *recognised*, rather
than regions inferred from PDF internals, and it hands back markdown and cropped images in
one response ([`shared/docai.py`](shared/docai.py)). That removes both weaknesses of the
PyMuPDF path — invisible vector diagrams, and proximity clustering that can merge two
adjacent figures or split a sparse one.

```powershell
$env:TEXT_SERVICE='mistral-docai'
$env:MISTRAL_API_KEY='...'          # unset -> simulated, same response shape
python scripts/run_local.py "samples/*.pdf"
```

Without a key the module simulates the service and figures fall back to PyMuPDF, so the
offline demo still produces realistic counts. `MISTRAL_ENDPOINT` overrides the URL for an
Azure AI Foundry deployment of the same model.

## Reliability

`RetryOptions` alone is not enough: it retries everything with one fixed backoff and
cannot read `Retry-After`. So failures are classified by cause first
([`shared/http_policy.py`](shared/http_policy.py)):

| Status | Classified as | Behaviour |
|---|---|---|
| 429 | `ThrottledError` | honour `Retry-After` (capped), + jitter |
| 408 / 502 / 503 / 504 / **507** | `TransientError` | short jittered backoff — ACA is adding a replica |
| 400 / 401 / 413 | `PermanentError` | **never retried**; page marked failed |
| 5xx other | `TransientError` | exponential backoff |

507 is what the ACA edge returns when it cannot place a request yet. Measured on this
repo: **21% of batches got 503/507 during burst** while the app scaled out — so this is a
real path, not a theoretical one.

Other guarantees:

- **Full jitter** on every backoff. Durable fans out hundreds of activities at once;
  without jitter they fail together, sleep the same interval and retry in lockstep.
- **`maxConcurrentActivityFunctions: 8`** in [`host.json`](host.json) caps the fan-out so
  a cold, scaling YOLO is not stampeded.
- **Permanent failures are returned as data, not raised**, so Durable does not retry them
  at the orchestration level either. The document completes as `partial`.
- **Blob references only** in orchestration history — never image bytes. Durable
  serialises and replays every activity input/output.
- **Deterministic instance IDs** — the caller's `correlationId`, else
  `sha256(blob path + size)` — so a retried or duplicated submission is a no-op
  instead of processing and billing the document twice.
- **Separate activities** for Content Understanding and OpenAI: a 429 from one must not
  re-run the other.

## Run it locally, without Azure

Proves the logic — retry classification, `(page, figure)` correlation after batching,
partial-failure handling — with no Functions host or storage account.

```powershell
pip install -r requirements.txt
python scripts/fetch_samples.py                  # real arXiv papers with diagrams
python scripts/run_local.py "samples/*.pdf" --workers 8 --chunk 8
```

Dial the injected faults up to watch the policy work:

```powershell
$env:SIM_429_RATE='0.20'; $env:SIM_503_RATE='0.08'; $env:SIM_400_RATE='0.02'
python scripts/run_local.py "samples/*.pdf"
```

```
4 document(s), 53 pages, 65 figures
  retry_ThrottledError         26     <- 429s, honoured Retry-After
  retry_TransientError          9     <- 503s, short jittered backoff
  permanent_cu / permanent_openai 3   <- 400s, correctly NOT retried
  partial docs                  2     <- completed with per-page errors
```

## Run it as Functions

```powershell
cp local.settings.json.example local.settings.json
# start Azurite, then:
func start
```

| Endpoint | Purpose |
|---|---|
| `GET /api/ui` | the demo page |
| `POST /api/upload` (multipart `file`) | upload PDFs **and** start the run |
| `POST /api/start` `{"files":["a.pdf"]}` | start a batch for files already in the container |
| `GET /api/status/{instanceId}` | progress for one file |
| `GET /api/batch/{batchId}` | progress for a whole upload set |
| `GET /api/status/by-file/{name}` | resolve a run from the file name |

`set_custom_status` publishes progress *while the run is going*, so a UI can show
"page 143 of 300" rather than just Running/Completed.

To exercise the real ACA scale-out retry path, point at the live app:

```
"YOLO_URL": "https://yolo-demo.<region>.azurecontainerapps.io"
```

With `YOLO_URL` empty, YOLO is simulated and the whole pipeline runs for free.

## Deploy it to Azure

[infra/](infra/README.md) has Terraform for the storage account, the Flex
Consumption API and worker Function Apps, and a **private** YOLO API on Container
Apps. The two Function Apps have independent FC1 plans, identities, host storage,
telemetry, and scaling. Storage is private-endpoint-only with shared keys
disabled, so every storage and task-hub connection uses managed identity.

```powershell
cd infra
Copy-Item terraform.tfvars.example terraform.tfvars   # then edit
terraform init ; terraform apply
cd ..
.\deploy.ps1 -Target both
```

For the one-time migration from the former combined Function App, the deployment
script sends the role-gated package to API first and worker second. Durable work
can wait safely in storage during that short gap, and the order prevents two
worker hosts from consuming the same task hub. Later releases can use
`-Target api` or `-Target worker` independently.

## Measured on Azure (2026-08-11)

Deployed for real: Flex Consumption (FC1) + private storage + private YOLO on
Container Apps. Driven by [scripts/azure_e2e.py](scripts/azure_e2e.py), 4
concurrent uploaders, one PDF each. These runs used the default
`TEXT_SERVICE=content-understanding` path: simulated Content Understanding and
Azure OpenAI calls, plus PyMuPDF raster/vector figure extraction. They did not
call live Mistral Document AI.

| | warm YOLO | YOLO cold (0 replicas) | re-verified, cold | final pre-shutdown |
|---|---:|---:|---:|---:|
| batches Completed | 4 / 4 | 4 / 4 | 4 / 4 | 4 / 4 |
| documents | 4 | 4 | 4 | 4 |
| pages | 53 | 53 | 53 | 53 |
| figures | 65 | 65 | 65 | 65 |
| partial documents | 2 | 2 | 2 | 0 |
| elapsed | 19.9 s | 39.2 s | 29.5 s | 38.7 s |
| throughput | 2.66 pages/s | 1.35 pages/s | 1.79 pages/s | 1.37 pages/s |

Four separate runs, hours apart, with a redeploy in between. Pages and figures
match the local baseline **exactly** every time (attention 15p/6f, bert 16p/39f,
resnet 12p/7f, yolo 10p/13f), so page splitting and figure extraction behave
identically in Azure. These historical runs did not assert on figure-level YOLO
errors, so they do **not** prove that ACA inference succeeded.

The former combined Function App was intentionally stopped after this historical
verification. The separate API and worker apps are live as of 2026-08-18.

Retries actually fired — 33 of them in the cold run, all absorbed:

| service | ThrottledError (429) | TransientError | max attempt |
|---|---:|---:|---:|
| content-understanding | 17 | 3 | 2 |
| azure-openai | 7 | 6 | 2 |
| yolo | 0 | 0 | — |

Two honest notes on that table:

- **The historical YOLO result was a false positive.** App-level internal
  ingress accepts traffic only from the same ACA environment. The Function is
  in another VNet subnet, so ACA returned its branded 404 "Container App -
  Unavailable" page before `/predict`; the old E2E check validated figure counts
  but not figure inference errors. The fix uses app-level external ingress on an
  internal-only ACA environment, which remains VNet-private, and the E2E check
  now fails on any `figure_errors`.
- **The 2 partial documents in the first three runs are correct behaviour, not failures.** They come
  from the injected 1% permanent-400 rate: a permanent error is deliberately
  *not* retried, the page records the error, and the document is marked
  `partial` instead of failing the batch. The final run had no partial documents.

Re-verified on 2026-08-18 after the ingress and assertion fixes. A warm run
completed one 15-page document with 6 figure responses from private ACA and
`figure_errors: 0`. A second run started with revision `yolo-api--0000004` at
zero replicas and completed with the same counts and zero figure errors in
89.0 seconds; the revision then reported one replica. YOLOv8n returned zero
object detections for those academic diagrams, which is valid model output:
the stock COCO model has no `diagram` class. These runs prove pipeline-to-ACA
execution and scale-from-zero, not diagram-recognition accuracy.

The API/worker split was applied and verified live on 2026-08-18 while
preserving `https://func-docpipe-v524ruos.azurewebsites.net`. Azure indexed
exactly 6 API functions and 7 worker functions. After a focused 15-page run
completed in 64.6 seconds, the four-user test completed all 4 batches in 43.1
seconds: 4 documents, 53 pages, 65 figures, zero figure errors, and exact local
baseline counts. One document was `partial` because the demo intentionally
injects permanent page errors; no batch failed.

## Output

One JSON per file, one entry per page:

```json
{ "file": "bert.pdf", "pages": 16, "figures": 39, "status": "partial",
  "pages_with_errors": 1,
  "results": [
    { "page": 3, "status": "ok",
      "content_understanding": { "markdown": "...", "fields": {}, "tables": [] },
      "openai": { "summary": "...", "usage": {} },
      "figures": [
        { "figure": 0, "kind": "vector", "width": 186, "height": 133,
          "count": 1, "detections": [ { "label": "diagram", "confidence": 0.9 } ] }
      ] },
    { "page": 7, "status": "partial",
      "errors": [ { "service": "content-understanding", "error": "... (400)" } ],
      "figures": [] }
  ] }
```

A page that fails permanently does not fail the document — the file is marked `partial`
and the failure is recorded per page and per service.

## Known limitations

- **Azure OpenAI and Content Understanding are simulated.** Swap the two functions in
  [`shared/simulators.py`](shared/simulators.py) for real SDK calls; the retry policy and
  the orchestration do not change. Mistral Document AI in
  [`shared/docai.py`](shared/docai.py) is a **real** client — it calls the live API as soon
  as `MISTRAL_API_KEY` is set.
- **Vector clustering is heuristic** (only used on the `content-understanding` path).
  Proximity merging can over-merge two adjacent diagrams or split a sparse one. Switching
  `TEXT_SERVICE=mistral-docai` replaces it with model-detected figure regions.
- **Figure extraction re-reads the PDF per page.** Fine at these sizes; for very large
  documents, cache the rendered pages rather than re-opening the file in every activity.
- **Per-page APIs dominate the cost.** A 300-page document is ~600 text/vision calls
  against a throttled service versus ~8 batched YOLO calls. Size
  `maxConcurrentActivityFunctions` against the OpenAI quota, not against YOLO.
- **No blob trigger on Flex Consumption.** The classic polling trigger is
  unsupported — the host logs *"The Flex Consumption SKU only supports EventGrid
  as the source for BlobTrigger functions"* once and disables the function.
  Every other function keeps working, so the app looks perfectly healthy while
  that entry point is dead. Wiring the Event Grid path (system topic +
  subscription → `/runtime/webhooks/blobs`) reached a passing validation
  handshake, but real `BlobCreated` deliveries still returned 500 — 5 events
  matched the filter, 19 delivery attempts, 0 successes. **I did not get it
  working.** It was removed rather than left half-wired, which costs nothing
  here because `/api/upload` is the only possible writer anyway.
- **`AzureWebJobsStorage__serviceUri` breaks the host.** Do not add it while
  chasing the above. A single generic URI cannot serve blob, queue and table, so
  Durable loses its queues and the site returns 503 *"Site Unavailable"* — while
  ARM still reports `state=Running`, `availabilityState=Normal`. Use
  `__accountName` + `__credential`.
- **The private hop to YOLO is HTTP, not HTTPS.** ACA's load-balancer ingress in
  this environment does not complete a TLS handshake on the default domain
  (`SSLEOFError`). Acceptable only because the ACA environment is internal and
  the hop never leaves the VNet; put a custom domain and certificate on the
  environment if it must be encrypted.
