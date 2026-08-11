# Document pipeline — Durable Functions fan-out / fan-in

Blob upload → per-page AI extraction → batched YOLO on figures → one JSON per file,
with status polling and a completion webhook.

```
uploads/*.pdf ──▶ blob trigger (deterministic instance id, duplicate-safe)
                    │
                    ▼   one orchestration PER FILE
        ┌───────────────────────────────────────────────┐
        │ PHASE 1 — fan out per PAGE                    │
        │   Content Understanding  (1 call / page)      │
        │   Azure OpenAI           (1 call / page)      │
        │   extract figures        (raster + vector)    │
        ├───────────────────────────────────────────────┤
        │ collect figures across ALL pages → chunk by 8 │
        ├───────────────────────────────────────────────┤
        │ PHASE 2 — fan out per CHUNK → YOLO (batched)  │
        ├───────────────────────────────────────────────┤
        │ PHASE 3 — merge → results/<file>.json         │
        └───────────────────────────────────────────────┘
                    │
                    ▼  webhook (NOTIFY_URL) + GET /api/status/{id}
```

**Diagrams** (open with [diagrams.net](https://app.diagrams.net)):
[docs/architecture.drawio](docs/architecture.drawio) — Azure hosting, private
networking and the notification path back to the uploading app.
[docs/pipeline.drawio](docs/pipeline.drawio) — the orchestration phases.

## Why the shape is what it is

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
- **Deterministic instance IDs** (`sha256(blob path + size)`) so a duplicate blob trigger
  is a no-op instead of processing and billing the document twice.
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
| drop a PDF in `uploads/` | starts a run automatically |
| `POST /api/start` `{"files":["a.pdf"]}` | start a batch, returns ids |
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
Consumption function app and a **private** YOLO API on Container Apps. Storage is
private-endpoint-only with shared keys disabled, so everything authenticates with
managed identity — the browser talks to `/api/upload`, and the function is the
only thing with a network path to either blob or the model.

```powershell
cd infra
Copy-Item terraform.tfvars.example terraform.tfvars   # then edit
terraform init ; terraform apply
```

## Measured on Azure (2026-08-11)

Deployed for real: Flex Consumption (FC1) + private storage + private YOLO on
Container Apps. Driven by [scripts/azure_e2e.py](scripts/azure_e2e.py), 4
concurrent uploaders, one PDF each.

| | warm YOLO | YOLO cold (0 replicas) |
|---|---:|---:|
| batches Completed | 4 / 4 | 4 / 4 |
| documents | 4 | 4 |
| pages | 53 | 53 |
| figures | 65 | 65 |
| partial documents | 2 | 2 |
| elapsed | 19.9 s | 39.2 s |
| throughput | 2.66 pages/s | 1.35 pages/s |

Pages and figures match the local baseline **exactly** (attention 15p/6f,
bert 16p/39f, resnet 12p/7f, yolo 10p/13f), so page splitting and figure
extraction behave identically in Azure.

Retries actually fired — 33 of them in the cold run, all absorbed:

| service | ThrottledError (429) | TransientError | max attempt |
|---|---:|---:|---:|
| content-understanding | 17 | 3 | 2 |
| azure-openai | 7 | 6 | 2 |
| yolo | 0 | 0 | — |

Two honest notes on that table:

- **YOLO needed no retries even from zero replicas.** ACA's ingress held the
  connection while a replica started rather than returning 503, so the
  `SCALE_SENSITIVE` path was never exercised in this run. It is not yet proven
  under real scale-out pressure — a 4-document test does not push ACA hard
  enough to shed load.
- **The 2 partial documents are correct behaviour, not failures.** They come
  from the injected 1% permanent-400 rate: a permanent error is deliberately
  *not* retried, the page records the error, and the document is marked
  `partial` instead of failing the batch.

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
