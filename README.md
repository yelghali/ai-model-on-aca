# AI Model Comparison and Durable Document Workflows on Azure

This repository has two connected tracks:

1. **AI model comparison on Azure Container Apps (ACA):** deploy and measure YOLO with
   PyTorch or ONNX Runtime, PP-StructureV3 on CPU or serverless GPU, and EasyOCR. The
   comparison covers concurrency, autoscaling, cold starts, correctness, failures, container
   size, memory, and cost.
2. **A Durable document-processing workflow:** upload PDFs through Azure Functions, fan out
   work per page, extract text and figures, batch figures through a private YOLO service,
   merge one result per document, expose status APIs, and send a completion webhook.

## Measured model comparison

Four randomized two-hour soaks ran on ACA in `swedencentral`. These models do not perform
identical tasks, so compare serving characteristics and unit economics rather than model
quality.

| Deployment | Workload | Compute | Processed | Failed batches | Mean inference / image | Median cold start | Cost / 1k processed | One warm replica / month |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| YOLO PyTorch | Object detection | 2 vCPU / 4 GiB | 5,371 | 0 / 1,047 | **104 ms** | 2.122 s | $0.11 | $47-$158 |
| YOLO ONNX | Object detection | 2 vCPU / 4 GiB | **15,140** | 0 / 2,697 | 109 ms | **0.181 s** | **$0.07** | $47-$158 |
| PP-StructureV3 CPU | Layout + OCR | 4 vCPU / 8 GiB | 417 of 640 sent | **37 / 177** | 11,170 ms | 25.642 s | **$2.68** | $95-$315 |
| PP-StructureV3 GPU | Layout + OCR | NC8as-T4 | 967 | 0 / 274 | 855 ms | 31.709 s | $0.85 | $237 |

Across the four runs: **22,118 images sent, 21,895 verified correct, zero
cross-contamination, and $3.62 estimated model-compute cost**. The synthetic workload and
cost methodology have important caveats; read [the benchmark report](BENCHMARK.md) before
quoting a value.

## Track 1: model APIs and comparison

Each model is wrapped in FastAPI and deployed with ACA HTTP-concurrency autoscaling. Blocking
inference runs in a bounded executor. Stateful high-level model wrappers use one instance per
worker; the thread-safe ONNX Runtime path shares one session.

| Folder | Model and runtime | Endpoint | Role |
|---|---|---|---|
| [yolo-demo/](yolo-demo/) | Ultralytics YOLO on PyTorch | `POST /predict` | detection baseline and model-pool design |
| [yolo-onnx-demo/](yolo-onnx-demo/) | YOLO on ONNX Runtime | `POST /predict` | lighter image, memory, and cold start |
| [paddle-demo/](paddle-demo/) | PP-StructureV3 on CPU or T4 | `POST /parse` | document layout and OCR comparison |
| [ocr-demo/](ocr-demo/) | EasyOCR on PyTorch | `POST /ocr` | additional pooled-model concurrency example |

Every API also exposes `GET /healthz`, `GET /stats`, and interactive documentation at
`GET /docs`. The load harnesses verify per-input identity under concurrency, not subjective
model accuracy.

## Track 2: Durable document pipeline

The [document pipeline](docpipeline/README.md) is a real Durable Functions fan-out/fan-in
workflow backed by private storage and a private ACA-hosted YOLO service. Public HTTP/client
functions and Durable processing run in separate Flex Consumption Function Apps:

```text
POST /api/upload
   -> API app targets the worker-owned task hub
   -> worker app starts one Durable orchestration per PDF
  -> fan out per page for text, summary, and figure extraction
  -> collect figures across the document and chunk by 8
  -> fan out batched YOLO calls
  -> merge results/<file>.json
  -> status API and signed completion webhook
```

The supported ingestion path is `POST /api/upload`; there is intentionally no blob-trigger
path on Flex Consumption.

### Which service extracts figures?

The verified Azure runs used the default `TEXT_SERVICE=content-understanding` path: the
repository's simulated Content Understanding supplied page text and **PyMuPDF extracted
embedded raster images plus clustered vector figures**.

Set `TEXT_SERVICE=mistral-docai` to use **Mistral Document AI
(`mistral-ocr-latest`)** for markdown and model-detected figure regions in one call. It calls
live Mistral only when `MISTRAL_API_KEY` is configured; without a key, that path is simulated
and figures fall back to PyMuPDF.

The 2026-08-11 pre-shutdown verification completed **4 of 4 batches, 53 pages, 65 figures,
zero partial documents, in 38.7 seconds**. The former combined Function App was stopped after
that run. The separate API and worker apps were deployed on 2026-08-18; their four-user E2E
completed the same 53 pages and 65 figures with zero figure errors in 43.1 seconds.

## Documentation

| Topic | Document |
|---|---|
| Setup and deployment for both tracks | [docs/SETUP.md](docs/SETUP.md) |
| In-process concurrency and ACA autoscaling | [docs/CONCURRENCY.md](docs/CONCURRENCY.md) |
| Full benchmark methodology, results, and caveats | [BENCHMARK.md](BENCHMARK.md) |
| Cost summary, exclusions, and shutdown controls | [docs/COST.md](docs/COST.md) |
| Model and Durable workflow testing | [docs/TESTING.md](docs/TESTING.md) |
| Durable workflow, APIs, reliability, output, and limitations | [docpipeline/README.md](docpipeline/README.md) |
| Terraform, private networking, identity, and Azure deployment | [docpipeline/infra/README.md](docpipeline/infra/README.md) |
| Azure hosting and private-network diagram | [docpipeline/docs/architecture.drawio](docpipeline/docs/architecture.drawio) |
| Durable orchestration diagram | [docpipeline/docs/pipeline.drawio](docpipeline/docs/pipeline.drawio) |
| Interactive presentation | [presentation.html](presentation.html) |

## Repository layout

```text
ai-model-on-aca/
|-- BENCHMARK.md              # authoritative four-model soak report
|-- docs/                     # setup, concurrency, cost, and testing guides
|-- yolo-demo/                # YOLO through PyTorch
|-- yolo-onnx-demo/           # YOLO through ONNX Runtime
|-- ocr-demo/                 # EasyOCR
|-- paddle-demo/              # PP-StructureV3 CPU/GPU
|-- deploy/                   # ACR build, ACA deploy, scale-down, cleanup
|-- loadtest/                 # correctness, latency, cost, and soak harnesses
`-- docpipeline/              # Durable workflow, UI, tests, diagrams, Terraform
```