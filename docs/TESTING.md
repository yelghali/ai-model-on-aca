# Testing and reproduction

There are two test surfaces: direct model APIs on Azure Container Apps and the Durable
document workflow. Run correctness checks before throughput tests.

## Install model-test dependencies

```powershell
Set-Location loadtest
python -m pip install -r requirements.txt
```

## Model API checks

Use the endpoint that matches the deployed API:

| Model | Endpoint | Soak `--model` | Soak `--profile` |
|---|---|---|---|
| YOLO PyTorch | `/predict` | `yolo` | `cpu-2` |
| YOLO ONNX | `/predict` | `yolo` | `cpu-2` |
| EasyOCR | `/ocr` | n/a | `cpu-2` |
| PP-StructureV3 CPU | `/parse` | `paddle` | `cpu-4` |
| PP-StructureV3 GPU | `/parse` | `paddle` | `gpu-t4` |

### 1. Correctness under concurrency

The correctness test marks every image uniquely and verifies that each response belongs to
its own input. It tests cross-request contamination, not model accuracy.

```powershell
python correctness.py --host https://<yolo-fqdn> --endpoint /predict `
  --count 24 --concurrency 8
```

### 2. In-replica concurrency

```powershell
python parallel_demo.py --host https://<yolo-fqdn> --endpoint /predict `
  --count 8 --concurrency 8
python bench.py --host https://<yolo-fqdn> --endpoint /predict `
  --levels 1,2,4,8 --requests 40
```

Compare these client results with `GET /stats`. `peak_running` proves how many calls entered
inference at once; `peak_in_flight` includes queued requests.

### 3. Batch and cost sweep

`cost_bench.py` is designed for PP-StructureV3's multi-file `/parse` request:

```powershell
python cost_bench.py --host https://<paddle-cpu-fqdn> --endpoint /parse `
  --profile cpu-4 --batches 1,2,4,8
python cost_bench.py --host https://<paddle-gpu-fqdn> --endpoint /parse `
  --profile gpu-t4 --batches 1,2,4,8,16
```

### 4. Multi-hour soak

Always set `--budget-usd`. The harness writes one CSV row per request and a JSON summary.

```powershell
python soak.py --host https://<yolo-fqdn> --model yolo --profile cpu-2 `
  --app yolo-demo --hours 2 --budget-usd 25 --seed 4242 --out soak-yolo

python soak.py --host https://<yolo-onnx-fqdn> --model yolo --profile cpu-2 `
  --app yolo-onnx-demo --hours 2 --budget-usd 25 --seed 4242 --out soak-yolo-onnx

python soak.py --host https://<paddle-cpu-fqdn> --model paddle --profile cpu-4 `
  --app paddle-demo-cpu --hours 2 --budget-usd 25 --seed 4242 --out soak-cpu

python soak.py --host https://<paddle-gpu-fqdn> --model paddle --profile gpu-t4 `
  --app paddle-demo-gpu --hours 2 --budget-usd 25 --seed 4242 --out soak-gpu
```

The same random seed does not produce identical traffic totals: faster services complete more
batches before a timed phase ends and therefore consume the random stream differently. Compare
per-image metrics, error rates, and cost per processed image rather than total requests.

See [BENCHMARK.md](../BENCHMARK.md) for the raw-result interpretation and caveats. Scale apps
back to zero after testing with `deploy/scaledown.ps1`.

## Durable pipeline: local verification

```powershell
Set-Location docpipeline
python -m pip install -r requirements.txt
python scripts/fetch_samples.py
python scripts/run_local.py "samples/*.pdf" --workers 8 --chunk 8
python scripts/batch_test.py "samples/*.pdf" --waves 3 --spike 8 `
  --workers 8 --files-parallel 3 --chunk 8
```

The local baseline for the checked-in sample corpus is:

| Document | Pages | Figures |
|---|---:|---:|
| `attention-is-all-you-need.pdf` | 15 | 6 |
| `bert.pdf` | 16 | 39 |
| `resnet-deep-residual-learning.pdf` | 12 | 7 |
| `yolo-unified-detection.pdf` | 10 | 13 |
| **Total** | **53** | **65** |

This 65-figure baseline is for the default `TEXT_SERVICE=content-understanding` path, where
PyMuPDF extracts raster and clustered vector figures. A live Mistral Document AI run uses
model-detected regions and may legitimately produce different figure counts.

## Durable pipeline: Azure end-to-end verification

After Terraform and Function App deployment:

```powershell
Set-Location docpipeline
python scripts/fetch_samples.py
python scripts/azure_e2e.py https://<function-app>.azurewebsites.net `
  --users 4 --files-per-user 1 --timeout 900
```

The test verifies that:

- `POST /api/upload` accepts multipart PDFs and immediately returns batch and instance IDs.
- Every Durable batch reaches `Completed`.
- Deployed page and figure counts match the local baseline.
- Every result records a result blob and a valid document status.
- Transient status-API 5xx responses during Flex scale-out do not create false failures.

The 2026-08-11 pre-shutdown run completed 4 of 4 batches with 53 pages, 65 figures, zero
partial documents, and 38.7 seconds elapsed. The former combined app was stopped after that
run. The two-app deployment was verified live on 2026-08-18: all 4 batches completed in 43.1
seconds with 53 pages, 65 figures, zero figure errors, and exact baseline counts.

The supported ingestion path is `/api/upload`. Do not test by dropping a blob into storage:
there is no supported blob trigger in this Flex Consumption deployment.