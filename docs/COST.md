# Cost guide

This page is the short operational view of cost. The calculations, raw methodology, pricing
rates, corrections, and caveats live in [BENCHMARK.md](../BENCHMARK.md#cost).

## Measured model-compute cost

Four randomized two-hour soaks ran on Azure Container Apps. Cost per 1,000 uses images that
were actually processed, not merely sent.

| Deployment | Run cost | Processed | Cost / 1k processed | One warm replica / month |
|---|---:|---:|---:|---:|
| YOLO PyTorch, 2 vCPU / 4 GiB | $0.60 | 5,371 | $0.11 | $47-$158 |
| YOLO ONNX, 2 vCPU / 4 GiB | $1.07 | 15,140 | $0.07 | $47-$158 |
| PP-StructureV3 CPU, 4 vCPU / 8 GiB | $1.12 | 417 | $2.68 | $95-$315 |
| PP-StructureV3 GPU, NC8as-T4 | $0.83 | 967 | $0.85 | $237 |

Total estimated compute for the four runs was **$3.62**.

The soak harness conservatively bills every observed replica-second at the active rate. CPU
replicas held warm but idle receive a lower ACA idle rate, so the run costs are upper bounds.
Serverless GPU has no idle tier.

## Keeping one replica warm

The monthly values use 730 hours, or 2,628,000 seconds. A CPU replica can move between idle
and active billing, which produces a range. An allocated serverless T4 is billed at its flat
rate even when no requests are running.

| Deployment | Median cold start avoided | Warm-idle floor | Warm-active ceiling |
|---|---:|---:|---:|
| YOLO PyTorch | 2.122 s | $47/month | $158/month |
| YOLO ONNX | 0.181 s | $47/month | $158/month |
| PP-StructureV3 CPU | 25.642 s | $95/month | $315/month |
| PP-StructureV3 GPU | 31.709 s | $237/month | $237/month |

Practical implications:

- Keep YOLO ONNX at `min-replicas 0`; paying at least $47/month to avoid a 181 ms process
  startup is not justified by these measurements.
- PP-StructureV3 has a 26-32 second process startup, so a warm replica can be justified by a
  latency requirement, but the GPU costs $237/month even when idle.
- The T4 was much cheaper than CPU per fully utilized PP-StructureV3 page, but a mostly idle
  GPU can still dominate the monthly bill.
- The synthetic benchmark is deliberately light and does not represent a dense production
  scan with every PP-StructureV3 feature enabled.

## What is not included

The comparison above is model-container compute only. It does not include:

- Azure Functions Flex Consumption execution.
- Storage capacity, transactions, or private endpoints.
- Azure AI Content Understanding usage.
- Azure OpenAI usage.
- Mistral Document AI API usage.
- ACR storage, networking, logging, or taxes.

The document pipeline uses simulated Content Understanding and Azure OpenAI implementations
in this repository, so there is no measured end-to-end per-document API cost to quote. Cost a
production pipeline from its real page count, token use, figure count, retention, and retry
rate rather than adding those services to the model benchmark.

## Cost controls

Use the controls that already ship with the repository:

```powershell
# Inspect and then scale every model-comparison ACA app to zero.
Set-Location deploy
.\scaledown.ps1 -ShowOnly
.\scaledown.ps1

# Every soak should have a hard budget stop.
Set-Location ..\loadtest
python soak.py --host https://<fqdn> --model yolo --profile cpu-2 `
  --app yolo-demo --hours 2 --budget-usd 25
```

Additional controls:

- Keep `yolo_min_replicas = 0` in `docpipeline/infra/terraform.tfvars` unless an SLO requires
  warm capacity.
- Stop a test Function App when it is not being demonstrated, or use `terraform destroy` to
  remove the private pipeline infrastructure.
- Set retention and lifecycle rules for uploaded PDFs, page images, extracted figures, and
  result JSON before production use.
- Recheck current Azure and external API pricing before using these numbers in a forecast.

The benchmark's [verified pricing table](../BENCHMARK.md#verified-rates) records the rates used
for these calculations.