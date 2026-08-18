# Setup and deployment

This guide covers both parts of the repository:

1. Model APIs on Azure Container Apps.
2. The Durable Functions document pipeline that calls a private YOLO API.

For benchmark commands, use [TESTING.md](TESTING.md). For measured prices and cost
controls, use [COST.md](COST.md).

## Prerequisites

- An Azure subscription with permission to create resource groups, Azure Container Apps,
  Azure Container Registry, networking, storage, and Function Apps.
- [Azure CLI](https://learn.microsoft.com/cli/azure/install-azure-cli) authenticated with
  `az login`.
- PowerShell 5.1 or newer.
- Python and `pip` for local tests.
- Terraform for the private document-pipeline deployment.
- Azure Functions Core Tools and Azurite only when running the Function App locally.

The model deployment script builds in ACR, so local Docker is optional. GPU deployment also
requires regional ACA serverless-GPU availability and quota.

If PowerShell blocks local scripts for the current process:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
```

## Deploy the model APIs

The deployment script creates or reuses a resource group, ACR, and ACA environment, builds
the selected image in ACR, and configures HTTP-concurrency autoscaling.

```powershell
Set-Location deploy

# YOLO through Ultralytics and PyTorch
.\deploy.ps1 -Demo yolo -Location swedencentral

# YOLO through ONNX Runtime
.\deploy.ps1 -Demo yolo-onnx -Location swedencentral

# PP-StructureV3 on CPU
.\deploy.ps1 -Demo paddle -NameSuffix cpu -Cpu 4.0 -Memory 8.0Gi `
  -InferenceWorkers 1 -ConcurrentRequests 1 -Location swedencentral

# PP-StructureV3 on a serverless T4 profile
.\deploy.ps1 -Demo paddle -NameSuffix gpu -Gpu `
  -InferenceWorkers 1 -ConcurrentRequests 1 -Location swedencentral

# Optional EasyOCR comparison
.\deploy.ps1 -Demo ocr -Location swedencentral
```

Useful switches include `-SkipBuild`, `-MinReplicas`, `-MaxReplicas`,
`-InferenceWorkers`, `-ConcurrentRequests`, `-Cpu`, and `-Memory`. Keep the HTTP scale
threshold aligned with the number of inference workers unless a measured workload justifies
queueing inside a replica.

| Demo | Endpoint | Purpose |
|---|---|---|
| `yolo` | `POST /predict` | Ultralytics YOLO object detection |
| `yolo-onnx` | `POST /predict` | YOLO object detection through ONNX Runtime |
| `ocr` | `POST /ocr` | EasyOCR text recognition |
| `paddle` | `POST /parse` | PP-StructureV3 layout and OCR |

Every model app also exposes `GET /`, `GET /healthz`, `GET /stats`, and interactive API
documentation at `GET /docs`.

## Run a model locally

Example for the PyTorch YOLO API:

```powershell
Set-Location yolo-demo
docker build -t yolo-demo:local .
docker run --rm -p 8000:8000 -e INFERENCE_WORKERS=2 yolo-demo:local
```

Then open `http://localhost:8000/docs` or call `http://localhost:8000/healthz`.

## Run the document pipeline locally

The offline harness exercises page fan-out, figure extraction, retry classification,
figure batching, result correlation, and partial-document behavior without requiring a
Functions host or Azure resources.

```powershell
Set-Location docpipeline
python -m pip install -r requirements.txt
python scripts/fetch_samples.py
python scripts/run_local.py "samples/*.pdf" --workers 8 --chunk 8
```

The default extraction configuration is:

```powershell
$env:TEXT_SERVICE = 'content-understanding'
```

In this mode, Content Understanding supplies page text and PyMuPDF extracts raster and
clustered vector figures. The Content Understanding implementation in this repository is
currently simulated.

To use Mistral Document AI for both markdown and model-detected figure regions:

```powershell
$env:TEXT_SERVICE = 'mistral-docai'
$env:MISTRAL_API_KEY = '<key>'
python scripts/run_local.py "samples/*.pdf" --workers 8 --chunk 8
```

Without `MISTRAL_API_KEY`, the Mistral path is simulated and figure extraction falls back to
PyMuPDF. `MISTRAL_ENDPOINT` can target an Azure AI Foundry deployment of the same model.

To run the Function App locally, copy the example settings, start Azurite, and then start the
Functions host:

```powershell
Copy-Item local.settings.json.example local.settings.json
func start
```

## Deploy the Durable pipeline

The Terraform module creates private blob, queue, and table endpoints; separate Flex
Consumption API and Durable worker Function Apps; and an internal ACA-hosted YOLO API.
Each Function App has its own FC1 plan, host storage, identity, telemetry, scale ceiling,
and deployment lifecycle. The API targets the worker-owned task hub through an explicit
managed-identity Durable client connection.

```powershell
Set-Location docpipeline\infra
Copy-Item terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars for the subscription, registry, and desired extraction mode.
terraform init
terraform plan
terraform apply
Set-Location ..
.\deploy.ps1 -Target both
```

The deployment script sends the shared role-gated package to API first and worker second.
After the initial split, deploy either app independently with `-Target api` or
`-Target worker`. See [the infrastructure guide](../docpipeline/infra/README.md#use) for
the task-hub ownership and migration-order rationale.

The supported public entry point is `POST /api/upload`. It stores each PDF and starts its
orchestration in the same request. There is intentionally no blob-trigger path on Flex
Consumption. See [the pipeline guide](../docpipeline/README.md) for the API contract,
reliability policy, output shape, and known limitations.

## Stop or remove resources

Scale the comparison apps to zero after testing:

```powershell
Set-Location deploy
.\scaledown.ps1 -ShowOnly
.\scaledown.ps1
```

To remove the model-demo resource group entirely, run `deploy/cleanup.ps1`. For the Durable
pipeline, use `terraform destroy` from `docpipeline/infra` when the private infrastructure is
no longer needed.