# Infrastructure

Terraform for the infrastructure that must exist before the pipeline runs:
private data/task-hub storage, a public **API Function App**, a separate
**Durable worker Function App**, and the **private YOLO API on Container Apps**.

The two Function Apps deploy the same Python package, but Terraform sets
`FUNCTION_ROLE=api` or `FUNCTION_ROLE=worker`. Role-gated blueprints ensure the
API host indexes only six HTTP/client functions and the worker host indexes only
two orchestrators and five activities.

## Where to host what: two Flex apps *and* Container Apps

Both, for different jobs. They are not competing options here.

**The API and orchestration run in separate Flex Consumption apps.** Flex permits
one app per FC1 plan, so each app has its own plan, host storage, identity,
Application Insights resource, configuration, deployment, and scale ceiling.
They share the delegated VNet integration subnet.

The split is an operational boundary, not a second workflow system:

- The API app owns UI, upload, start, and status. It scales from HTTP traffic.
- The worker app owns all orchestrators and activities. It scales from Durable
  task-hub work, keeps one `durable` instance always ready, and is the only app
  allowed to consume that hub.
- The API uses an explicit identity-based `WorkerStorage` client connection to
  start and query the worker hub. Status state lives in storage, so the functions
  do not need to share a host.

The worker remains on Functions Flex Consumption because the workload is
bursty, idle most of the time, and its hardest constraint is *bounding* fan-out
rather than raw throughput:

- `maxConcurrentActivityFunctions` is the only knob that bounds how many pages
  hit Azure OpenAI at once. That is a Functions host feature, not something you
  get for free in a container.
- One always-ready `durable` instance keeps the private keyless task hub polling;
  Flex still scales out from Durable backlog and scales excess instances in.
- Native VNet integration, which plain Consumption (Y1) does not have — and this
  deployment cannot work without it, since storage is private-endpoint-only.
- Managed identity for `AzureWebJobsStorage`, which is mandatory when shared keys
  are disabled.
- No container image to build, scan or patch for the orchestration tier.

**The model runs on Container Apps**, because YOLO has the opposite profile:

- The image is large (528 MB for PyTorch, 6 GB for the GPU Paddle build) and the
  model must stay resident in memory across requests. A pool of model instances
  inside one long-lived process is the whole performance story — measured 104 ms
  per image at batch 8.
- GPU is available through workload profiles. Functions has no GPU option.
- KEDA concurrency scaling can be matched to the in-process worker pool, so a
  replica only sheds load once its own pool is genuinely saturated.

Running YOLO *inside* the worker would mean loading a multi-hundred-megabyte
model into Durable instances and coupling GPU, model memory, cold start, and
deployment to orchestration. Running the orchestration as a plain ACA service
would mean rebuilding Durable state, replay, retries, and status.

## The shape, and why

```
Browser (public)
   │  HTTPS
   ▼
API Function App
 /api/ui, /api/upload, /api/status
  │          │ Durable client: WorkerStorage
  │          ▼
  │    Worker task hub (blob / queue / table)
  │          │
  │          ▼
  │    Durable Worker Function App
  │          │ VNet-private HTTP
  │          ▼
  └────> data blobs        Container Apps YOLO API
       public access     external app ingress on an
       DISABLED          internal-only ACA environment
```

The API app has a separate private storage account for its own Functions host
and deployment package. The worker/data account contains `uploads`, `work`,
`results`, the worker deployment package, and Durable task-hub state. The API
identity can write application blobs and act as a Durable client; only the
worker identity indexes triggers that consume task-hub messages.

`external_enabled = true` on the YOLO app does **not** make it internet-facing.
ACA uses "external" to mean traffic entering the app from outside its own ACA
environment. The ACA environment has an internal load balancer, so the endpoint
is still private to the VNet; this setting is required for the Function subnet
to reach it. With app-level external ingress disabled, only containers in the
same ACA environment can call the app, and the Function receives ACA's branded
404 "Container App - Unavailable" page.

The subscription's policy disables public network access and shared access keys
on every storage account. Rather than fight that, the configuration declares the
same posture, which has two consequences worth stating plainly:

- **No connection strings anywhere.** Both hosts use
  `AzureWebJobsStorage__accountName` + `managedidentity`; the app code uses
  `DefaultAzureCredential`. The API's cross-app Durable connection uses
  `WorkerStorage__accountName` + `managedidentity`, and the Durable extension in
  both packages explicitly selects `WorkerStorage`. That is why
  [rbac.tf](rbac.tf) grants Blob **and** Queue **and** Table data roles.
- **The browser cannot be handed a SAS.** There is no network path from the
  internet to the blob endpoint, so uploads go through the function, which is
  the only component inside the VNet. This is also what lets YOLO stay private:
  nothing except the function ever needs to reach it.

## Files

| File | Contains |
|---|---|
| [providers.tf](providers.tf) | azurerm ~> 5.0, `storage_use_azuread = true` |
| [network.tf](network.tf) | VNet, 3 subnets, private endpoints + DNS for blob/queue/table |
| [storage.tf](storage.tf) | Private key-less worker/data and API-host storage accounts |
| [functions.tf](functions.tf) | Two FC1 plans and role-specific API/worker Function Apps |
| [aca.tf](aca.tf) | Internal ACA environment, YOLO app, private DNS for its domain |
| [rbac.tf](rbac.tf) | Data-plane role assignments, AcrPull |
| [outputs.tf](outputs.tf) | API/worker names, UI URL, task hub, storage, private YOLO FQDN |

## Details that are easy to get wrong

**Subnet sizes are not arbitrary.** `snet-func` must be delegated to
`Microsoft.App/environments` for Flex VNet integration; `snet-pe` must have
private endpoint network policies disabled; `snet-aca` must be at least a `/23`.

**The ACA private DNS zone is required, not optional.** With
`internal_load_balancer_enabled = true` the ingress FQDN resolves to a VNet
address that public DNS knows nothing about. Without the zone and wildcard A
record, every YOLO call fails with a name-resolution error that looks nothing
like a networking problem.

**`function_max_instances` applies to the worker and multiplies with `host.json`.** The
`maxConcurrentActivityFunctions: 8` setting bounds concurrency *per instance*,
so 40 instances means up to 320 simultaneous calls into Azure OpenAI. Size this
against the model quota, not against YOLO.

**The API has its own ceiling.** `api_function_max_instances` controls the HTTP
tier independently, so upload/status traffic cannot request worker instances and
page fan-out cannot request API instances.

**One task hub has one worker application.** The API may start and query
`docpipeline` through its Durable client binding, but it must never register
orchestrator or activity triggers for that hub. Two worker apps consuming one
hub compete for messages and can leave orchestrations stuck.

**Durable must use `WorkerStorage`, not Flex's generated host connection.** The
subscription disables shared keys, while the provider-generated
`AzureWebJobsStorage` value contains an empty `AccountKey`. The empty app-setting
override keeps the Functions host on managed identity; `host.json` selects the
separate `WorkerStorage__accountName` + `managedidentity` prefix for all Durable
state. Azure normalizes the empty overrides to `null`, so Terraform ignores only
those two map entries after creation rather than hiding broader app-setting
drift.

**`INFERENCE_WORKERS` × threads must be ≤ `yolo_cpu`.** `os.cpu_count()` inside
a container reports host cores, not the ACA limit, so both are set explicitly.
The HTTP scale rule uses the same worker count so a replica only sheds load once
its own pool is genuinely saturated.

## Use

```powershell
Copy-Item terraform.tfvars.example terraform.tfvars   # then edit
terraform init
terraform plan
terraform apply
```

Then deploy the application code (there is no `func` dependency):

```powershell
cd ..
.\deploy.ps1 -Target both
```

The script deploys API first and worker second. That order matters during the
one-time migration from the former combined host: it creates a short processing
gap while Durable messages wait safely in storage, rather than an overlap where
two workers consume the same hub. After migration, deploy independently with
`-Target api` or `-Target worker`.

Open the `ui_url` output.

## Verified deployment

The two-app Terraform migration was applied to `rg-docpipeline` on 2026-08-18.
It preserved the public API app and hostname, then added the worker FC1 plan,
Function App, Application Insights resource, private API host-storage account,
private endpoints, and RBAC. Azure indexes exactly 6 API functions and 7 worker
functions. A post-deployment Terraform plan exits `0` with no changes.

The final live four-user E2E completed all 4 batches in 43.1 seconds: 4
documents, 53 pages, 65 figures, zero figure errors, and exact local baseline
counts. A focused 15-page/6-figure run also completed in 64.6 seconds. Both runs
used the public API, worker-owned private task hub, private blob/queue/table
endpoints, and private ACA YOLO ingress.

The hardened 2026-08-18 verification started with the YOLO revision at zero
replicas and completed a 15-page document with 6 figure responses and zero
figure inference errors in 89.0 seconds. ACA then reported one active replica.
The E2E check now fails if any figure carries an inference error; see the
pipeline README for why the earlier count-only checks were insufficient.
