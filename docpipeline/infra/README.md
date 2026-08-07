# Infrastructure

Terraform for the three pieces that need to exist before the pipeline can run:
**storage**, the **Durable Functions** app, and the **YOLO API on Container Apps
(private)**.

The demo UI and its upload API are *not* here on purpose — they are application
code inside the same function app (`GET /api/ui`, `POST /api/upload`), so they
ship with the deployment package rather than with the infrastructure.

## The shape, and why

```
Browser (public)
   │  HTTPS
   ▼
Function App  ── VNet integrated ──┐
 (public ingress: /api/ui, /api/upload, /api/status)
   │                               │
   │ private endpoints             │ internal ingress
   ▼                               ▼
Storage                        Container Apps
blob / queue / table           YOLO API
public access DISABLED         external_enabled = false
shared keys DISABLED
```

The subscription's policy disables public network access and shared access keys
on every storage account. Rather than fight that, the configuration declares the
same posture, which has two consequences worth stating plainly:

- **No connection strings anywhere.** The host uses
  `AzureWebJobsStorage__accountName` + `managedidentity`; the app code uses
  `DefaultAzureCredential`. That is why [rbac.tf](rbac.tf) grants Blob **and**
  Queue **and** Table data roles — Durable uses all three, and missing one
  produces a runtime that starts happily and then never schedules anything.
- **The browser cannot be handed a SAS.** There is no network path from the
  internet to the blob endpoint, so uploads go through the function, which is
  the only component inside the VNet. This is also what lets YOLO stay private:
  nothing except the function ever needs to reach it.

## Files

| File | Contains |
|---|---|
| [providers.tf](providers.tf) | azurerm ~> 5.0, `storage_use_azuread = true` |
| [network.tf](network.tf) | VNet, 3 subnets, private endpoints + DNS for blob/queue/table |
| [storage.tf](storage.tf) | Storage account (private, key-less) and containers |
| [functions.tf](functions.tf) | Flex Consumption plan + function app, identity-based |
| [aca.tf](aca.tf) | Internal ACA environment, YOLO app, private DNS for its domain |
| [rbac.tf](rbac.tf) | Data-plane role assignments, AcrPull |
| [outputs.tf](outputs.tf) | UI URL, function name, private YOLO FQDN |

## Details that are easy to get wrong

**Subnet sizes are not arbitrary.** `snet-func` must be delegated to
`Microsoft.App/environments` for Flex VNet integration; `snet-pe` must have
private endpoint network policies disabled; `snet-aca` must be at least a `/23`.

**The ACA private DNS zone is required, not optional.** With
`internal_load_balancer_enabled = true` the ingress FQDN resolves to a VNet
address that public DNS knows nothing about. Without the zone and wildcard A
record, every YOLO call fails with a name-resolution error that looks nothing
like a networking problem.

**`function_max_instances` multiplies with `host.json`.** The
`maxConcurrentActivityFunctions: 8` setting bounds concurrency *per instance*,
so 40 instances means up to 320 simultaneous calls into Azure OpenAI. Size this
against the model quota, not against YOLO.

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
Compress-Archive -Path * -DestinationPath app.zip -Force
az functionapp deployment source config-zip `
  -g <resource_group_name> -n $(terraform -chdir=infra output -raw function_app_name) `
  --src app.zip
```

Open the `ui_url` output.

## Not verified

This configuration passes `terraform validate` against azurerm 5.x. It has
**not** been applied end to end — the earlier CLI attempt was blocked by the
storage policy, which is what this module is designed to work with rather than
around. Treat the first `apply` as the real test.
