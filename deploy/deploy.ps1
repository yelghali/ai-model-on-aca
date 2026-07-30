<#
.SYNOPSIS
    Build one of the model demos in ACR and deploy it to Azure Container Apps with HTTP autoscaling.

.EXAMPLE
    ./deploy.ps1 -Demo yolo -ResourceGroup rg-aca-model-demo -Location westeurope
    ./deploy.ps1 -Demo ocr  -ResourceGroup rg-aca-model-demo -Location westeurope
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)][ValidateSet('yolo', 'ocr')][string]$Demo,
    [string]$ResourceGroup = 'rg-aca-model-demo',
    [string]$Location = 'westeurope',
    # ACA capacity varies by region. If environment creation fails with
    # AKSCapacityHeavyUsage, rerun with a different -EnvLocation and -SkipBuild.
    [string]$EnvLocation = '',
    [string]$Registry = '',
    [string]$Environment = 'aca-model-demo-env',
    [int]$MinReplicas = 0,
    [int]$MaxReplicas = 10,

    # Container size. ACA gives CPU in cores; memory must follow the allowed ratios.
    [string]$Cpu = '2.0',
    [string]$Memory = '4.0Gi',

    # Parallel inferences INSIDE one replica (model-pool size).
    # NOTE: os.cpu_count() inside a container reports the HOST cores, not the ACA
    # limit, so we always set this explicitly rather than letting the app guess.
    [int]$InferenceWorkers = 2,
    [int]$TorchThreadsPerWorker = 1,

    # KEDA scale-out trigger. Matching it to InferenceWorkers means a replica only
    # gets more requests than it can run in parallel once a new replica is coming up.
    [int]$ConcurrentRequests = 0,

    [switch]$SkipBuild
)

$ErrorActionPreference = 'Stop'
$env:AZURE_CORE_ONLY_SHOW_ERRORS = 'true'

if ($ConcurrentRequests -le 0) { $ConcurrentRequests = $InferenceWorkers }
if (-not $EnvLocation) { $EnvLocation = $Location }

# --- helpers ---------------------------------------------------------------
# PowerShell 5.1 turns native-command stderr into a terminating error when
# $ErrorActionPreference = 'Stop' -- and `az` writes warnings to stderr. So run
# every az call with 'Continue' and decide success from $LASTEXITCODE instead.
#
# These are deliberately SIMPLE functions using $args: an advanced function
# (one with [Parameter(...)]) gains PowerShell common parameters, which makes
# az's own short flags like `-o` ambiguous with -OutVariable/-OutBuffer.

# Never echo raw arguments: they can contain registry passwords.
function Format-AzArgs {
    param([object[]]$AzArgs)
    $redactNext = $false
    $parts = foreach ($a in $AzArgs) {
        if ($redactNext) { $redactNext = $false; '***' }
        elseif ($a -match '^--(registry-password|password|secrets?)$') { $redactNext = $true; $a }
        else { $a }
    }
    $parts -join ' '
}

function Invoke-Az {
    $azArgs = $args
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $raw = & az @azArgs 2>&1
        $out = ($raw | Where-Object { $_ -isnot [System.Management.Automation.ErrorRecord] }) -join "`n"
        if ($LASTEXITCODE -ne 0) { throw "az $(Format-AzArgs $azArgs) failed:`n$raw" }
        return $out.Trim()
    }
    finally { $ErrorActionPreference = $prev }
}

# Same, but returns $null instead of throwing (for "does this exist?" probes).
function Try-Az {
    $azArgs = $args
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $raw = & az @azArgs 2>&1
        if ($LASTEXITCODE -ne 0) { return $null }
        $out = ($raw | Where-Object { $_ -isnot [System.Management.Automation.ErrorRecord] }) -join "`n"
        if ([string]::IsNullOrWhiteSpace($out)) { return $null }
        return $out.Trim()
    }
    finally { $ErrorActionPreference = $prev }
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$context = Join-Path $repoRoot "$Demo-demo"
$appName = "$Demo-demo"
$imageTag = "${appName}:v1"

if (-not (Test-Path $context)) { throw "Demo folder not found: $context" }

Write-Host "==> Demo: $Demo | RG: $ResourceGroup | Location: $Location | Env: $EnvLocation" -ForegroundColor Cyan
Write-Host "==> Sizing: $Cpu vCPU / $Memory | workers=$InferenceWorkers | scale @ $ConcurrentRequests concurrent" -ForegroundColor Cyan

Write-Host '==> Ensuring containerapp extension + providers' -ForegroundColor Cyan
Invoke-Az extension add --name containerapp --upgrade --allow-preview true --only-show-errors | Out-Null
Invoke-Az provider register --namespace Microsoft.App --wait | Out-Null
Invoke-Az provider register --namespace Microsoft.OperationalInsights --wait | Out-Null

# A resource group created in an earlier run may live in another region; that's fine,
# resources inside it can still be placed elsewhere.
$rgLocation = Try-Az group show -n $ResourceGroup --query location -o tsv
if ($rgLocation) {
    Write-Host "==> Reusing resource group: $ResourceGroup ($rgLocation)" -ForegroundColor Cyan
}
else {
    Write-Host "==> Creating resource group: $ResourceGroup ($Location)" -ForegroundColor Cyan
    Invoke-Az group create -n $ResourceGroup -l $Location | Out-Null
}

# --- ACR -------------------------------------------------------------------
$existing = Try-Az acr list -g $ResourceGroup --query "[0].name" -o tsv
if ($existing) {
    $Registry = $existing
    Write-Host "==> Reusing ACR: $Registry" -ForegroundColor Cyan
}
else {
    if (-not $Registry) { $Registry = "acamodeldemo$((New-Guid).ToString('N').Substring(0,8))" }
    Write-Host "==> Creating ACR: $Registry" -ForegroundColor Cyan
    Invoke-Az acr create -g $ResourceGroup -n $Registry --sku Basic --admin-enabled true | Out-Null
}
$loginServer = Invoke-Az acr show -n $Registry --query loginServer -o tsv

# --- Build image in ACR (no local Docker required) -------------------------
if ($SkipBuild) {
    Write-Host '==> Skipping build (-SkipBuild)' -ForegroundColor Yellow
}
else {
    Write-Host '==> Building image in ACR (a few minutes)' -ForegroundColor Cyan
    $prev = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
    az acr build --registry $Registry --image $imageTag --file "$context/Dockerfile" $context
    $buildExit = $LASTEXITCODE
    $ErrorActionPreference = $prev
    if ($buildExit -ne 0) { throw "ACR build failed with exit code $buildExit" }
}

# --- Container Apps environment -------------------------------------------
# Check the provisioning STATE, not just existence: a previous run may have left
# a Failed environment behind (e.g. regional AKSCapacityHeavyUsage), and creating
# an app against it fails with ManagedEnvironmentNotProvisioned.
$envState = Try-Az containerapp env show -n $Environment -g $ResourceGroup --query properties.provisioningState -o tsv

if ($envState -and $envState -ne 'Succeeded') {
    Write-Host "==> Environment '$Environment' is in state '$envState' - deleting it" -ForegroundColor Yellow
    Invoke-Az containerapp env delete -n $Environment -g $ResourceGroup --yes | Out-Null
    $envState = $null
}

if (-not $envState) {
    Write-Host "==> Creating Container Apps environment: $Environment ($EnvLocation)" -ForegroundColor Cyan
    Invoke-Az containerapp env create -n $Environment -g $ResourceGroup -l $EnvLocation | Out-Null

    $envState = Try-Az containerapp env show -n $Environment -g $ResourceGroup --query properties.provisioningState -o tsv
    if ($envState -ne 'Succeeded') {
        throw "Environment provisioning ended in state '$envState'. If this is a capacity error, rerun with a different -EnvLocation and -SkipBuild."
    }
}
else {
    Write-Host "==> Reusing Container Apps environment: $Environment" -ForegroundColor Cyan
}

# --- Container App ---------------------------------------------------------
$acrUser = Invoke-Az acr credential show -n $Registry --query username -o tsv
$acrPass = Invoke-Az acr credential show -n $Registry --query "passwords[0].value" -o tsv

$appExists = Try-Az containerapp show -n $appName -g $ResourceGroup --query name -o tsv

if ($appExists) {
    Write-Host "==> Updating Container App: $appName" -ForegroundColor Cyan
    Invoke-Az containerapp update `
        -n $appName -g $ResourceGroup `
        --image "$loginServer/$imageTag" `
        --cpu $Cpu --memory $Memory `
        --min-replicas $MinReplicas --max-replicas $MaxReplicas `
        --set-env-vars "INFERENCE_WORKERS=$InferenceWorkers" "TORCH_THREADS_PER_WORKER=$TorchThreadsPerWorker" `
        --scale-rule-name http-concurrency `
        --scale-rule-type http `
        --scale-rule-http-concurrency $ConcurrentRequests | Out-Null
}
else {
    Write-Host "==> Creating Container App: $appName" -ForegroundColor Cyan
    Invoke-Az containerapp create `
        -n $appName -g $ResourceGroup --environment $Environment `
        --image "$loginServer/$imageTag" `
        --registry-server $loginServer --registry-username $acrUser --registry-password $acrPass `
        --target-port 8000 --ingress external --transport auto `
        --cpu $Cpu --memory $Memory `
        --min-replicas $MinReplicas --max-replicas $MaxReplicas `
        --env-vars "INFERENCE_WORKERS=$InferenceWorkers" "TORCH_THREADS_PER_WORKER=$TorchThreadsPerWorker" `
        --scale-rule-name http-concurrency `
        --scale-rule-type http `
        --scale-rule-http-concurrency $ConcurrentRequests | Out-Null
}

$fqdn = Invoke-Az containerapp show -n $appName -g $ResourceGroup --query properties.configuration.ingress.fqdn -o tsv
$path = if ($Demo -eq 'yolo') { '/predict' } else { '/ocr' }

Write-Host ''
Write-Host '=================================================================' -ForegroundColor Green
Write-Host " App URL   : https://$fqdn"         -ForegroundColor Green
Write-Host " Docs      : https://$fqdn/docs"    -ForegroundColor Green
Write-Host " Health    : https://$fqdn/healthz" -ForegroundColor Green
Write-Host " Stats     : https://$fqdn/stats"   -ForegroundColor Green
Write-Host " Autoscale : $MinReplicas..$MaxReplicas replicas @ $ConcurrentRequests concurrent req/replica" -ForegroundColor Green
Write-Host " Capacity  : $InferenceWorkers parallel inferences/replica -> up to $($InferenceWorkers * $MaxReplicas) cluster-wide" -ForegroundColor Green
Write-Host '=================================================================' -ForegroundColor Green
Write-Host ''
Write-Host 'Benchmark (concurrency sweep):' -ForegroundColor Yellow
Write-Host "  cd ../loadtest; python bench.py --host https://$fqdn --endpoint $path"
Write-Host 'Load test:' -ForegroundColor Yellow
Write-Host "  cd ../loadtest; locust -f locustfile.py --host https://$fqdn --endpoint $path --headless -u 100 -r 10 -t 3m"
Write-Host 'Watch replicas:' -ForegroundColor Yellow
Write-Host "  az containerapp replica list -n $appName -g $ResourceGroup -o table"
