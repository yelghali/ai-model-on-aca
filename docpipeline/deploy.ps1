<#
.SYNOPSIS
    Package the document pipeline and deploy the API app, worker app, or both.

.DESCRIPTION
    Both Function Apps use the same role-gated package. Terraform sets
    FUNCTION_ROLE=api or FUNCTION_ROLE=worker so each host indexes only its own
    functions. For migration from the former combined app, API is deployed
    first so it stops consuming task-hub messages before the worker starts.

.EXAMPLE
    .\deploy.ps1 -Target both
    .\deploy.ps1 -Target api
    .\deploy.ps1 -Target worker
#>
[CmdletBinding()]
param(
    [ValidateSet('api', 'worker', 'both')]
    [string]$Target = 'both',
    [string]$TerraformDir = (Join-Path $PSScriptRoot 'infra')
)

$ErrorActionPreference = 'Stop'
$package = Join-Path $env:TEMP "docpipeline-functions-$PID.zip"

function Invoke-TerraformOutput {
    param([Parameter(Mandatory)][string]$Name)

    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $value = & terraform "-chdir=$TerraformDir" output -raw $Name 2>&1
        if ($LASTEXITCODE -ne 0) {
            throw "terraform output '$Name' failed:`n$value"
        }
        return ($value -join "`n").Trim()
    }
    finally {
        $ErrorActionPreference = $previous
    }
}

function Deploy-FunctionApp {
    param(
        [Parameter(Mandatory)][string]$Role,
        [Parameter(Mandatory)][string]$AppName,
        [Parameter(Mandatory)][string]$ResourceGroup
    )

    Write-Host "==> Deploying $Role package to $AppName" -ForegroundColor Cyan
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $output = & az functionapp deployment source config-zip `
            --resource-group $ResourceGroup `
            --name $AppName `
            --src $package `
            --only-show-errors 2>&1
        if ($LASTEXITCODE -ne 0) {
            throw "Deployment to '$AppName' failed:`n$output"
        }
    }
    finally {
        $ErrorActionPreference = $previous
    }
}

if (-not (Test-Path $TerraformDir)) {
    throw "Terraform directory not found: $TerraformDir"
}

$runtimeFiles = @(
    (Join-Path $PSScriptRoot 'function_app.py'),
    (Join-Path $PSScriptRoot 'host.json'),
    (Join-Path $PSScriptRoot 'requirements.txt'),
    (Join-Path $PSScriptRoot 'shared'),
    (Join-Path $PSScriptRoot 'ui')
)

foreach ($path in $runtimeFiles) {
    if (-not (Test-Path $path)) {
        throw "Runtime path not found: $path"
    }
}

try {
    Write-Host "==> Packaging Function runtime files" -ForegroundColor Cyan
    Compress-Archive -Path $runtimeFiles -DestinationPath $package -Force

    $resourceGroup = Invoke-TerraformOutput -Name 'resource_group_name'
    $apiApp = Invoke-TerraformOutput -Name 'api_function_app_name'
    $workerApp = Invoke-TerraformOutput -Name 'worker_function_app_name'

    # This order is intentional for the one-time split migration. There can be
    # a short processing gap, but never two different worker apps competing for
    # the same task hub. Pending Durable messages remain safely in storage.
    if ($Target -in @('api', 'both')) {
        Deploy-FunctionApp -Role 'api' -AppName $apiApp -ResourceGroup $resourceGroup
    }
    if ($Target -in @('worker', 'both')) {
        Deploy-FunctionApp -Role 'worker' -AppName $workerApp -ResourceGroup $resourceGroup
    }

    Write-Host "==> Deployment complete ($Target)" -ForegroundColor Green
    Write-Host "    UI: $(Invoke-TerraformOutput -Name 'ui_url')" -ForegroundColor Green
}
finally {
    Remove-Item $package -Force -ErrorAction SilentlyContinue
}