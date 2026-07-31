<#
.SYNOPSIS
    Scale every demo app to zero replicas. Run this after testing.

.DESCRIPTION
    The tests themselves cost very little; the money is lost by leaving apps
    warm. With min-replicas 1 the three demos cost roughly:

        yolo-demo       2 vCPU / 4 GiB     ~$5 / day
        paddle-demo-cpu 4 vCPU / 8 GiB     ~$10 / day
        paddle-demo-gpu T4 serverless GPU  ~$8 / day   <- billed whenever allocated

    Scaling to zero stops all compute charges. The apps stay deployed and cold
    start on the next request.

.EXAMPLE
    ./scaledown.ps1
    ./scaledown.ps1 -ResourceGroup rg-aca-model-demo
#>
[CmdletBinding()]
param(
    [string]$ResourceGroup = 'rg-aca-model-demo',
    [switch]$ShowOnly
)

$ErrorActionPreference = 'Stop'
$env:AZURE_CORE_ONLY_SHOW_ERRORS = 'true'

$apps = az containerapp list -g $ResourceGroup -o json 2>$null | ConvertFrom-Json
if (-not $apps) {
    Write-Host "No container apps found in $ResourceGroup" -ForegroundColor Yellow
    exit 0
}

Write-Host "`nCurrent state" -ForegroundColor Cyan
Write-Host ("-" * 62)
foreach ($a in $apps) {
    $scale = $a.properties.template.scale
    $reps = az containerapp replica list -n $a.name -g $ResourceGroup -o json 2>$null | ConvertFrom-Json
    Write-Host ("  {0,-20} min={1} max={2}  running={3}  profile={4}" -f `
        $a.name, $scale.minReplicas, $scale.maxReplicas, $reps.Count, `
        ($a.properties.workloadProfileName ?? 'Consumption'))
}

if ($ShowOnly) { exit 0 }

Write-Host "`nScaling all apps to min-replicas 0" -ForegroundColor Cyan
foreach ($a in $apps) {
    if ($a.properties.template.scale.minReplicas -eq 0) {
        Write-Host "  $($a.name): already 0" -ForegroundColor DarkGray
        continue
    }
    Write-Host "  $($a.name): -> 0" -ForegroundColor Yellow
    az containerapp update -n $a.name -g $ResourceGroup --min-replicas 0 -o none 2>$null
}

Write-Host "`nDone. Apps remain deployed and will cold start on the next request." -ForegroundColor Green
Write-Host "To remove everything entirely, run cleanup.ps1.`n" -ForegroundColor Green
