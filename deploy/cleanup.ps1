<#
.SYNOPSIS
    Delete the demo resource group and everything in it.

.EXAMPLE
    ./cleanup.ps1 -ResourceGroup rg-aca-model-demo
#>
[CmdletBinding()]
param(
    [string]$ResourceGroup = 'rg-aca-model-demo'
)

$ErrorActionPreference = 'Stop'

Write-Host "This will DELETE resource group '$ResourceGroup' and all its resources." -ForegroundColor Yellow
$confirm = Read-Host "Type the resource group name to confirm"
if ($confirm -ne $ResourceGroup) {
    Write-Host 'Aborted.' -ForegroundColor Red
    exit 1
}

az group delete -n $ResourceGroup --yes --no-wait
Write-Host "Deletion started for '$ResourceGroup'." -ForegroundColor Green
