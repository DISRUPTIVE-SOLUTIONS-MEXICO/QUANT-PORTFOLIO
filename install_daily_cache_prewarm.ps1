param(
    [string]$TaskName = "QuantPortfolioKaizenDailyCachePrewarm",
    [string]$RunTime = "08:30",
    [switch]$IncludeGeopolitical
)

$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = (Get-Command python).Source
$Script = Join-Path $ProjectDir "prewarm_quant_cache.py"

if (!(Test-Path -LiteralPath $Script)) {
    throw "Missing prewarm script: $Script"
}

$Args = "`"$Script`" --period 5y --ttl-hours 24 --country `"United States`""
if ($IncludeGeopolitical) {
    $Args = "$Args --include-geopolitical"
}

$Action = New-ScheduledTaskAction -Execute $Python -Argument $Args -WorkingDirectory $ProjectDir
$Trigger = New-ScheduledTaskTrigger -Daily -At $RunTime
$Settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 2)

Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings -Description "Daily zero-cost public data cache prewarm for Quant Portfolio-Kaizen before market open." -Force | Out-Null

Write-Host "Installed scheduled task '$TaskName' at $RunTime."
Write-Host "Project: $ProjectDir"
Write-Host "Command: $Python $Args"
