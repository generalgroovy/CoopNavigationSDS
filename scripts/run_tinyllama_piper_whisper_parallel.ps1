param(
    [int]$MaxParallel = 2,
    [string]$ResultsDir = "results"
)

$ErrorActionPreference = "Stop"
if ($MaxParallel -lt 1) {
    throw "MaxParallel must be at least 1."
}
$python = (Get-Command python).Source
$root = Split-Path -Parent $PSScriptRoot
$jobs = Get-ChildItem -Path (Join-Path $root "jobs") `
    -Filter "tinyllama_piper_faster_whisper_parallel_*.job" |
    Sort-Object Name
$running = @()

foreach ($jobFile in $jobs) {
    while ($running.Count -ge $MaxParallel) {
        $finished = Wait-Job -Job $running -Any
        Receive-Job -Job $finished
        Remove-Job -Job $finished
        $running = @($running | Where-Object { $_.Id -ne $finished.Id })
    }
    $running += Start-Job -ScriptBlock {
        param($Python, $Root, $JobPath, $OutputRoot)
        Set-Location $Root
        & $Python -m coop_navigation_sds.batch `
            --job-file $JobPath `
            --results-dir $OutputRoot `
            --progress
        if ($LASTEXITCODE -ne 0) {
            throw "Batch shard failed with exit code ${LASTEXITCODE}: $JobPath"
        }
    } -ArgumentList $python, $root, $jobFile.FullName, $ResultsDir
}

if ($running.Count) {
    Wait-Job -Job $running | Out-Null
    foreach ($job in $running) {
        Receive-Job -Job $job
        Remove-Job -Job $job
    }
}
