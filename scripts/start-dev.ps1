[CmdletBinding()]
param(
    [switch]$NoBrowser,
    [ValidateRange(10, 600)][int]$StartupTimeoutSeconds = 180
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$frontendRoot = Join-Path $projectRoot "frontend"
$logRoot = Join-Path $projectRoot '.workbuddy\dev-logs'
New-Item -ItemType Directory -Path $logRoot -Force | Out-Null

function Test-Endpoint {
    param([Parameter(Mandatory)][string]$Uri)

    try {
        $response = Invoke-WebRequest -UseBasicParsing -Uri $Uri -TimeoutSec 2
        if ($response.StatusCode -ne 200) { return $false }
        if ($Uri.EndsWith('/api/health')) {
            $health = $response.Content | ConvertFrom-Json
            return $health.status -eq 'ok' -and $health.service -eq 'lightgraphrag-workbench'
        }
        return $response.Content -match '<title>LightGraphRAG' -and $response.Content -match '/src/main.tsx'
    }
    catch {
        return $false
    }
}

function Test-ListeningPort {
    param([Parameter(Mandatory)][int]$Port)

    return [bool](
        Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue
    )
}

function ConvertTo-EncodedCommand {
    param([Parameter(Mandatory)][string]$Command)

    return [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($Command))
}

function ConvertTo-PowerShellLiteral {
    param([Parameter(Mandatory)][string]$Value)

    return "'" + $Value.Replace("'", "''") + "'"
}

function Assert-ChildRunning {
    param($Process, [string]$Name, [string]$Logs)
    if ($null -ne $Process -and $Process.HasExited) {
        throw "$Name exited before startup completed (exit code $($Process.ExitCode)). Check logs under $Logs."
    }
}

$uvCommand = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uvCommand) {
    throw "uv was not found. Install uv and add it to PATH."
}

$npmCommand = Get-Command npm.cmd -ErrorAction SilentlyContinue
if (-not $npmCommand) {
    throw "npm.cmd was not found. Install Node.js 20+ and add it to PATH."
}

$childShell = Get-Command pwsh.exe -ErrorAction SilentlyContinue
if (-not $childShell) {
    $childShell = Get-Command powershell.exe -ErrorAction Stop
}

$backendUri = "http://127.0.0.1:8101/api/health"
$frontendUri = "http://127.0.0.1:5173/"
$backendReady = Test-Endpoint -Uri $backendUri
$frontendReady = Test-Endpoint -Uri $frontendUri

if (-not $backendReady -and (Test-ListeningPort -Port 8101)) {
    throw "Port 8101 is already used by another application."
}
if (-not $frontendReady -and (Test-ListeningPort -Port 5173)) {
    throw "Port 5173 is already used by another application."
}

$projectLiteral = ConvertTo-PowerShellLiteral -Value $projectRoot
$frontendLiteral = ConvertTo-PowerShellLiteral -Value $frontendRoot
$uvLiteral = ConvertTo-PowerShellLiteral -Value $uvCommand.Source
$npmLiteral = ConvertTo-PowerShellLiteral -Value $npmCommand.Source
$backendProcess = $null
$frontendProcess = $null

if (-not $backendReady) {
    $backendScript = @"
`$Host.UI.RawUI.WindowTitle = 'LightGraphRAG API :8101'
`$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $projectLiteral
& $uvLiteral run python -m src.app.cli server --no-reload
exit `$LASTEXITCODE
"@
    $backendProcess = Start-Process -PassThru -FilePath $childShell.Source -ArgumentList @(
        "-NoProfile",
        "-EncodedCommand",
        (ConvertTo-EncodedCommand -Command $backendScript)
    ) -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logRoot 'backend.log') -RedirectStandardError (Join-Path $logRoot 'backend-error.log')
    Write-Host "Starting backend..." -ForegroundColor Cyan
}
else {
    Write-Host "Backend is already running; skipping duplicate startup." -ForegroundColor DarkGray
}

if (-not $frontendReady) {
    $frontendScript = @"
`$Host.UI.RawUI.WindowTitle = 'LightGraphRAG Web :5173'
`$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $frontendLiteral
if (-not (Test-Path -LiteralPath 'node_modules')) {
    & $npmLiteral ci
    if (`$LASTEXITCODE -ne 0) { throw 'npm ci failed' }
}
& $npmLiteral run dev -- --host 127.0.0.1 --port 5173 --strictPort
exit `$LASTEXITCODE
"@
    $frontendProcess = Start-Process -PassThru -FilePath $childShell.Source -ArgumentList @(
        "-NoProfile",
        "-EncodedCommand",
        (ConvertTo-EncodedCommand -Command $frontendScript)
    ) -WorkingDirectory $frontendRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logRoot 'frontend.log') -RedirectStandardError (Join-Path $logRoot 'frontend-error.log')
    Write-Host "Starting frontend..." -ForegroundColor Cyan
}
else {
    Write-Host "Frontend is already running; skipping duplicate startup." -ForegroundColor DarkGray
}

$deadline = (Get-Date).AddSeconds($StartupTimeoutSeconds)
do {
    Assert-ChildRunning -Process $backendProcess -Name 'Backend' -Logs $logRoot
    Assert-ChildRunning -Process $frontendProcess -Name 'Frontend' -Logs $logRoot
    $backendReady = Test-Endpoint -Uri $backendUri
    $frontendReady = Test-Endpoint -Uri $frontendUri
    if ($backendReady -and $frontendReady) {
        break
    }
    Start-Sleep -Milliseconds 500
} while ((Get-Date) -lt $deadline)

if (-not $backendReady -or -not $frontendReady) {
    throw "Startup timed out. Check logs under $logRoot. Services may still be starting."
}

Write-Host "Backend and frontend are ready: $frontendUri" -ForegroundColor Green
if (-not $NoBrowser) {
    Start-Process $frontendUri
}
