param(
    [string]$ProjectRoot = (Split-Path -Parent $MyInvocation.MyCommand.Path)
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$envPath = Join-Path $ProjectRoot '.env'
if (!(Test-Path $envPath)) {
    Write-Host "Missing .env at $envPath" -ForegroundColor Red
    Write-Host "Create it with: DHAN_TOKEN=YOUR_TOKEN" -ForegroundColor Yellow
    exit 1
}

# Minimal .env loader: KEY=VALUE lines, ignore blanks and comments.
Get-Content $envPath | ForEach-Object {
    $line = $_.Trim()
    if ($line -eq '' -or $line.StartsWith('#')) { return }
    $parts = $line.Split('=', 2)
    if ($parts.Length -ne 2) { return }
    $key = $parts[0].Trim()
    $val = $parts[1].Trim().Trim('"')
    if ($key) { Set-Item -Path ("Env:" + $key) -Value $val }
}

$python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
if (!(Test-Path $python)) {
    Write-Host "Virtual env python not found at $python" -ForegroundColor Red
    exit 1
}

# Start Streamlit in a separate process, then open Chrome.
$streamlitArgs = "-m streamlit run indexemaUserDefined.py"
Start-Process -FilePath $python -ArgumentList $streamlitArgs
Start-Sleep -Seconds 2
Start-Process chrome "http://localhost:8501"
