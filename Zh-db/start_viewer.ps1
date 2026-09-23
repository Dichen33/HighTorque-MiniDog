$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$port = 8765
$url = "http://127.0.0.1:$port"
$existing = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
if ($existing) {
  Write-Host "Zh-db is already running."
  Write-Host "Open: $url"
  exit 0
}

$env:PYTHONIOENCODING = "utf-8"
$python = (Get-Command python).Source
$logDir = Join-Path $PSScriptRoot "db-data"
$stdout = Join-Path $logDir "zhdb_server.out.log"
$stderr = Join-Path $logDir "zhdb_server.err.log"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
Remove-Item -LiteralPath $stdout -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath $stderr -Force -ErrorAction SilentlyContinue

$process = Start-Process `
  -WindowStyle Hidden `
  -FilePath $python `
  -ArgumentList @("-X", "utf8", "server.py", "--host", "127.0.0.1", "--port", "$port") `
  -WorkingDirectory $PSScriptRoot `
  -RedirectStandardOutput $stdout `
  -RedirectStandardError $stderr `
  -PassThru

Start-Sleep -Seconds 2
$listening = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
if ($listening) {
  Write-Host "Zh-db started. PID=$($process.Id)"
  Write-Host "Open: $url"
  exit 0
}

Write-Host "Zh-db failed to start. See logs:"
Write-Host "  $stdout"
Write-Host "  $stderr"
if (Test-Path $stderr) {
  Get-Content $stderr
}
exit 1
