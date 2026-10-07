param([switch]$Boot)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $projectRoot
$pythonPath = Join-Path $projectRoot 'venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Install the application dependencies before running production.' }
if ($Boot) { & $pythonPath (Join-Path $projectRoot 'production_launcher.py') --boot }
else { & $pythonPath (Join-Path $projectRoot 'production_launcher.py') }
exit $LASTEXITCODE
