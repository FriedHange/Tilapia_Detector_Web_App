# Optional local setup. This script is never executed by the app or during installation.
param([switch]$Uninstall)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Split-Path $PSScriptRoot -Parent)).Path
$taskName = 'AquaTrack-' + (Split-Path $projectRoot -Leaf)
$marker = Join-Path $projectRoot 'private_data\startup-installed.json'
if ($Uninstall) {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
    if (Test-Path -LiteralPath $marker) { Remove-Item -LiteralPath $marker }
    exit
}
if (-not (Test-Path -LiteralPath (Join-Path $projectRoot 'venv\Scripts\python.exe'))) { throw 'Set up the Python environment first.' }
# Windows stores task credentials; passwords are never written into app files or logs.
$credential = Get-Credential -UserName ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -Message 'Windows account with access to the cameras and this application'
$taskScript = Join-Path $projectRoot 'scripts\run-production.ps1'
$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument ('-NoProfile -WindowStyle Hidden -File "' + $taskScript + '" -Boot') -WorkingDirectory $projectRoot
$trigger = New-ScheduledTaskTrigger -AtStartup
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($credential.Password)
try {
    $taskPassword = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer)
    Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -User $credential.UserName -Password $taskPassword -RunLevel Limited -Force | Out-Null
} finally {
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer)
    $taskPassword = $null
}
New-Item -ItemType Directory -Path (Split-Path $marker -Parent) -Force | Out-Null
@{task=$taskName; installed=(Get-Date).ToUniversalTime().ToString('o'); production_guard=$true} | ConvertTo-Json | Set-Content -LiteralPath $marker -Encoding UTF8
Write-Output 'Startup support configured. No task was started. At boot the runner exits unless Production Mode is on. Verify camera access under this Windows account before unattended use.'
