param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ScriptArgs
)

Set-StrictMode -Version Latest

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$target = Join-Path $scriptDir 'SEM_cudaopenvinofinetuning.py'

if (-not (Test-Path $target)) {
    Write-Error "Target script not found: $target"
    exit 1
}

$venvPython = Join-Path $scriptDir '.venv\Scripts\python.exe'
if (Test-Path $venvPython) {
    & $venvPython $target @ScriptArgs
    exit $LASTEXITCODE
}

$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if ($pythonCmd) {
    & $pythonCmd.Source $target @ScriptArgs
    exit $LASTEXITCODE
}

$pyCmd = Get-Command py -ErrorAction SilentlyContinue
if ($pyCmd) {
    & $pyCmd.Source -3 $target @ScriptArgs
    exit $LASTEXITCODE
}

Write-Error 'Python launcher not found. Install Python or add it to PATH.'
exit 1
