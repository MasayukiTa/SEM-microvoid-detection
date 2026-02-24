param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ScriptArgs
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$target = Join-Path $scriptDir 'SEM_cudaopenvinofinetuning.py'

if (-not (Test-Path $target)) {
    Write-Error "Target script not found: $target"
    exit 1
}

if (Get-Command python -ErrorAction SilentlyContinue) {
    & python $target @ScriptArgs
    exit $LASTEXITCODE
}

if (Get-Command py -ErrorAction SilentlyContinue) {
    & py -3 $target @ScriptArgs
    exit $LASTEXITCODE
}

Write-Error 'Python launcher not found. Install Python or add it to PATH.'
exit 1
