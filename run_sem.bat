@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "TARGET=%SCRIPT_DIR%SEM_cudaopenvinofinetuning.py"

if not exist "%TARGET%" (
  echo Target script not found: "%TARGET%"
  exit /b 1
)

where python >nul 2>&1
if %ERRORLEVEL%==0 (
  python "%TARGET%" %*
  exit /b %ERRORLEVEL%
)

where py >nul 2>&1
if %ERRORLEVEL%==0 (
  py -3 "%TARGET%" %*
  exit /b %ERRORLEVEL%
)

echo Python launcher not found. Install Python or add it to PATH.
exit /b 1
