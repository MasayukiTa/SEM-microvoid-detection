@echo off
setlocal EnableExtensions

set "SCRIPT_DIR=%~dp0"
set "TARGET=%SCRIPT_DIR%SEM_cudaopenvinofinetuning.py"
set "VENV_PY=%SCRIPT_DIR%.venv\Scripts\python.exe"

if not exist "%TARGET%" (
  echo Target script not found: "%TARGET%"
  exit /b 1
)

if exist "%VENV_PY%" (
  "%VENV_PY%" "%TARGET%" %*
  exit /b %ERRORLEVEL%
)

where python >nul 2>&1
if %ERRORLEVEL%==0 (
  goto :run_python
)

where py >nul 2>&1
if %ERRORLEVEL%==0 (
  goto :run_py
)

echo Python launcher not found. Install Python or add it to PATH.
exit /b 1

:run_python
python "%TARGET%" %*
exit /b %ERRORLEVEL%

:run_py
py -3 "%TARGET%" %*
exit /b %ERRORLEVEL%
