@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"
echo ====================================================================
echo [CLARITY Detection] Stratified 20-Alert Pilot Export & Schema Validator
echo ====================================================================

set "CONFIG_FILE=%~1"
if "%CONFIG_FILE%"=="" set "CONFIG_FILE=config\reference.json"

set "PYTHON_EXE="
if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=.venv\Scripts\python.exe"
) else if exist "..\..\.venv\Scripts\python.exe" (
    set "PYTHON_EXE=..\..\.venv\Scripts\python.exe"
) else (
    where python >nul 2>nul
    if not errorlevel 1 set "PYTHON_EXE=python"
)

if "%PYTHON_EXE%"=="" (
    echo [ERROR] Python not found. Please create a virtual environment:
    echo   py -m venv .venv
    echo   .venv\Scripts\activate
    echo   pip install -r requirements.txt
    exit /b 1
)

echo Using Python: %PYTHON_EXE%
echo Using Config: %CONFIG_FILE%
echo.

"%PYTHON_EXE%" -m src.export_pilot --config "%CONFIG_FILE%"
if errorlevel 1 (
    echo [ERROR] Pilot export and validation failed with exit code %errorlevel%.
    exit /b %errorlevel%
)

echo.
echo [SUCCESS] Pilot export and schema validation passed successfully.
exit /b 0
