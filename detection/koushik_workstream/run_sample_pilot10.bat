@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"
echo ====================================================================
echo [CLARITY Detection] Reproducible 10-Flow Sample Pilot Runner and Verifier
echo ====================================================================

set "CONFIG_FILE=config\run_sample_pilot10.json"

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
    echo [ERROR] Python not found. Please activate the virtual environment.
    exit /b 1
)

echo Using Python: %PYTHON_EXE%
echo Using Config: %CONFIG_FILE%
echo.

echo [Step 1/4] Running Pipeline End-to-End...
"%PYTHON_EXE%" -m src.run_all --config "%CONFIG_FILE%"
if errorlevel 1 (
    echo [ERROR] Pipeline execution failed.
    exit /b %errorlevel%
)

echo.
echo [Step 2/4] Validating Draft 2020-12 Schema and Semantic Invariants...
"%PYTHON_EXE%" ..\..\scripts\validate_schemas.py --config "%CONFIG_FILE%"
if errorlevel 1 (
    echo [ERROR] Schema validation failed.
    exit /b %errorlevel%
)

echo.
echo [Step 3/4] Running Data Leakage Audit...
"%PYTHON_EXE%" -m src.leakage_audit --config "%CONFIG_FILE%"
if errorlevel 1 (
    echo [ERROR] Leakage audit failed.
    exit /b %errorlevel%
)

echo.
echo [Step 4/4] Executing Automated Pytest Invariant Checks...
"%PYTHON_EXE%" -m pytest tests\test_pilot10_run.py -v
if errorlevel 1 (
    echo [ERROR] Pytest verification failed.
    exit /b %errorlevel%
)

echo.
echo ====================================================================
echo [SUCCESS] 10-Flow Sample Pilot Run and All Verifications PASSED!
echo ====================================================================
exit /b 0
