@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"
echo ====================================================================
echo [CLARITY Detection] Master End-to-End Pipeline Runner
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

"%PYTHON_EXE%" -m src.run_all --config "%CONFIG_FILE%"
if errorlevel 1 (
    echo.
    echo [ERROR] Pipeline stopped due to failure.
    exit /b %errorlevel%
)

echo.
echo [SUCCESS] Complete workflow executed successfully.
exit /b 0
