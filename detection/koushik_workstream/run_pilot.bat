@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Virtual environment not found. Create it and install requirements first.
  pause
  exit /b 1
)
.venv\Scripts\python.exe -m src.export_pilot
if errorlevel 1 echo Pilot export failed. Read the error above and check prerequisite outputs.
pause
