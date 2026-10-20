@echo off
cd /d "%~dp0"
echo ======================================================
echo KOU IIoT Detection - one-time environment setup
echo ======================================================
where py >nul 2>nul
if errorlevel 1 (
  echo Python Launcher not found. Install Python 3.10 or 3.11 64-bit from python.org.
  echo During installation, enable the Python Launcher and Add Python to PATH.
  pause
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" py -3.11 -m venv .venv
if errorlevel 1 (
  echo Python 3.11 was not found. Trying the default Python installation...
  py -m venv .venv
)
if not exist ".venv\Scripts\python.exe" (
  echo Could not create the virtual environment. Check your Python installation.
  pause
  exit /b 1
)
.venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 goto fail
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto fail
echo Setup completed. Next, put your CSV files in input_data\network and input_data\iot_devices, then run run_audit.bat.
pause
exit /b 0
:fail
echo Dependency installation failed. Check the error above and your internet connection.
pause
exit /b 1
