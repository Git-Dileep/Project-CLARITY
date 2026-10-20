@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Virtual environment not found. Create .venv and install requirements.txt first.
  pause
  exit /b 1
)
echo [1/4] Auditing both datasets...
.venv\Scripts\python.exe -m src.audit_datasets --network-dir input_data\network --iot-dir input_data\iot_devices
if errorlevel 1 goto fail
echo [2/4] Training traditional RF+XGBoost baseline and exporting shared sample...
.venv\Scripts\python.exe -m src.train_traditional --network-dir input_data\network --max-per-class 20000 --test-size 0.2
if errorlevel 1 goto fail
echo [3/4] Training CNN/LSTM/Transformer feasibility models...
.venv\Scripts\python.exe -m src.train_dl --network-dir input_data\network --max-per-class 3000 --epochs 3
if errorlevel 1 goto fail
echo [4/4] Exporting 20-alert pilot and SHAP explanations...
.venv\Scripts\python.exe -m src.export_pilot --pilot-size 20 --min-classes 5
if errorlevel 1 goto fail
echo Workflow finished. Inspect reports\ and reports\dl\; review all warnings before claiming final results.
pause
exit /b 0
:fail
echo Workflow stopped because a step failed. Read the error above before continuing.
pause
exit /b 1
