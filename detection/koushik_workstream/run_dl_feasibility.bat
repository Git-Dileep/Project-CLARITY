@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Virtual environment not found. Run: py -m venv .venv
  echo Then activate it and run: pip install -r requirements.txt
  pause
  exit /b 1
)
.venv\Scripts\python.exe -m src.train_dl --network-dir input_data\network --max-per-class 3000 --epochs 3
pause
