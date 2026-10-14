@echo off
rem Start the AquaWatch dashboard: http://localhost:8501  (close this window to stop it).
rem Uses .venv in this folder (or the one next to it) when there is one, else the "python" on PATH.
rem Runs offline and on 2 CPU threads at below-normal priority (cpu_limits.py); set AQUAWATCH_THREADS=4 for more speed.
cd /d "%~dp0"
set "YOLO_OFFLINE=true"
set "HF_HUB_OFFLINE=1"
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"
if not defined AQUAWATCH_THREADS set "AQUAWATCH_THREADS=2"
set "PY=python"
if exist "%~dp0..\.venv\Scripts\python.exe" set "PY=%~dp0..\.venv\Scripts\python.exe"
if exist "%~dp0.venv\Scripts\python.exe" set "PY=%~dp0.venv\Scripts\python.exe"
"%PY%" -m streamlit run app.py --server.port 8501 --server.address 127.0.0.1 --server.headless true
