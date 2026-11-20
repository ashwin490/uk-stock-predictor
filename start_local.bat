@echo off
cd /d "%~dp0"
set ALPHA_STORAGE_MODE=local
if not exist ".venv\Scripts\python.exe" (
 echo Run setup_windows.bat first.
 pause
 exit /b 1
)
echo ALPHA-LSE is starting at http://127.0.0.1:8501
.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true
if errorlevel 1 pause
