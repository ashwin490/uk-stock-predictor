@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
 echo Set up the project first with setup_windows.bat
 pause
 exit /b 1
)
echo Please close the Streamlit app first to let the database open safely.
.venv\Scripts\python.exe health_check.py --verify-backup
pause
