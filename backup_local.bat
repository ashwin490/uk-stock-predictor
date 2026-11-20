@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
 echo Run setup_windows.bat first.
 pause
 exit /b 1
)
echo For a standalone backup, STOP the Streamlit app before continuing.
pause
.venv\Scripts\python.exe alpha_backup.py backup
pause
