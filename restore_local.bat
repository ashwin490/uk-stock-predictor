@echo off
cd /d "%~dp0"
if "%~1"=="" (
 echo Drag a previous alpha_backup_*.zip onto this BAT file to restore it.
 echo STOP Streamlit first. Existing files will be kept in pre_restore.
 pause
 exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
 echo Run setup_windows.bat first.
 pause
 exit /b 1
)
echo STOP Streamlit before restoring. Existing learning files will be moved to pre_restore.
pause
.venv\Scripts\python.exe alpha_backup.py restore "%~1" --force
pause
