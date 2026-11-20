@echo off
cd /d "%~dp0"
echo Setting up ALPHA-LSE for Windows (Python 3.11 required)...
py -3.11 -m venv .venv
if errorlevel 1 (
 echo Python 3.11 was not found. Install Python 3.11 and rerun setup_windows.bat.
 pause
 exit /b 1
)
.venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 goto error
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto error
if not exist ".streamlit\secrets.toml" (
 echo Creating your local login...
 .venv\Scripts\python.exe setup_auth.py
 if errorlevel 1 goto error
)
echo Finished. Copy your original feature_engine.py and models\ensemble_ranker.joblib into this folder if missing.
pause
exit /b 0
:error
echo Setup did not finish. Check the messages above.
pause
exit /b 1
