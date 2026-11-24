@echo off
cd /d "%~dp0"
set PYTHON_EXE=%~dp0\.venv\Scripts\python.exe
set APP_FILE=%~dp0\app.py

for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000') do (
    if not "%%a"=="" taskkill /F /PID %%a >nul 2>&1
)

start "SDR Backend" cmd /k ""%PYTHON_EXE%" "%APP_FILE%""

echo Backend started.
echo Open http://127.0.0.1:8000 in your browser when ready.
exit
