@echo off
setlocal

cd /d "%~dp0backend"

if not exist "venv\Scripts\python.exe" (
    echo [1/3] venv not found, creating ...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create venv. Check if Python is installed.
        pause
        exit /b 1
    )
    echo [2/3] Installing dependencies ...
    call "venv\Scripts\activate.bat"
    pip install --upgrade pip --quiet
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies. Check network or requirements.txt.
        pause
        exit /b 1
    )
    echo [3/3] Dependencies installed. Starting server ...
) else (
    echo [1/2] venv found, activating ...
    call "venv\Scripts\activate.bat"
    echo [2/2] Starting server ...
)

python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload --timeout-keep-alive 60

pause