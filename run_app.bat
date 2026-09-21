@echo off
:: ============================================================================
:: run_app.bat — Tilapia Fingerling Counter Web Application Launcher
:: ============================================================================
:: This script:
::   1. Checks for Python installation
::   2. Creates a virtual environment if it doesn't exist
::   3. Activates the venv and installs dependencies
::   4. Starts the FastAPI server via Uvicorn
::   5. Opens the dashboard in the default browser
:: ============================================================================

setlocal EnableDelayedExpansion

:: Change to the script's own directory
cd /d "%~dp0"

echo.
echo  ==========================================================
echo   Tilapia Fingerling Counter  ^|  Aquaculture Vision App
echo  ==========================================================
echo.

:: ── 1. Check Python (Prefer 3.12 for NVIDIA CUDA GPU support) ─────────────────
set PYTHON_CMD=
py -3.12 --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON_CMD=py -3.12
) else (
    where python >nul 2>&1
    if not errorlevel 1 (
        set PYTHON_CMD=python
    ) else (
        echo  [ERROR] Python not found in PATH.
        echo          Please install Python 3.12 from https://python.org
        pause
        exit /b 1
    )
)

for /f "tokens=*" %%v in ('!PYTHON_CMD! --version 2^>^&1') do set PY_VER=%%v
echo  [OK] Found %PY_VER% (using !PYTHON_CMD!)

:: ── 2. Create virtual environment if needed ──────────────────────────────────
if not exist "venv\" (
    echo.
    echo  [SETUP] Creating virtual environment with !PYTHON_CMD!...
    !PYTHON_CMD! -m venv venv
    if errorlevel 1 (
        echo  [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo  [OK] Virtual environment created.
)

:: ── 3. Activate venv ─────────────────────────────────────────────────────────
echo.
echo  [SETUP] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo  [ERROR] Failed to activate virtual environment.
    pause
    exit /b 1
)
echo  [OK] Virtual environment active.

:: ── 4. Install / upgrade dependencies ───────────────────────────────────────
echo.
echo  [SETUP] Checking and installing dependencies...
echo          (This may take a few minutes on first run)
echo.

:: Check if PyTorch with CUDA is installed
python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
if errorlevel 1 (
    echo  [SETUP] Checking NVIDIA GPU for CUDA acceleration...
    nvidia-smi >nul 2>&1
    if not errorlevel 1 (
        echo  [INFO] NVIDIA GPU detected! Installing PyTorch with CUDA 12.4 support...
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    )
)

pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo.
    echo  [WARN] Some packages may not have installed correctly.
    echo         Check the output above for errors.
)
echo.
echo  [OK] Dependencies ready.

:: ── 5. Create models directory if missing ────────────────────────────────────
if not exist "models\" (
    mkdir models
    echo  [INFO] Created models\ directory.
    echo         Place your .pt model files (yolov8n.pt, yolov9c.pt, yolov10n.pt)
    echo         in the models\ folder before running.
)

:: ── 6. Open browser after a short delay ─────────────────────────────────────
echo.
echo  [INFO] Starting server at http://localhost:8000
echo  [INFO] Dashboard will open automatically in your browser.
echo  [INFO] Press CTRL+C to stop the server.
echo.

:: Open browser after 2-second delay (runs in background)
start "" cmd /c "timeout /t 2 >nul && start http://localhost:8000"

:: ── 7. Launch FastAPI server ─────────────────────────────────────────────────
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload --log-level info

:: If the server exits cleanly
echo.
echo  [INFO] Server stopped.
pause
