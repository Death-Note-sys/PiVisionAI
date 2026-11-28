@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "NO_PAUSE=0"
if /i "%~1"=="--nopause" set "NO_PAUSE=1"

echo ==================================================
echo   PiVisionAI - First-Time Client Setup
echo ==================================================
echo.

REM 1. Locate Python 3.10+
set "PY_CMD="
set "PY_ARGS="

where python >nul 2>&1
if not errorlevel 1 (
    python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
    if not errorlevel 1 (
        set "PY_CMD=python"
    )
)

if not defined PY_CMD (
    where py >nul 2>&1
    if not errorlevel 1 (
        py -3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
        if not errorlevel 1 (
            set "PY_CMD=py"
            set "PY_ARGS=-3"
        )
    )
)

if not defined PY_CMD (
    echo [ERROR] Python 3.10 or newer was not found on PATH.
    echo Please install Python 3.10-3.14 from https://www.python.org/downloads/
    echo and make sure "Add Python to PATH" is checked during installation.
    goto :fail
)

echo [1/5] Using Python:
"%PY_CMD%" %PY_ARGS% --version

REM 2. Create virtual environment if missing
if not exist ".venv\Scripts\python.exe" (
    echo.
    echo [2/5] Creating virtual environment in .venv ...
    "%PY_CMD%" %PY_ARGS% -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create .venv.
        goto :fail
    )
) else (
    echo.
    echo [2/5] Existing .venv found, reusing it.
)

set "VENV_PY=.venv\Scripts\python.exe"

REM 3. Install dependencies
echo.
echo [3/5] Installing Python dependencies from requirements.txt ...
"%VENV_PY%" -m pip install --upgrade pip
if errorlevel 1 (
    echo [WARN] Could not upgrade pip; continuing with installed pip.
)
"%VENV_PY%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies. Check internet connectivity and retry.
    goto :fail
)

REM 4. Pre-download AI weights so first inspection does not stall
echo.
echo [4/5] Checking / pre-downloading AI model weights (YOLO, SAM, EasyOCR) ...
if not exist "models" mkdir "models"
call :prefetch_models

REM 5. Run diagnostics & create Desktop shortcut
echo.
echo [5/5] Running system diagnostics and creating Desktop shortcut ...
"%VENV_PY%" cli.py doctor
if errorlevel 1 (
    echo [WARN] System diagnostics reported a warning above.
)

powershell -NoProfile -ExecutionPolicy Bypass -Command "try { $d = [Environment]::GetFolderPath('Desktop'); if ($d) { $s = (New-Object -ComObject WScript.Shell).CreateShortcut((Join-Path $d 'PiVisionAI.lnk')); $s.TargetPath = '%~dp0Start_PiVisionAI.bat'; $s.WorkingDirectory = '%~dp0'; $s.Description = 'Launch PiVisionAI Inspection Platform'; $s.Save(); Write-Host ('Desktop shortcut created: ' + (Join-Path $d 'PiVisionAI.lnk')) } } catch { Write-Host ('[WARN] Could not create Desktop shortcut: ' + $_.Exception.Message) }"

echo.
echo ==================================================
echo   Setup complete!
echo   Double-click Start_PiVisionAI.bat (or the
echo   "PiVisionAI" Desktop shortcut) to launch.
echo ==================================================
if "%NO_PAUSE%"=="0" pause
exit /b 0

:fail
echo.
echo Setup did not complete. See error details above.
if "%NO_PAUSE%"=="0" pause
exit /b 1

:prefetch_models
if exist "yolo11n.pt" (
    echo  - YOLO11n already present.
) else if exist "ai_models\object_detection\yolo11n\model.pt" (
    echo  - YOLO11n already present.
) else (
    echo  - Downloading YOLO11n weights...
    "%VENV_PY%" -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
    if errorlevel 1 echo [WARN] YOLO11n download failed; will retry automatically on first Object Detection run.
)

if exist "models\mobile_sam.pt" (
    echo  - MobileSAM already present.
) else if exist "mobile_sam.pt" (
    echo  - MobileSAM already present.
) else (
    echo  - Downloading MobileSAM weights...
    "%VENV_PY%" -c "import os, shutil; from ultralytics import SAM; SAM('mobile_sam.pt'); os.makedirs('models', exist_ok=True); os.path.isfile('mobile_sam.pt') and shutil.move('mobile_sam.pt', os.path.join('models', 'mobile_sam.pt'))"
    if errorlevel 1 echo [WARN] MobileSAM download failed; will retry automatically on first triggered Target Extraction run.
)

echo  - Verifying EasyOCR English weights...
"%VENV_PY%" -c "import easyocr; easyocr.Reader(['en'], gpu=False, verbose=False)"
if errorlevel 1 echo [WARN] EasyOCR download failed; will retry automatically on first OCR run.
goto :eof

