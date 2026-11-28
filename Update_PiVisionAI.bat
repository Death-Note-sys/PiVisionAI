@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "NO_PAUSE=0"
if /i "%~1"=="--nopause" set "NO_PAUSE=1"

echo ==================================================
echo   PiVisionAI - Update from GitHub
echo ==================================================
echo.

if not exist ".git" (
    echo [ERROR] This folder is not a Git repository; .git folder not found.
    echo Updates via Update_PiVisionAI.bat require a GitHub clone.
    goto :fail
)

where git >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Git is not installed or not on PATH.
    goto :fail
)

REM Warn if PiVisionAI server is currently running (prevents Windows file locks on update)
set "CHECK_PY="
if exist ".venv\Scripts\python.exe" set "CHECK_PY=.venv\Scripts\python.exe"
if not defined CHECK_PY (
    where python >nul 2>&1
    if not errorlevel 1 set "CHECK_PY=python"
)

if defined CHECK_PY (
    "%CHECK_PY%" -c "import socket, sys; s = socket.socket(); s.settimeout(0.5); sys.exit(0 if s.connect_ex(('127.0.0.1', 5000)) == 0 else 1)" >nul 2>&1
    if not errorlevel 1 (
        echo [ERROR] PiVisionAI appears to be running on port 5000.
        echo Please close the PiVisionAI window before updating, then run Update_PiVisionAI.bat again.
        goto :fail
    )
)

echo Note: Your client settings, databases, taught references, and downloaded models
echo in data/ and models/ are untouched by updates.
echo.

echo [1/2] Pulling latest updates from GitHub ...
git pull --ff-only
if errorlevel 1 (
    echo.
    echo [ERROR] Could not fast-forward cleanly.
    echo If tracked files were edited locally or the remote branch changed, review "git status".
    goto :fail
)

echo.
if exist ".venv\Scripts\python.exe" (
    echo [2/2] Checking Python dependencies ...
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to update dependencies.
        goto :fail
    )
) else (
    echo [2/2] .venv not found; skipping dependency sync. Run setup.bat if needed.
)

echo.
echo ==================================================
echo   Update complete! Launch Start_PiVisionAI.bat.
echo ==================================================
if "%NO_PAUSE%"=="0" pause
exit /b 0

:fail
echo.
echo Update stopped. See details above.
if "%NO_PAUSE%"=="0" pause
exit /b 1
