@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "HOST=127.0.0.1"
set "PORT=5000"
set "NO_BROWSER=0"
set "NO_PAUSE=0"
set "POS_COUNT=0"

:parse_args
if "%~1"=="" goto :args_done
if /i "%~1"=="--no-browser" (
    set "NO_BROWSER=1"
    shift
    goto :parse_args
)
if /i "%~1"=="--nopause" (
    set "NO_PAUSE=1"
    shift
    goto :parse_args
)
if /i "%~1"=="--host" (
    if not "%~2"=="" (
        set "HOST=%~2"
        shift
    )
    shift
    goto :parse_args
)
if /i "%~1"=="--port" (
    if not "%~2"=="" (
        set "PORT=%~2"
        shift
    )
    shift
    goto :parse_args
)
if "%POS_COUNT%"=="0" (
    echo(%~1| findstr /r "^[0-9][0-9]*$" >nul
    if not errorlevel 1 (
        set "PORT=%~1"
    ) else (
        set "HOST=%~1"
    )
    set "POS_COUNT=1"
    shift
    goto :parse_args
)
set "PORT=%~1"
shift
goto :parse_args

:args_done
set "BROWSER_HOST=%HOST%"
if "%BROWSER_HOST%"=="0.0.0.0" set "BROWSER_HOST=127.0.0.1"
set "APP_URL=http://%BROWSER_HOST%:%PORT%"

REM 1. Pick Python interpreter (.venv first, then global Python if dependencies exist)
set "PY_EXE="
if exist ".venv\Scripts\python.exe" (
    set "PY_EXE=.venv\Scripts\python.exe"
) else (
    where python >nul 2>&1
    if not errorlevel 1 (
        python -c "import flask, cv2, ultralytics" >nul 2>&1
        if not errorlevel 1 (
            set "PY_EXE=python"
        )
    )
)

if not defined PY_EXE (
    echo [ERROR] PiVisionAI environment is not set up yet.
    echo Please double-click setup.bat first to install dependencies.
    echo.
    if "%NO_PAUSE%"=="0" pause
    exit /b 1
)

REM 2. Check if port is already in use (avoid duplicate startup failures)
"%PY_EXE%" -c "import socket, sys; s = socket.socket(); s.settimeout(0.5); sys.exit(0 if s.connect_ex(('127.0.0.1', %PORT%)) == 0 else 1)" >nul 2>&1
if not errorlevel 1 (
    "%PY_EXE%" -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('%APP_URL%/api/v1/system/status', timeout=2).status == 200 else 1)" >nul 2>&1
    if not errorlevel 1 (
        echo PiVisionAI is already running at %APP_URL%. Opening window...
        if "%NO_BROWSER%"=="0" call :open_browser
        exit /b 0
    ) else (
        echo [ERROR] Port %PORT% is already in use by another application.
        echo Close the program using port %PORT% or pass a different port: Start_PiVisionAI.bat 5050
        echo.
        if "%NO_PAUSE%"=="0" pause
        exit /b 1
    )
)

echo ==================================================
echo   Starting PiVisionAI on %APP_URL%
echo   Keep this window open while inspecting.
echo   Press Ctrl+C or close this window to stop.
echo ==================================================
echo.

REM 3. Wait for API readiness in background, then launch UI window
if "%NO_BROWSER%"=="0" call :launch_browser_waiter

REM 4. Launch server in foreground
"%PY_EXE%" cli.py run --host %HOST% --port %PORT%
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo [ERROR] PiVisionAI stopped with exit code %EXIT_CODE%.
    if "%NO_PAUSE%"=="0" pause
)
exit /b %EXIT_CODE%

:launch_browser_waiter
start "" /min powershell -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -Command "$u = '%APP_URL%'; for ($i = 0; $i -lt 90; $i++) { try { $r = Invoke-WebRequest -UseBasicParsing -Uri ($u + '/api/v1/system/status') -TimeoutSec 1; if ($r.StatusCode -eq 200) { $e = @('C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe', 'C:\Program Files\Microsoft\Edge\Application\msedge.exe') | Where-Object { Test-Path $_ } | Select-Object -First 1; if (-not $e) { $c = Get-Command msedge -ErrorAction SilentlyContinue; if ($c) { $e = $c.Source } }; if ($e) { Start-Process $e -ArgumentList ('--app=' + $u) } else { Start-Process $u }; break } } catch {} Start-Sleep -Milliseconds 500 }"
goto :eof

:open_browser
powershell -NoProfile -ExecutionPolicy Bypass -Command "$u = '%APP_URL%'; $e = @('C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe', 'C:\Program Files\Microsoft\Edge\Application\msedge.exe') | Where-Object { Test-Path $_ } | Select-Object -First 1; if (-not $e) { $c = Get-Command msedge -ErrorAction SilentlyContinue; if ($c) { $e = $c.Source } }; if ($e) { Start-Process $e -ArgumentList ('--app=' + $u) } else { Start-Process $u }"
goto :eof
