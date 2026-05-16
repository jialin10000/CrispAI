@echo off
setlocal

rem ── CrispAI one-click launcher ───────────────────────────────────────────
rem  Starts the backend if not already running, then opens the CrispAI app
rem  window. Uses curl (Windows 10+) for health checks — much faster than
rem  PowerShell. Safe to double-click multiple times.

set "ROOT=%~dp0"
set "BACKEND=%ROOT%backend"
set "PORT=7788"
set "URL=http://localhost:%PORT%"

rem ── Step 1: is server already up? ─────────────────────────────
curl -sf --max-time 1 "%URL%/health" >nul 2>&1
if %ERRORLEVEL%==0 (
    echo Server already running.
    goto :open_window
)

rem ── Step 2: start backend in detached minimized window ────────
echo Starting CrispAI backend...
start "CrispAI Server" /MIN cmd /c "cd /d "%BACKEND%" && py -3.11 server.py"

rem ── Step 3: poll /health until up (max 30s) ───────────────────
echo Waiting for server to come up...
for /L %%i in (1,1,30) do (
    curl -sf --max-time 1 "%URL%/health" >nul 2>&1
    if not errorlevel 1 goto :open_window
    timeout /t 1 /nobreak >nul
)
echo.
echo ERROR: Server did not come up in 30 seconds.
echo Check the "CrispAI Server" window for the actual error.
echo.
pause
exit /b 1

:open_window
echo Opening CrispAI window...

rem ── Step 4: launch as borderless app window (Chrome/Edge --app) ──
for %%P in (
    "%ProgramFiles%\Google\Chrome\Application\chrome.exe"
    "%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"
    "%LocalAppData%\Google\Chrome\Application\chrome.exe"
    "%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe"
    "%ProgramFiles%\Microsoft\Edge\Application\msedge.exe"
) do (
    if exist %%P (
        start "" %%P --app="%URL%/ui" --window-size=1280,820 ^
            --user-data-dir="%LocalAppData%\CrispAI\browser"
        exit /b 0
    )
)

rem ── Fallback: default browser ─────────────────────────────────
echo No Chrome/Edge found, opening in default browser.
start "" "%URL%/ui"
exit /b 0
