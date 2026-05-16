@echo off
setlocal

rem ── CrispAI one-click launcher ───────────────────────────────────────────
rem  Starts the backend if it's not already running, then opens the CrispAI
rem  app window. Safe to double-click multiple times — won't spawn duplicates.

set "ROOT=%~dp0"
set "BACKEND=%ROOT%backend"
set "PORT=7788"

rem Quick check: is server already up?
powershell -NoProfile -Command "try { $r = Invoke-WebRequest -UseBasicParsing -TimeoutSec 1 http://localhost:%PORT%/health; exit 0 } catch { exit 1 }"
if %ERRORLEVEL%==0 goto :open_window

rem Start backend in a new window (so closing this .bat doesn't kill it)
echo Starting CrispAI backend...
start "CrispAI Server" /MIN cmd /c "cd /d "%BACKEND%" && py -3.11 server.py"

rem Wait up to 20s for /health to come up
echo Waiting for server...
for /L %%i in (1,1,40) do (
  powershell -NoProfile -Command "try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 1 http://localhost:%PORT%/health | Out-Null; exit 0 } catch { exit 1 }"
  if not errorlevel 1 goto :open_window
  timeout /t 1 /nobreak >nul
)
echo Server did not start in time. Check the "CrispAI Server" window for errors.
pause
exit /b 1

:open_window
rem Open as app window via Chrome/Edge --app (no URL bar, no tabs)
set "URL=http://localhost:%PORT%/ui"

for %%P in (
  "%ProgramFiles%\Google\Chrome\Application\chrome.exe"
  "%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"
  "%LocalAppData%\Google\Chrome\Application\chrome.exe"
  "%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe"
  "%ProgramFiles%\Microsoft\Edge\Application\msedge.exe"
) do (
  if exist %%P (
    start "" %%P --app="%URL%" --window-size=1280,820 --user-data-dir="%LocalAppData%\CrispAI\browser"
    exit /b 0
  )
)

rem Fallback: default browser
start "" "%URL%"
exit /b 0
