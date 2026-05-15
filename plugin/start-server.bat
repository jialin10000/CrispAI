@echo off
cd /d "%~dp0..\backend"
echo Starting CrispAI backend server...
py -3.11 server.py
pause
