@echo off
cd /d "%~dp0..\backend"
echo Starting CrispAI backend server...
python server.py
pause
