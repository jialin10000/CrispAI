@echo off
cd /d "%~dp0backend"
echo Starting CrispAI backend server...
python server.py
pause
