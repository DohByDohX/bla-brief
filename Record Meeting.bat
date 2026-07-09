@echo off
title Meeting Recorder
echo.
echo   Starting Meeting Recorder...
echo   (Mic + System Audio - Dual Capture)
echo.
cd /d "%~dp0"
"C:\Users\pchavan\AppData\Local\Programs\Python\Python312\python.exe" -m meeting_recorder %*
pause
