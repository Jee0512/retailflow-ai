@echo off
title RetailFlow AI - System Launcher
cd /d "%~dp0"
echo ===================================================
echo   RetailFlow AI - Supermarket Analytics Platform
echo ===================================================
echo [1/3] Verifying Python and SQLite dependencies...
echo [2/3] Launching Home Portal on http://127.0.0.1:8000
echo [3/3] Launching Streamlit Dashboard on http://127.0.0.1:8501
echo.
python run.py
pause
