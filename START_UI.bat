@echo off
title Heightmap Generator UI
echo 🏔️  Starting Heightmap Generator UI...
cd /d "%~dp0"
python run_ui.py
if errorlevel 1 (
    echo.
    echo ❌ Error running the application
    echo 📦 Please ensure Python and dependencies are installed:
    echo    pip install numpy opencv-python Pillow tkinter
    echo.
    pause
)
