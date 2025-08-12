@echo off
title Run Tests
echo 🧪 Running Heightmap Generator Tests...
cd /d "%~dp0"
python launchers\run_tests.py
echo.
echo 🏁 Test run completed!
pause
