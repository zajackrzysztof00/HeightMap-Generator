@echo off
title Cleanup Project
echo 🧹 Cleaning up project files...
cd /d "%~dp0"

echo Removing Python cache files...
rmdir /s /q __pycache__ 2>nul
rmdir /s /q .pytest_cache 2>nul
del /q *.pyc 2>nul

echo Removing temporary files...
del /q *.tmp 2>nul
del /q *.log 2>nul

echo ✅ Cleanup completed!
pause
