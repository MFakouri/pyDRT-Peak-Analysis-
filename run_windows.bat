@echo off
cd /d "%~dp0"
py launch.py
if errorlevel 1 pause
