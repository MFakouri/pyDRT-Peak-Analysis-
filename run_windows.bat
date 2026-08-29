@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "VENV_PYTHON=%~dp0.venv\Scripts\python.exe"

if not exist "%VENV_PYTHON%" (
    echo First run: the local Python environment is not installed yet.
    echo Starting the setup now...
    echo.
    call "%~dp0install_windows.bat"
    if errorlevel 1 exit /b 1
)

"%VENV_PYTHON%" "%~dp0launch.py"
if errorlevel 1 (
    echo.
    echo The application exited with an error.
    pause
    exit /b 1
)
exit /b 0
