@echo off
setlocal EnableExtensions
cd /d "%~dp0"

title pyDRT Peak Analysis - Setup

echo ============================================================
echo   pyDRT Peak Analysis - Windows setup
echo ============================================================
echo.

rem This installer uses a dedicated local virtual environment so it does
rem not modify or depend on the user's existing Python packages.
set "VENV_DIR=%~dp0.venv"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
set "PYTHON_VERSION=3.13.15"
set "PYTHON_DEFAULT=%LocalAppData%\Programs\Python\Python313\python.exe"
set "PYTHON_URL=https://www.python.org/ftp/python/3.13.15/python-3.13.15-amd64.exe"
set "PYTHON_SHA256=edec09c4853aeae9ac36efb8c9f95b6b8e2fee65eee56d9767a8b7c69c574403"
set "PYTHON_INSTALLER=%TEMP%\pyDRT-python-%PYTHON_VERSION%-amd64.exe"

if not exist "%~dp0requirements.txt" (
    echo ERROR: requirements.txt was not found next to this installer.
    goto :fail
)

if exist "%VENV_PYTHON%" (
    echo Existing local environment found.
    goto :install_packages
)

set "BASE_PYTHON="
for /f "delims=" %%P in ('py -3.13 -c "import sys; print(sys.executable)" 2^>nul') do set "BASE_PYTHON=%%P"

if not defined BASE_PYTHON if exist "%PYTHON_DEFAULT%" set "BASE_PYTHON=%PYTHON_DEFAULT%"

if not defined BASE_PYTHON (
    echo Python 3.13 was not found.
    echo Downloading official Python %PYTHON_VERSION% 64-bit installer from python.org...
    echo.

    where powershell.exe >nul 2>&1
    if errorlevel 1 (
        echo ERROR: Windows PowerShell is required for the automatic Python download.
        goto :fail
    )

    powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference='SilentlyContinue'; Invoke-WebRequest -Uri $env:PYTHON_URL -OutFile $env:PYTHON_INSTALLER; $actual=(Get-FileHash -LiteralPath $env:PYTHON_INSTALLER -Algorithm SHA256).Hash.ToLower(); $expected=($env:PYTHON_SHA256).ToLower(); if ($actual -ne $expected) { Write-Error ('Python installer checksum mismatch. Actual: ' + $actual); exit 2 }"
    if errorlevel 1 (
        echo ERROR: Python download or SHA-256 verification failed.
        if exist "%PYTHON_INSTALLER%" del /q "%PYTHON_INSTALLER%" >nul 2>&1
        goto :fail
    )

    echo Checksum verified. Installing Python %PYTHON_VERSION% for the current user...
    "%PYTHON_INSTALLER%" /quiet InstallAllUsers=0 TargetDir="%LocalAppData%\Programs\Python\Python313" PrependPath=1 Include_pip=1 Include_launcher=1 InstallLauncherAllUsers=0 Include_test=0 Include_doc=0 Shortcuts=0
    if errorlevel 1 (
        echo ERROR: Python installation failed.
        del /q "%PYTHON_INSTALLER%" >nul 2>&1
        goto :fail
    )
    del /q "%PYTHON_INSTALLER%" >nul 2>&1

    if not exist "%PYTHON_DEFAULT%" (
        echo ERROR: Python was installed but python.exe was not found at the expected location:
        echo        %PYTHON_DEFAULT%
        goto :fail
    )
    set "BASE_PYTHON=%PYTHON_DEFAULT%"
)

echo Using Python:
echo   %BASE_PYTHON%
"%BASE_PYTHON%" --version
if errorlevel 1 goto :fail

echo.
echo Creating isolated environment: .venv
"%BASE_PYTHON%" -m venv "%VENV_DIR%"
if errorlevel 1 (
    echo ERROR: Could not create the virtual environment.
    goto :fail
)

:install_packages
echo.
echo Installing/updating required Python packages...
"%VENV_PYTHON%" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 goto :fail
"%VENV_PYTHON%" -m pip install -r "%~dp0requirements.txt"
if errorlevel 1 (
    echo ERROR: One or more required packages could not be installed.
    goto :fail
)

echo.
echo Verifying required imports...
"%VENV_PYTHON%" -c "import numpy, scipy, pandas, matplotlib, sklearn, click, cvxopt, galvani, zahner_analysis; from PyQt5 import QtWidgets; print('Dependency import check: PASS')"
if errorlevel 1 (
    echo ERROR: Dependency import verification failed.
    goto :fail
)

echo.
echo Running numerical smoke tests...
"%VENV_PYTHON%" "%~dp0smoke_test.py"
if errorlevel 1 (
    echo ERROR: smoke_test.py failed.
    goto :fail
)
"%VENV_PYTHON%" "%~dp0smoke_test_lambda.py"
if errorlevel 1 (
    echo ERROR: smoke_test_lambda.py failed.
    goto :fail
)

echo.
echo ============================================================
echo   Setup completed successfully.
echo   Double-click run_windows.bat to start the program.
echo ============================================================
echo.
pause
exit /b 0

:fail
echo.
echo ============================================================
echo   SETUP FAILED
echo   Check the error message above and confirm that the computer
echo   has an internet connection for the first installation.
echo ============================================================
echo.
pause
exit /b 1
