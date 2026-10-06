@echo off
setlocal EnableDelayedExpansion
title ISL Communication System
cd /d "%~dp0"

echo ============================================================
echo  ISL Communication System - Starting
echo ============================================================
echo.

rem ---------------------------------------------------------------
rem 1. Locate Python 3.12 specifically
rem ---------------------------------------------------------------
set "PYEXE="

where py >nul 2>nul
if %errorlevel%==0 (
    py -3.12 --version >nul 2>nul
    if !errorlevel!==0 (
        set "PYEXE=py -3.12"
    )
)

if not defined PYEXE (
    python --version 2>nul | findstr /r "3\.12\." >nul 2>nul
    if !errorlevel!==0 (
        set "PYEXE=python"
    )
)

if not defined PYEXE (
    echo [ERROR] Python 3.12 was not found on this system.
    echo.
    echo This project requires Python 3.12.10 specifically. Please install it from:
    echo   https://www.python.org/downloads/release/python-31210/
    echo During installation, make sure to check "Add python.exe to PATH".
    echo.
    echo After installing, double-click RUN.bat again.
    echo.
    pause
    exit /b 1
)

echo [OK] Using Python: %PYEXE%
echo.

rem ---------------------------------------------------------------
rem 2. Create the virtual environment if it does not exist yet
rem ---------------------------------------------------------------
if not exist "%~dp0.venv\Scripts\python.exe" (
    echo [SETUP] Creating a virtual environment in .venv ...
    %PYEXE% -m venv "%~dp0.venv"
    if not exist "%~dp0.venv\Scripts\python.exe" (
        echo [ERROR] Failed to create the virtual environment.
        pause
        exit /b 1
    )
)

set "VENV_PY=%~dp0.venv\Scripts\python.exe"

rem ---------------------------------------------------------------
rem 3. Install/verify dependencies (only re-installs if requirements.txt
rem    changed since the last successful install, so restarts are fast)
rem ---------------------------------------------------------------
set "MARKER=%~dp0.venv\requirements.installed"
set "NEED_INSTALL=1"

if exist "%MARKER%" (
    fc /b "%MARKER%" "%~dp0requirements.txt" >nul 2>nul
    if !errorlevel!==0 set "NEED_INSTALL=0"
)

if "%NEED_INSTALL%"=="1" (
    echo [SETUP] Installing Python dependencies. This may take a few minutes
    echo         the first time - please be patient...
    echo.
    "%VENV_PY%" -m pip install --upgrade pip >nul
    "%VENV_PY%" -m pip install -r "%~dp0requirements.txt"
    if !errorlevel! neq 0 (
        echo.
        echo [ERROR] Dependency installation failed. See the error above.
        echo Common fixes:
        echo   - Check your internet connection.
        echo   - Make sure you have disk space available.
        echo   - Try running RUN.bat again ^(pip caches downloads^).
        echo.
        pause
        exit /b 1
    )
    copy /y "%~dp0requirements.txt" "%MARKER%" >nul
    echo.
    echo [OK] Dependencies installed successfully.
    echo.
) else (
    echo [OK] Dependencies already installed and up to date.
    echo.
)

rem ---------------------------------------------------------------
rem 4. Launch the application
rem ---------------------------------------------------------------
echo [SETUP] If no trained model exists yet, the app will now download
echo         real ISL training videos and train the model automatically.
echo         This needs internet access and only happens once.
echo.
echo ============================================================
echo.

"%VENV_PY%" "%~dp0main.py"
set "APP_EXIT_CODE=%errorlevel%"

echo.
echo ============================================================
if "%APP_EXIT_CODE%"=="0" (
    echo  ISL Communication System closed normally.
) else (
    echo  The application exited with an error ^(code %APP_EXIT_CODE%^).
    echo  Scroll up to see the actual Python error message above.
)
echo ============================================================
echo.
pause
endlocal
