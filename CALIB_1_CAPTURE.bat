@echo off
cd /d "%~dp0"
set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=%~dp0..\.venv\Scripts\python.exe"
if exist "%PY%" (
    echo Using: "%PY%"
    "%PY%" "%~dp0calibrate_capture.py" %*
) else (
    echo .venv not found - using py -3.12
    py -3.12 "%~dp0calibrate_capture.py" %*
)
echo.
pause
