@echo off
REM QuantMind local environment bootstrap (NO Docker required)
REM Usage: open Windows Terminal / CMD, cd into project root, then run:
REM   scripts\bootstrap_windows.bat
setlocal enabledelayedexpansion
cd /d %~dp0\..
echo [QuantMind] project root: %CD%

REM Prefer workbuddy managed python; fall back to system python on PATH
set PY=
if exist "%USERPROFILE%\.workbuddy\binaries\python\versions\3.13.12\python.exe" (
  set PY=%USERPROFILE%\.workbuddy\binaries\python\versions\3.13.12\python.exe
) else (
  set PY=python
)
echo [QuantMind] using python: %PY%
%PY% --version || (echo [QuantMind] python not found, install Python 3.13 first & exit /b 1)

if not exist .venv (
  echo [QuantMind] creating venv .venv ...
  %PY% -m venv .venv
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
echo [QuantMind] installing deps (needs internet via clash system proxy) ...
pip install -e .

echo.
echo [QuantMind] bootstrap done. Quick start:
echo   python -m quantmind.cli backtest --symbol 600000 --exchange SSE --strategy dual_ma --cost
echo   python -m quantmind.cli e2e
echo.
echo [QuantMind] optional real local data (set env vars before running):
echo   set QM_LOCAL_DATA_ROOT=path\to\china-futures-5min
echo   set QM_LOCAL_STOCK_ROOT=path\to\astock-data-toolkit
echo   set QM_SEAT_DATA_ROOT=path\to\TradingAgents_for_Futures\qihuo\database\positioning
endlocal
