@echo off
REM Local quality gate (git-free). Runs the same checks a pre-commit hook would.
REM Usage:  check.bat
cd /d "%~dp0"
set FAILED=0

echo === ruff (lint) ===
python -m ruff check . || set FAILED=1

echo === ruff (format check) ===
python -m ruff format --check . || set FAILED=1

echo === mypy (types) ===
python -m mypy || set FAILED=1

echo === pytest (unit tier) ===
python -m pytest || set FAILED=1

if "%FAILED%"=="1" (
  echo.
  echo   CHECKS FAILED
  exit /b 1
) else (
  echo.
  echo   ALL CHECKS PASSED
)
