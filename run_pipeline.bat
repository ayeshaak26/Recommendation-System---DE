@echo off
REM ---------------------------------------------------------------------------
REM  Runs the full pipeline. Used by Windows Task Scheduler, but you can also
REM  double-click it or run it from a terminal.
REM
REM    run_pipeline.bat                   -> python run_pipeline.py --stage all
REM    run_pipeline.bat --full-refresh    -> extra arguments are passed through
REM
REM  Output is appended to logs\scheduler.log. The detailed, rotating log
REM  written by the Python code itself is logs\pipeline.log.
REM ---------------------------------------------------------------------------
setlocal

REM Task Scheduler starts in C:\Windows\System32, so move to this file's folder first.
cd /d "%~dp0"

if not exist "logs" mkdir "logs"
set "LOGFILE=logs\scheduler.log"

REM Make Python print UTF-8 (movie titles contain accents) and flush each line.
set "PYTHONIOENCODING=utf-8"
set "PYTHONUNBUFFERED=1"

echo ================================================================>> "%LOGFILE%"
echo [%date% %time%] Pipeline starting>> "%LOGFILE%"

if not exist ".venv\Scripts\activate.bat" (
    echo [%date% %time%] ERROR: virtual environment not found at "%~dp0.venv">> "%LOGFILE%"
    exit /b 2
)
call ".venv\Scripts\activate.bat"

python run_pipeline.py --stage all %* >> "%LOGFILE%" 2>&1
set "EXITCODE=%ERRORLEVEL%"

echo [%date% %time%] Pipeline finished with exit code %EXITCODE%>> "%LOGFILE%"

REM Data quality report: papermill executes notebooks\02_data_quality_monitor.ipynb and
REM saves a dated copy in notebooks\reports\. It runs even after a failed pipeline run
REM (so the failure shows up in the report) and does not change the exit code.
python -m pipeline.report >> "%LOGFILE%" 2>&1
echo [%date% %time%] Report finished with exit code %ERRORLEVEL%>> "%LOGFILE%"

REM Pass the result back so Task Scheduler's "Last Run Result" shows failures (0 = success).
endlocal & exit /b %EXITCODE%
