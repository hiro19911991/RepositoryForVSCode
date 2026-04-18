@echo off
chcp 65001 > nul
echo ====================================
echo Whisper Transcription Tool Starting...
echo ====================================
echo.

REM GPU check
echo Checking GPU status...
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
echo.

REM Virtual environment check
if exist venv\ (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo Warning: virtual environment not found. Using global environment.
)

echo.
echo Starting Streamlit application...
echo Open http://localhost:8501 in your browser
echo.
echo Press Ctrl+C to stop
echo.

echo Running preflight syntax check...
python -m py_compile app.py
if errorlevel 1 (
    echo Error: app.py has syntax issues. Please fix before starting Streamlit.
    pause
    exit /b 1
)

echo Checking unresolved merge conflict markers...
findstr /n /r "^<<<<<<< ^======= ^>>>>>>>" app.py > nul
if %errorlevel%==0 (
    echo Error: unresolved merge markers found in app.py.
    echo Please remove lines that start with ^<^<^<^<^<^<^<, =======, or ^>^>^>^>^>^>^>.
    pause
    exit /b 1
)

streamlit run app.py

pause
