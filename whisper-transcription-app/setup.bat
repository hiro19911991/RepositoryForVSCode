@echo off
chcp 65001 > nul
echo ====================================
echo Whisper Transcription Tool Setup
echo ====================================
echo.

echo 1. Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo Error: failed to create virtual environment.
    echo Please make sure Python 3.8+ is installed.
    pause
    exit /b 1
)

echo.
echo 2. Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo 3. Upgrading pip...
python -m pip install --upgrade pip

echo.
echo 4. Installing base packages...
pip install streamlit>=1.28.0 openai-whisper>=20230918 librosa>=0.10.0 numpy>=1.24.0 soundfile>=0.12.0 ffmpeg-python>=0.2.0

echo.
echo 5. Installing CUDA-enabled PyTorch...
echo Uninstalling CPU-only torch packages...
pip uninstall torch torchaudio -y

echo Installing CUDA wheels...
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

echo.
echo 6. Checking GPU status...
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU detected')"

echo.
echo ====================================
echo Setup complete.
echo ====================================
echo.
echo To start, double-click start.bat
echo or run: streamlit run app.py
echo.

pause
