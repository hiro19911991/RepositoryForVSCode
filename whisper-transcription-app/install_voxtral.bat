@echo off
echo Installing Voxtral dependencies...
echo.

REM Voxtral dependencies installation
echo Step 1: Installing vLLM (may take several minutes)...
pip install vllm>=0.6.0
if %errorlevel% neq 0 (
    echo Warning: vLLM installation failed. Trying alternative method...
    pip install --no-deps vllm
)

echo.
echo Step 2: Installing mistral-common...
pip install mistral-common>=1.9.0
if %errorlevel% neq 0 (
    echo Error: mistral-common installation failed
    goto :error
)

echo.
echo Step 3: Installing additional audio libraries...
pip install sox>=1.4.1
pip install --upgrade transformers>=4.45.0

echo.
echo Step 4: Checking installations...
python -c "import vllm; print('vLLM:', vllm.__version__)"
python -c "import mistral_common; print('mistral-common installed successfully')"

echo.
echo Installation completed! You can now use Voxtral Realtime in the app.
echo Note: Voxtral requires a GPU with 16GB+ VRAM for optimal performance.
echo.
pause
exit /b 0

:error
echo.
echo Installation failed. Please check the error messages above.
echo You may need to install CUDA toolkit and compatible PyTorch first.
echo.
pause
exit /b 1