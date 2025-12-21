@echo off
chcp 65001 > nul
echo ====================================
echo Whisper文字起こしツール 起動中...
echo ====================================
echo.

REM GPU状況確認
echo GPU状況を確認中...
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
echo.

REM 仮想環境の確認
if exist venv\ (
    echo 仮想環境を有効化しています...
    call venv\Scripts\activate.bat
) else (
    echo 注意: 仮想環境が見つかりません。グローバル環境を使用します。
)

echo.
echo Streamlitアプリケーションを起動しています...
echo ブラウザで http://localhost:8501 にアクセスしてください
echo.
echo 終了するには Ctrl+C を押してください
echo.

streamlit run app.py

pause
