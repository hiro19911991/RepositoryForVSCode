@echo off
chcp 65001 > nul
echo ====================================
echo Whisper文字起こしツール セットアップ
echo ====================================
echo.

echo 1. 仮想環境を作成しています...
python -m venv venv
if errorlevel 1 (
    echo エラー: 仮想環境の作成に失敗しました。
    echo Python 3.8以上がインストールされていることを確認してください。
    pause
    exit /b 1
)

echo.
echo 2. 仮想環境を有効化しています...
call venv\Scripts\activate.bat

echo.
echo 3. pipをアップグレードしています...
python -m pip install --upgrade pip

echo.
echo 4. 基本パッケージをインストールしています...
pip install streamlit>=1.28.0 openai-whisper>=20230918 librosa>=0.10.0 numpy>=1.24.0 soundfile>=0.12.0 ffmpeg-python>=0.2.0

echo.
echo 5. GPU対応PyTorchをインストールしています...
echo CPU版PyTorchをアンインストール中...
pip uninstall torch torchaudio -y

echo CUDA対応PyTorchをインストール中...
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

echo.
echo 6. GPU状況を確認しています...
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU detected')"

echo.
echo ====================================
echo セットアップ完了！
echo ====================================
echo.
echo 起動するには start.bat をダブルクリックしてください。
echo または、コマンドプロンプトで「streamlit run app.py」を実行してください。
echo.

pause
