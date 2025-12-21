# 🎤 Whisper文字起こしツール

OpenAIのWhisperモデルを使用した高精度な音声文字起こしWebアプリケーションです。GPU加速対応で高速処理が可能です。

## ✨ 機能

- **高精度文字起こし**: OpenAI Whisperの5つのモデル（tiny、base、small、medium、large）から選択可能
- **多言語対応**: 日本語、英語、中国語など9言語の音声認識
- **GPU加速**: CUDA対応GPUを自動検出し、高速処理
- **タイムスタンプ付き出力**: セグメント単位の詳細な時間情報
- **複数フォーマット対応**: MP3、WAV、M4A、OGG、FLAC
- **Webインターフェース**: 直感的なStreamlitベースのUI
- **結果ダウンロード**: テキストファイルとして結果を保存

## 📋 必要要件

### システム要件
- Python 3.8以上
- FFmpeg（音声処理用）
- （推奨）CUDA対応GPU（高速処理用）

### FFmpegのインストール
1. [FFmpeg公式サイト](https://ffmpeg.org/download.html)からダウンロード
2. システムPATHに追加
3. コマンドプロンプトで `ffmpeg -version` で動作確認

## 🚀 セットアップ

### 1. リポジトリのクローン
```bash
git clone <このリポジトリのURL>
cd whisper-transcription-app
```

### 2. 仮想環境の作成（推奨）
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. パッケージのインストール

#### CPU版（標準）
```bash
pip install -r requirements.txt
```

#### GPU版（CUDA対応、推奨）
```bash
# まずCPU版をアンインストール
pip uninstall torch torchaudio -y

# CUDA版をインストール（CUDA 11.8対応）
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# その他の依存関係をインストール
pip install streamlit>=1.28.0 openai-whisper>=20230918 librosa>=0.10.0 numpy>=1.24.0 soundfile>=0.12.0 ffmpeg-python>=0.2.0
```

### 4. アプリケーションの起動
```bash
streamlit run app.py
```

ブラウザで `http://localhost:8501` にアクセスしてください。

## 📖 使用方法

1. **モデル選択**: サイドバーで処理精度を選択
   - `tiny`: 最速・低精度
   - `base`: バランス型
   - `small`: 中程度の精度
   - `medium`: 高精度
   - `large`: 最高精度（推奨、GPU使用時）

2. **言語設定**: 音声の言語を指定（デフォルト：日本語）
   - 自動検出も可能

3. **音声ファイルアップロード**: 対応フォーマットの音声ファイルを選択

4. **文字起こし実行**: 「文字起こし開始」ボタンをクリック

5. **結果確認・ダウンロード**: 
   - テキスト結果をブラウザで確認
   - タイムスタンプ付き詳細表示
   - テキストファイルとしてダウンロード

## ⚙️ 設定オプション

### モデル性能比較
| モデル | サイズ | VRAM使用量 | 処理速度 | 精度 |
|--------|--------|------------|----------|------|
| tiny   | ~39MB  | ~1GB       | 最速     | 低   |
| base   | ~74MB  | ~1GB       | 高速     | 中   |
| small  | ~244MB | ~2GB       | 中速     | 中高 |
| medium | ~769MB | ~5GB       | 低速     | 高   |
| large  | ~1550MB| ~10GB      | 最低速   | 最高 |

### GPU設定
- CUDA対応GPU使用時は自動でGPU加速が有効化
- サイドバーに現在の使用デバイスを表示
- GPU使用時はfp16精度で高速化

## 🔧 技術仕様

- **フレームワーク**: Streamlit
- **音声処理**: OpenAI Whisper
- **深層学習**: PyTorch
- **音声読み込み**: librosa
- **対応フォーマット**: MP3, WAV, M4A, OGG, FLAC
- **最大ファイルサイズ**: 200MB

## ⚠️ 注意事項

- 初回実行時はWhisperモデルのダウンロードが行われます
- largeモデルは高精度ですが処理時間が長くなります
- GPU使用には十分なVRAMが必要です
- 長時間の音声ファイルは大量のメモリを使用する可能性があります

## 🐛 トラブルシューティング

### FFmpegエラー
```
⚠️ FFmpegがインストールされていません
```
→ FFmpegをインストールし、システムPATHに追加してください

### CUDAエラー
```
GPU加速が利用できません
```
→ CUDA対応PyTorchがインストールされているか確認してください

### メモリエラー
→ より小さなモデル（small、base）を使用するか、音声ファイルを分割してください

## 📝 ライセンス

このプロジェクトはOpenAI Whisperを使用しており、MITライセンスの下で公開されています。

## 🤝 貢献

バグ報告や機能要求は、GitHubのIssuesページでお知らせください。
