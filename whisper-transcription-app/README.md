# 🎤 Whisper文字起こしツール with Speaker Diarization

OpenAI Whisper・WhisperX・ReazonSpeechとpyannote.audioを使用した高精度な音声文字起こしWebアプリケーションです。話者分離機能により、誰が話しているかを自動識別できます。

## ✨ 機能

- **高精度文字起こし**: 4つのASRエンジンから用途に応じて選択可能
- **🚀 WhisperX対応**: faster-whisper + VADで最大70倍高速化。単語レベルのタイムスタンプ取得が可能
- **🎌 ReazonSpeech v2.0対応**: 日本語専用の高精度モデル（sherpa-onnxベース・Python 3.13対応）
- **🎭 話者分離（Speaker Diarization）**: pyannote.audio 4.0による高精度な話者自動識別
  - 誰がいつ話しているかを自動認識
  - 話者数の自動推定または手動指定
  - 話者ラベル付きテキスト出力
- **多言語対応**: 日本語、英語、中国語など9言語の音声認識
- **日本語特化モード**: 日本語音声に最適化された設定
- **GPU加速**: CUDA対応GPUを自動検出し、高速処理
- **タイムスタンプ付き出力**: セグメント単位・単語単位の詳細な時間情報
- **長時間音声対応**: 10分以上の音声を自動チャンク分割処理
- **ハルシネーション防止**: 繰り返しや無音の自動検出
- **複数フォーマット対応**: MP3、WAV、M4A、OGG、FLAC
- **Webインターフェース**: 直感的なStreamlitベースのUI
- **結果ダウンロード**: テキストファイルとして結果を保存

## 📋 必要要件

### システム要件
- Python 3.8以上（3.13動作確認済み）
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

# CUDA 12.8版をインストール（torch 2.8.0）
pip install torch==2.8.0+cu128 torchaudio==2.8.0+cu128 --index-url https://download.pytorch.org/whl/cu128

# その他の依存関係をインストール
pip install -r requirements.txt
```

> **注意**: CUDA 11.8（cu118）向けには torch 2.8.0 のホイールが存在しません。cu128 を推奨します。

#### WhisperX（オプション）
```bash
pip install whisperx
# WhisperX インストール後、torch が CPU 版に戻る場合があるため再インストール
pip install torch==2.8.0+cu128 torchaudio==2.8.0+cu128 --index-url https://download.pytorch.org/whl/cu128 --force-reinstall
```

#### ReazonSpeech v2.0（オプション・日本語専用）
```bash
pip install git+https://github.com/reazon-research/reazonspeech.git#subdirectory=pkg/k2-asr
```
sherpa-onnxが自動でインストールされます。Python 3.13・Windowsに完全対応しています。

#### 🎭 話者分離（pyannote.audio）を使う場合

**重要**: pyannote.audioは既にインストールされていますが、実際に使用するには以下の手順が必要です：

1. **Hugging Faceアカウント作成**（まだの場合）
   - https://huggingface.co/ でアカウントを作成

2. **利用規約への同意（4モデル全て必須）**
   - 以下のモデルページを開き、それぞれで「Agree and access repository」をクリック：
     - https://huggingface.co/pyannote/speaker-diarization-3.1
     - https://huggingface.co/pyannote/segmentation-3.0
     - https://huggingface.co/pyannote/speaker-diarization-community-1
     - https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM（公開モデルのため任意）

3. **アクセストークンの作成**
   - https://huggingface.co/settings/tokens でトークンを作成
   - トークンタイプ: **「Classic」の「Read」を推奨**（Fine-grainedトークンは権限設定が複雑で403エラーになる場合があります）

4. **アプリでの設定**
   - アプリのサイドバー「話者分離を有効化」にチェック
   - "Hugging Face Access Token" にトークンを貼り付け
   - 話者数を設定（0で自動推定）

### 4. アプリケーションの起動
```bash
streamlit run app.py
```

ブラウザで `http://localhost:8501` にアクセスしてください。

## 📖 使用方法

### 基本的な使い方

1. **ASRエンジン選択**: サイドバーでエンジンを選択
   - `Whisper (標準)`: OpenAI Whisper。多言語対応の汎用音声認識エンジン
   - `Whisper (日本語特化)`: 日本語向け設定（initial_prompt・word_timestamps）を適用したWhisper
   - `WhisperX (高速・単語精度)`: faster-whisper + VADで最大70倍高速化。単語レベルのタイムスタンプ取得が可能
   - `ReazonSpeech v2.0`: 日本語専用の高精度モデル（sherpa-onnxベース・別途インストールが必要）

2. **モデル選択**: サイドバーで処理精度を選択
   - `turbo`: 最新最適化モデル（Whisper標準用・推奨）
   - `large-v3`: 最高精度・最新世代（日本語特化・WhisperX用デフォルト）
   - `large-v2`: large-v3に次ぐ高精度
   - `large`: 旧世代largeモデル
   - `medium`: 高精度・バランス型
   - `small`: 中程度の精度
   - `base`, `tiny`: 軽量・高速（精度は低め）

3. **言語設定**: 音声の言語を指定（デフォルト：日本語）
   - 自動検出も可能（WhisperXでは自動検出時に事前言語検出を実行）

4. **音声ファイルアップロード**: 対応フォーマットの音声ファイルを選択

5. **文字起こし実行**: 「文字起こし開始」ボタンをクリック

6. **結果確認・ダウンロード**:
   - テキスト結果をブラウザで確認
   - タイムスタンプ付き詳細表示
   - テキストファイルとしてダウンロード

### 🎭 話者分離機能の使い方

話者分離を有効にすると、複数人が話している音声で誰が話しているかを自動識別できます。

1. **サイドバーで「話者分離を有効化」にチェック**

2. **Hugging Face Access Tokenを入力**
   - HuggingFaceの各モデルページで利用規約に同意済みであること
   - https://huggingface.co/settings/tokens でトークンを作成（Classic / Read推奨）
   - トークンをサイドバーの入力欄に貼り付け
   - **注意**: アプリを再起動するとトークンはリセットされるため、再入力が必要です

3. **話者数の設定**
   - **話者数（0で自動推定）**: 話者数が分かっている場合は指定（例：2人の会話なら2）
   - **最小話者数**: 自動推定時の最小値（デフォルト：1）
   - **最大話者数**: 自動推定時の最大値（デフォルト：8）

4. **文字起こし実行**
   - 通常通り音声ファイルをアップロードして「文字起こし開始」をクリック
   - 文字起こし後、自動的に話者分離処理が実行されます

5. **結果の確認**
   - **基本テキスト**: 通常の文字起こし結果
   - **🎭 話者付きテキスト**: 各セグメントに話者ラベル（SPEAKER_00、SPEAKER_01など）が付与
   - **🕐 詳細（タイムスタンプ付き）**: 話者とタイムスタンプ付きの詳細表示

### 話者分離の出力例

```
[SPEAKER_00] こんにちは、今日はよろしくお願いします。
[SPEAKER_01] はい、こちらこそよろしくお願いします。
[SPEAKER_00] それでは早速始めましょう。
[SPEAKER_01] 了解しました。
```

## ⚙️ 設定オプション

### ASRエンジン詳細

| エンジン | ベース技術 | 処理速度 | 主な特徴 | 必要条件 |
|---|---|---|---|---|
| Whisper (標準) | openai-whisper | 標準 | 多言語対応・安定した精度 | なし |
| Whisper (日本語特化) | openai-whisper | 標準 | 日本語initial_prompt・word_timestamps有効 | なし |
| WhisperX (高速・単語精度) | faster-whisper + VAD | 最大70倍高速 | 単語タイムスタンプ・ハルシネーション削減・話者分離統合 | `pip install whisperx` |
| ReazonSpeech v2.0 | sherpa-onnx | 標準（CPU） | 日本語専用・サブワードタイムスタンプ・Python 3.13対応 | 別途インストール |

#### Whisper (標準)
- **openai-whisper**ライブラリを使用した公式実装
- tiny〜large-v3・turboまでのモデルを選択可能
- 日本語・英語・中国語など多言語に対応
- 10分超の音声は自動でチャンク分割処理

#### Whisper (日本語特化)
- Whisper (標準) と同じモデルを使用しつつ、以下の設定を適用：
  - `language="ja"` で日本語を強制指定
  - `initial_prompt` で文脈を日本語に誘導
  - `word_timestamps=True` でセグメント精度向上
- 日本語音声では標準モードより誤認識が少ない

#### WhisperX (高速・単語精度)
- **faster-whisper**（CTranslate2バックエンド）により標準Whisperより大幅に高速
- **VAD（Voice Activity Detection）**で無音区間を事前除去し、ハルシネーションを削減
- バッチ処理（batch_size設定可）による並列化で長時間音声も高速処理
- **wav2vec2**アライメントモデルによる単語レベルの正確なタイムスタンプ（オン/オフ可）
- pyannote統合の話者分離をWhisperXパイプライン内で実行
- float16 / int8 の計算精度を選択可能（int8でVRAM削減）
- **言語未指定時**: 最初の30秒で自動言語検出してから文字起こしを実行

#### ReazonSpeech v2.0
- reazon-research社が日本語コーパスで学習した日本語専用モデル
- パッケージv3.0.0より**sherpa-onnxバックエンド**に移行。Python 3.13・Windowsに完全対応
- `fp32`（高精度）・`int8`（省メモリ）・`int8-fp32`（中間）の3精度から選択可能
- サブワードレベルのタイムスタンプ（2秒間隔でセグメント化）
- CPU実行（GPU版は`sherpa-onnx-cuda`の別途インストールが必要）
- インストール: `pip install git+https://github.com/reazon-research/reazonspeech.git#subdirectory=pkg/k2-asr`

### Whisper / WhisperX モデル比較

| モデル | サイズ | VRAM目安 | 処理速度 | 精度 | 対応エンジン |
|--------|--------|----------|----------|------|---|
| tiny   | ~39MB  | ~1GB     | 最速     | 低   | Whisper / WhisperX |
| base   | ~74MB  | ~1GB     | 高速     | 中   | Whisper / WhisperX |
| small  | ~244MB | ~2GB     | 中速     | 中高 | Whisper / WhisperX |
| medium | ~769MB | ~5GB     | 低速     | 高   | Whisper / WhisperX |
| large  | ~1.5GB | ~10GB    | 最低速   | 高   | Whisper / WhisperX |
| large-v2 | ~1.5GB | ~10GB  | 最低速   | 高+  | Whisper / WhisperX |
| large-v3 | ~1.5GB | ~10GB  | 最低速   | 最高 | Whisper / WhisperX |
| turbo  | ~809MB | ~6GB     | 高速     | 高   | Whisper (標準)のみ |

> **WhisperXのモデルサイズ注記**: WhisperXはfaster-whisperバックエンドを使用するため、同じモデル名でも標準Whisperとは異なる最適化済みのCTranslate2形式でダウンロードされます。精度・VRAM消費はほぼ同等ですが、推論速度は大幅に向上します。

### GPU設定
- CUDA対応GPU使用時は自動でGPU加速が有効化
- サイドバーに現在の使用デバイスを表示
- GPU使用時はfp16精度で高速化
- 話者分離もGPU対応（大幅に高速化）
- ReazonSpeechはCPU実行（sherpa-onnx-cudaで別途GPU対応可）

### 話者分離のパフォーマンス

| 音声長 | CPU処理時間 | GPU処理時間 |
|--------|-------------|-------------|
| 1分    | ~30秒       | ~5秒        |
| 5分    | ~2.5分      | ~20秒       |
| 10分   | ~5分        | ~40秒       |
| 30分   | ~15分       | ~2分        |

※実際の処理時間はハードウェアや音声内容により変動します

## 🔧 技術仕様

- **フレームワーク**: Streamlit
- **音声認識**:
  - OpenAI Whisper（openai-whisper）
  - WhisperX（faster-whisper + CTranslate2）
  - ReazonSpeech v2.0（reazonspeech-k2-asr 3.0.0 / sherpa-onnx）
- **深層学習**: PyTorch 2.8.0+cu128
- **話者分離**: pyannote.audio 4.0.4
  - Pipeline: pyannote/speaker-diarization-3.1
  - Segmentation: pyannote/segmentation-3.0
  - Community: pyannote/speaker-diarization-community-1
  - Embedding: pyannote/wespeaker-voxceleb-resnet34-LM
- **音声読み込み**: librosa（torchcodec回避のためテンソル事前変換）, soundfile
- **対応フォーマット**: MP3, WAV, M4A, OGG, FLAC
- **最大ファイルサイズ**: 1GB
- **長時間音声処理**: 10分超の音声は8分ごとに自動分割

## ⚠️ 注意事項

### 一般
- 初回実行時はWhisperモデルのダウンロードが行われます（~1.5GB for large-v3）
- largeモデルは高精度ですが処理時間が長くなります
- GPU使用には十分なVRAMが必要です（large: ~10GB、medium: ~5GB）
- 長時間の音声ファイルは大量のメモリを使用する可能性があります

### 話者分離機能
- **Hugging Face Access Tokenが必須**です（Classic / Readトークンを推奨）
- **4つのモデル全ての利用規約に同意が必要**です（下記トラブルシューティング参照）
- アプリ再起動時にトークンはリセットされるため、再入力が必要です
- 初回実行時にpyannoteモデルがダウンロードされます（合計~500MB）
- GPUを使用すると大幅に高速化されます
- 話者数が多い（5人以上）場合、精度が低下する可能性があります
- 背景音が大きい場合、話者分離の精度が低下します
- 音声の重なり（同時発話）は1人としてカウントされる場合があります

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
```bash
pip install torch==2.8.0+cu128 torchaudio==2.8.0+cu128 --index-url https://download.pytorch.org/whl/cu128 --force-reinstall
```

### WhisperXインストール後にGPUが使えなくなった
→ whisperxインストールがtorch CPU版に上書きする場合があります。以下で再インストール：
```bash
pip install torch==2.8.0+cu128 torchaudio==2.8.0+cu128 --index-url https://download.pytorch.org/whl/cu128 --force-reinstall
```

### メモリエラー
→ より小さなモデル（small、base）を使用するか、音声ファイルを分割してください

### 話者分離 403エラー
```
Access to model XXX is restricted
```
→ 以下の**全4モデル**の利用規約に同意してください（いずれか1つでも未同意だと403になります）：
- https://huggingface.co/pyannote/speaker-diarization-3.1
- https://huggingface.co/pyannote/segmentation-3.0
- https://huggingface.co/pyannote/speaker-diarization-community-1

→ トークンは**Classic / Read**タイプを使用してください。Fine-grainedトークンは権限設定が複雑で403になる場合があります。

### 話者分離でモデルが見つからないエラー
```
An error happened while trying to locate the file on the Hub
```
→ アプリ再起動後はHFトークンがリセットされます。サイドバーのトークン欄に再入力してください。

### torchcodec / AudioDecoderエラー
```
name 'AudioDecoder' is not defined
```
→ torchcodecのDLL問題です。アプリ内部でlibrosによる音声テンソル変換を行っているため、通常は自動回避されます。問題が続く場合はtorchcodecを再インストールしてください。

### ReazonSpeechが表示されない
→ 別途インストールが必要です：
```bash
pip install git+https://github.com/reazon-research/reazonspeech.git#subdirectory=pkg/k2-asr
```

## 📝 ライセンス

このプロジェクトはOpenAI Whisperを使用しており、MITライセンスの下で公開されています。

## 🤝 貢献

バグ報告や機能要求は、GitHubのIssuesページでお知らせください。
