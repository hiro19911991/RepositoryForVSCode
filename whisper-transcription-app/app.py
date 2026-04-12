#!/usr/bin/env python3
"""
Whisper文字起こしWebアプリ（Streamlit使用）
"""

import os
import sys
import time
import tempfile
import inspect
import whisper
import torch
import streamlit as st
from datetime import datetime
import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from pyannote.audio import Pipeline
<<<<<<< ours
<<<<<<< ours
<<<<<<< ours

try:
    from pyannote.audio import Pipeline as PyannotePipeline
    from pyannote.audio.pipelines.utils.hook import ProgressHook
    PYANNOTE_AVAILABLE = True
except ImportError:
    PyannotePipeline = None
    ProgressHook = None
    PYANNOTE_AVAILABLE = False

# ReazonSpeech関連のインポート（オプション）
try:
    from reazonspeech.k2.asr import load_model as load_reazonspeech_model, transcribe as reazonspeech_transcribe, audio_from_path
    REAZONSPEECH_AVAILABLE = True
    print("ReazonSpeechが利用可能です。")
except ImportError:
    REAZONSPEECH_AVAILABLE = False
    print("ReazonSpeechが利用できません。Whisperのみを使用します。")

# Voxtral関連のインポート（オプション）
try:
    from vllm import LLM
    import mistral_common
    VOXTRAL_AVAILABLE = True
    print("✅ Voxtralが利用可能です。")
    print(f"   vLLM version: {getattr(__import__('vllm'), '__version__', 'unknown')}")
    print(f"   mistral_common available: {hasattr(mistral_common, '__version__')}")
except ImportError as e:
    VOXTRAL_AVAILABLE = False
    print("❌ Voxtralが利用できません。詳細:")
    print(f"   エラー: {str(e)}")
    print("   vLLMとmistral-commonが必要です。")
    try:
        import vllm
        print("   vLLMは利用可能")
    except ImportError:
        print("   vLLMが見つかりません")
    try:
        import mistral_common
        print("   mistral_commonは利用可能")
    except ImportError:
        print("   mistral_commonが見つかりません")

# 日本語最適化Whisperモデルの定義
VOXTRAL_IMPLEMENTED = False
REPOSITORY_URL = ""
DEFAULT_PYANNOTE_MODEL = "pyannote/speaker-diarization-community-1"

JAPANESE_OPTIMIZED_MODELS = {
    "large-v3 (日本語最適化)": "large-v3",
    "large-v2 (日本語最適化)": "large-v2", 
    "medium (日本語最適化)": "medium",
    "small (日本語最適化)": "small"
}
=======
>>>>>>> theirs
=======
>>>>>>> theirs
=======
>>>>>>> theirs

# ページ設定
st.set_page_config(
    page_title="Whisper文字起こしツール",
    page_icon="🎤",
    layout="wide"
)

# キャッシュ設定（モデルを再ロードしないようにする）
@st.cache_resource
def load_whisper_model(model_name):
    """Whisperモデルをロードする（キャッシュ使用）"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return whisper.load_model(model_name, device=device)

@st.cache_resource
<<<<<<< ours
<<<<<<< ours
<<<<<<< ours
<<<<<<< ours
def load_reazonspeech_model_cached():
    """ReazonSpeechモデルをロードする（キャッシュ使用）"""
    if not REAZONSPEECH_AVAILABLE:
        raise ImportError("ReazonSpeechが利用できません")
    # 公式APIに従って修正
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return load_reazonspeech_model(device=device)

@st.cache_resource
def load_pyannote_pipeline(model_id, hf_token):
    """Load pyannote speaker diarization pipeline."""
    if not PYANNOTE_AVAILABLE:
        raise ImportError("pyannote.audio is not installed.")
    if not hf_token:
        raise ValueError("Hugging Face token is required for speaker diarization.")

    pipeline = PyannotePipeline.from_pretrained(model_id, token=hf_token)
=======
=======
>>>>>>> theirs
def load_diarization_pipeline(hf_token):
    """話者分離パイプラインをロードする（キャッシュ使用）"""
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization@2.1",
        use_auth_token=hf_token
    )
<<<<<<< ours
>>>>>>> theirs
=======
>>>>>>> theirs
=======
=======
>>>>>>> theirs
def load_diarization_pipeline(hf_token):
    """話者分離パイプラインをロードする（キャッシュ使用）"""
    from_pretrained_params = inspect.signature(Pipeline.from_pretrained).parameters
    auth_kwargs = {"token": hf_token} if "token" in from_pretrained_params else {"use_auth_token": hf_token}

    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", **auth_kwargs)
<<<<<<< ours
>>>>>>> theirs
=======
>>>>>>> theirs
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
    return pipeline

<<<<<<< ours
<<<<<<< ours
<<<<<<< ours
<<<<<<< ours
@st.cache_resource
def load_voxtral_model():
    """Voxtralモデルをロードする（キャッシュ使用）"""
    if not VOXTRAL_AVAILABLE:
        raise ImportError("Voxtralが利用できません")
    
    # GPU メモリ確認
    if torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if gpu_memory_gb < 15:
            st.warning(f"⚠️ GPU メモリが {gpu_memory_gb:.1f}GB しかありません。Voxtralには16GB以上が推奨されます。")
    else:
        st.error("❌ Voxtralの使用にはGPUが必要です。")
        raise RuntimeError("VoxtralにはGPUが必要です")
    
    # vLLMでVoxtralモデルをロード
    model = LLM(
        model="mistralai/Voxtral-Mini-4B-Realtime-2602",
        trust_remote_code=True,
        max_model_len=131072,  # ~ca. 3h for recommended settings
        tensor_parallel_size=1,
        dtype="bfloat16",  # BF16フォーマット
        gpu_memory_utilization=0.9,
    )
    return model

=======
>>>>>>> theirs
=======
>>>>>>> theirs
=======
>>>>>>> theirs
=======
>>>>>>> theirs
def check_ffmpeg():
    """FFmpegがインストールされているか確認"""
    if os.system("ffmpeg -version > nul 2>&1") != 0:
        st.error("⚠️ FFmpegがインストールされていません。https://ffmpeg.org/download.html からダウンロードしてください。")
        st.stop()

def get_available_models():
    """利用可能なWhisperモデルの一覧を返す"""
    return [
        "tiny", "tiny.en", 
        "base", "base.en", 
        "small", "small.en", 
        "medium", "medium.en", 
        "large", 
        "turbo"
    ]

def split_audio_into_chunks(audio_path, chunk_duration_seconds=480):
    """
    音声ファイルを指定された長さのチャンクに分割する
    
    Args:
        audio_path: 音声ファイルのパス
        chunk_duration_seconds: チャンクの長さ（秒）デフォルト8分（480秒）
    
    Returns:
        list: 分割されたチャンクファイルのパスのリスト
    """
    try:
        # pydubを使用して音声ファイルを読み込み
        audio = AudioSegment.from_file(audio_path)
        total_duration_ms = len(audio)
        chunk_duration_ms = chunk_duration_seconds * 1000
        
        chunk_files = []
        chunk_count = 0
        
        for start_time in range(0, total_duration_ms, chunk_duration_ms):
            end_time = min(start_time + chunk_duration_ms, total_duration_ms)
            chunk = audio[start_time:end_time]
            
            # チャンクファイルの保存
            base_name = os.path.splitext(audio_path)[0]
            chunk_filename = f"{base_name}_chunk_{chunk_count:03d}.wav"
            chunk.export(chunk_filename, format="wav")
            
            chunk_files.append(chunk_filename)
            chunk_count += 1
            
        return chunk_files
    
    except Exception as e:
        st.error(f"音声分割エラー: {str(e)}")
        return []

def transcribe_chunks(model, chunk_files, options, progress_callback=None, chunk_duration_seconds=480):
    """
    複数のチャンクを順次転写処理する
    
    Args:
        model: Whisperモデル
        chunk_files: チャンクファイルのパスのリスト
        options: 転写オプション
        progress_callback: 進捗コールバック関数
        chunk_duration_seconds: チャンクの長さ（秒）
    
    Returns:
        dict: 統合された転写結果
    """
    all_segments = []
    full_text = ""
    time_offset = 0.0
    detected_language = "unknown"
    
    for i, chunk_file in enumerate(chunk_files):
        chunk_duration = get_audio_duration(chunk_file) or chunk_duration_seconds
        try:
            if progress_callback:
                progress_callback(f"チャンク {i+1}/{len(chunk_files)} を処理中...")
            
            # ハルシネーション対策の強化されたオプション
            chunk_options = options.copy()
            chunk_options.update({
                "condition_on_previous_text": False,  # 前のテキストに依存しない
                "compression_ratio_threshold": 2.4,  # 圧縮率でハルシネーション検出
                "logprob_threshold": -1.0,  # 確率でハルシネーション検出
                "no_speech_threshold": 0.6,  # 無音検出の閾値
                "temperature": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],  # 温度を段階的に上げる
            })
            
            # チャンクを転写
            result = model.transcribe(chunk_file, **chunk_options)
            
            # 言語検出（最初のチャンクから）
            if i == 0 and result.get("language"):
                detected_language = result["language"]
            
            # テキストが有効かチェック（ハルシネーション検出）
            text = result["text"].strip()
            if text and not is_hallucination(text):
                # タイムスタンプを調整して追加
                for segment in result["segments"]:
                    adjusted_segment = segment.copy()
                    adjusted_segment["start"] += time_offset
                    adjusted_segment["end"] += time_offset
                    all_segments.append(adjusted_segment)
                
                # テキストを結合
                if full_text and not full_text.endswith(" "):
                    full_text += " "
                full_text += text
            else:
                st.warning(f"チャンク {i+1} はハルシネーションまたは無音と判定されました。")
            
            # 次のチャンクのためのオフセット計算（固定時間を使用）
            time_offset += chunk_duration
                
        except Exception as e:
            st.warning(f"チャンク {i+1} の処理でエラーが発生しました: {str(e)}")
            time_offset += chunk_duration  # Keep timestamps aligned even after chunk failures.
            continue
        finally:
            # チャンクファイルを削除
            if os.path.exists(chunk_file):
                try:
                    os.unlink(chunk_file)
                except:
                    pass
    
    # 統合結果を返す
    return {
        "text": full_text.strip(),
        "segments": all_segments,
        "language": detected_language
    }

def is_hallucination(text):
    """
    ハルシネーション（幻覚）テキストかどうかを判定する
    
    Args:
        text: 判定するテキスト
    
    Returns:
        bool: ハルシネーションの場合True
    """
    if not text or len(text.strip()) < 5:
        return True
    
    # 同じフレーズの繰り返しを検出
    words = text.split()
    if len(words) < 3:
        return False
    
    # 短いフレーズ（2-4語）の繰り返しを検出
    for phrase_length in range(2, min(5, len(words) // 3 + 1)):
        for i in range(len(words) - phrase_length * 3 + 1):
            phrase = words[i:i + phrase_length]
            phrase_str = " ".join(phrase)
            
            # 同じフレーズが3回以上連続で現れるかチェック
            remaining_text = " ".join(words[i:])
            phrase_count = remaining_text.count(phrase_str)
            
            if phrase_count >= 3:
                # フレーズの繰り返しがテキストの大部分を占める場合
                total_repeated_length = phrase_count * len(phrase_str)
                if total_repeated_length > len(text) * 0.7:
                    return True
    
    return False

def transcribe_chunks_reazonspeech(model, chunk_files, progress_callback=None, chunk_duration_seconds=480):
    """
    ReazonSpeechで複数のチャンクを順次転写処理する
    
    Args:
        model: ReazonSpeechモデル
        chunk_files: チャンクファイルのパスのリスト
        progress_callback: 進捗コールバック関数
        chunk_duration_seconds: チャンクの長さ（秒）
    
    Returns:
        dict: 統合された転写結果
    """
    import gc
    all_segments = []
    full_text = ""
    time_offset = 0.0
    
    for i, chunk_file in enumerate(chunk_files):
        chunk_duration = get_audio_duration(chunk_file) or chunk_duration_seconds
        audio = None
        result = None
        try:
            if progress_callback:
                progress_callback(f"チャンク {i+1}/{len(chunk_files)} を処理中...")
            
            # ReazonSpeechで転写
            audio = audio_from_path(chunk_file)
            result = reazonspeech_transcribe(model, audio)
            
            # テキストを結合
            text = result.text.strip()
            if text and not is_hallucination(text):
                # セグメントを調整して追加（ReazonSpeechの場合）
                for segment in result.segments:
                    adjusted_segment = {
                        "start": segment.start_seconds + time_offset,
                        "end": segment.end_seconds + time_offset,
                        "text": segment.text
                    }
                    all_segments.append(adjusted_segment)
                
                # テキストを結合
                if full_text and not full_text.endswith(" "):
                    full_text += " "
                full_text += text
            else:
                st.warning(f"チャンク {i+1} はハルシネーションまたは無音と判定されました。")
            
            # 次のチャンクのためのオフセット計算（固定時間を使用）
            time_offset += chunk_duration
                
        except Exception as e:
            st.warning(f"チャンク {i+1} の処理でエラーが発生しました: {str(e)}")
            time_offset += chunk_duration  # Keep timestamps aligned even after chunk failures.
            continue
        finally:
            # チャンクファイルを削除
            if os.path.exists(chunk_file):
                try:
                    os.unlink(chunk_file)
                except:
                    pass
            # メモリクリア
            if audio is not None:
                del audio
            if result is not None:
                del result
            gc.collect()
    
    # 統合結果を返す
    return {
        "text": full_text.strip(),
        "segments": all_segments,
        "language": "ja"  # ReazonSpeechは日本語専用
    }

def transcribe_with_voxtral(model, audio_path, delay_ms=480):
    """Voxtral is not ready for production transcription yet."""
    raise NotImplementedError("Voxtral Realtime is not implemented yet.")

def get_audio_duration(audio_path):
    """
    音声ファイルの長さを秒単位で取得する
    
    Args:
        audio_path: 音声ファイルのパス
    
    Returns:
        float: 音声の長さ（秒）
    """
    try:
        audio = AudioSegment.from_file(audio_path)
        return len(audio) / 1000.0  # ミリ秒を秒に変換
    except Exception as e:
        st.error(f"音声ファイルの長さ取得エラー: {str(e)}")
        return 0.0

<<<<<<< ours
<<<<<<< ours
<<<<<<< ours
<<<<<<< ours
def diarize_audio(audio_path, pipeline, progress_bar=None, num_speakers=None, min_speakers=None, max_speakers=None):
    """Run speaker diarization on an audio file."""
    diarize_kwargs = {}
    if num_speakers:
        diarize_kwargs["num_speakers"] = int(num_speakers)
    else:
        if min_speakers:
            diarize_kwargs["min_speakers"] = int(min_speakers)
        if max_speakers:
            diarize_kwargs["max_speakers"] = int(max_speakers)

    if ProgressHook is not None and progress_bar is not None:
        with ProgressHook() as hook:
            diarization = pipeline(audio_path, hook=hook, **diarize_kwargs)
            progress_bar.progress(1.0)
            return diarization

    return pipeline(audio_path, **diarize_kwargs)

def speaker_for_segment(start_time, end_time, diarization):
    """Pick the speaker with the largest overlap for an ASR segment."""
    best_speaker = None
    best_overlap = 0.0

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        overlap_start = max(start_time, turn.start)
        overlap_end = min(end_time, turn.end)
        overlap = max(0.0, overlap_end - overlap_start)
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = speaker

    return best_speaker or "SPEAKER_UNKNOWN"

def attach_speakers_to_segments(result, diarization):
    """Attach speaker labels to transcription segments."""
    speaker_segments = []
    diarized_lines = []

    for segment in result.get("segments", []):
        speaker = speaker_for_segment(segment["start"], segment["end"], diarization)
        enriched_segment = segment.copy()
        enriched_segment["speaker"] = speaker
        speaker_segments.append(enriched_segment)
        diarized_lines.append(f"[{speaker}] {segment['text'].strip()}")

    enriched_result = result.copy()
    enriched_result["segments"] = speaker_segments
    enriched_result["diarized_text"] = "\n".join(diarized_lines).strip()
    return enriched_result
=======
=======
>>>>>>> theirs
=======
>>>>>>> theirs
=======
>>>>>>> theirs
def assign_speakers_to_segments(whisper_segments, diarization):
    """
    Whisperセグメントに話者ラベルを付与する

    Args:
        whisper_segments: Whisperのセグメント一覧
        diarization: pyannoteの話者分離結果

    Returns:
        list: 話者ラベル付きセグメント一覧
    """
    speaker_segments = []

    diarization_tracks = list(diarization.itertracks(yield_label=True))

    for segment in whisper_segments:
        segment_start = segment["start"]
        segment_end = segment["end"]
        speaker_overlaps = {}

        for turn, _, speaker in diarization_tracks:
            overlap_start = max(segment_start, turn.start)
            overlap_end = min(segment_end, turn.end)
            overlap_duration = overlap_end - overlap_start

            if overlap_duration > 0:
                speaker_overlaps[speaker] = speaker_overlaps.get(speaker, 0.0) + overlap_duration

        assigned_speaker = max(speaker_overlaps, key=speaker_overlaps.get) if speaker_overlaps else "UNKNOWN"

        speaker_segments.append({
            "speaker": assigned_speaker,
            "start": segment_start,
            "end": segment_end,
            "text": segment["text"].strip()
        })

    return speaker_segments
<<<<<<< ours
<<<<<<< ours
<<<<<<< ours
>>>>>>> theirs
=======
>>>>>>> theirs
=======
=======
>>>>>>> theirs

def normalize_diarization_output(diarization_output):
    """
    pyannoteのバージョン差異を吸収して Annotation を取り出す
    """
    if hasattr(diarization_output, "speaker_diarization"):
        return diarization_output.speaker_diarization
    if hasattr(diarization_output, "diarization"):
        return diarization_output.diarization
    return diarization_output
<<<<<<< ours
>>>>>>> theirs
=======
>>>>>>> theirs

def main():
    """メイン関数"""
    st.title("🎤 Whisper文字起こしツール")
    st.markdown("""
    OpenAIのWhisperモデルを使用して、音声ファイルからテキストへの文字起こしを行います。
    """)
    
    # FFmpegの確認
    check_ffmpeg()
    
    # サイドバー設定
    st.sidebar.title("設定")
    
    # ASRエンジン選択
    available_engines = ["Whisper (標準)", "Whisper (日本語特化)"]
    if VOXTRAL_AVAILABLE and VOXTRAL_IMPLEMENTED:
        available_engines.append("Voxtral Realtime")
    if REAZONSPEECH_AVAILABLE:
        available_engines.append("ReazonSpeech v2.0")
    
    engine_option = st.sidebar.selectbox(
        "ASRエンジンを選択",
        options=available_engines,
        help="使用する音声認識エンジンを選択してください。日本語特化版は日本語音声により適した設定です。Voxtral Realtimeは超低遅延リアルタイム音声認識です。"
    )
    
    # ReazonSpeechが利用できない場合の説明
    if not REAZONSPEECH_AVAILABLE:
        st.sidebar.info("💡 高精度な日本語音声認識には「Whisper (日本語特化)」をお選びください。")
    
    # モデル選択（Whisperの場合のみ表示）
    if "Whisper" in engine_option:
        if engine_option == "Whisper (日本語特化)":
            # 日本語特化の場合は推奨モデルから選択
            model_option = st.sidebar.selectbox(
                "日本語特化Whisperモデルを選択",
                options=["large-v3", "large-v2", "large", "medium", "small"],
                index=0,  # large-v3をデフォルトに（日本語に最適）
                help="日本語音声認識に最適化されたモデル設定。large-v3が最も高精度です。"
            )
        else:
            # 標準Whisperの場合
            model_option = st.sidebar.selectbox(
                "Whisperモデルサイズを選択",
                options=get_available_models(),
                index=9,  # turboをデフォルトに（最新の最適化モデル）
                help="モデルの詳細: turbo(最新最適化)、.enは英語専用で高精度、largeは最高精度"
            )
    else:
        model_option = None
    
    # 言語選択
    language_option = st.sidebar.selectbox(
        "言語を選択（自動検出する場合は空欄）",
        options=["", "en", "ja", "zh", "de", "fr", "es", "ko", "ru"],
        index=2,  # 日本語をデフォルトに
        format_func=lambda x: {
            "": "自動検出", "en": "英語", "ja": "日本語", "zh": "中国語",
            "de": "ドイツ語", "fr": "フランス語", "es": "スペイン語",
            "ko": "韓国語", "ru": "ロシア語"
        }.get(x, x),
        help="音声の言語を指定します。自動検出も可能です。"
    )
    
    # デバイス情報表示
    device = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
    st.sidebar.info(f"使用デバイス: {device}")
    
    if device == "CPU":
        st.sidebar.warning("GPUが検出されませんでした。処理が遅くなる可能性があります。")

    # 話者分離設定
    st.sidebar.markdown("### 話者分離（pyannote）")
    use_diarization = st.sidebar.checkbox(
        "話者分離を有効化",
        value=False,
        help="Hugging Faceのpyannote/speaker-diarizationを使って話者を自動識別します。"
    )
    hf_token = ""
    diarization_num_speakers = 0
    diarization_min_speakers = 1
    diarization_max_speakers = 8

    if use_diarization:
        hf_token = st.sidebar.text_input(
            "Hugging Face Access Token",
            type="password",
            help="pyannote/speaker-diarization と pyannote/segmentation の利用規約同意済みトークンを設定してください。"
        ).strip()

        diarization_num_speakers = st.sidebar.number_input(
            "話者数（0で自動推定）",
            min_value=0,
            max_value=20,
            value=0,
            step=1
        )
        diarization_min_speakers = st.sidebar.number_input(
            "最小話者数",
            min_value=1,
            max_value=20,
            value=1,
            step=1
        )
        diarization_max_speakers = st.sidebar.number_input(
            "最大話者数",
            min_value=1,
            max_value=20,
            value=8,
            step=1
        )
    
    # サイドバーにGitHubリンク
    if REPOSITORY_URL:
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"[GitHub Repository]({REPOSITORY_URL})")
    # ファイルアップロード
    uploaded_file = st.file_uploader("音声ファイルをアップロード", 
                                    type=["mp3", "wav", "m4a", "ogg", "flac"],
                                    help="対応フォーマット: MP3, WAV, M4A, OGG, FLAC（最大1GB）")
    
    if uploaded_file is not None:
        # ファイル情報表示
        file_size_mb = uploaded_file.size / (1024 * 1024)
        st.info(f"ファイル: {uploaded_file.name} ({file_size_mb:.2f} MB)")
        
        # 音声再生機能
        st.audio(uploaded_file, format=f"audio/{uploaded_file.name.split('.')[-1]}")
        
        # 文字起こし実行ボタン
        transcribe_button = st.button("文字起こし開始", type="primary")
        
        if transcribe_button:
            # 処理開始
            with st.spinner("文字起こし処理中..."):
                # 一時ファイルとして保存
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    temp_filename = tmp_file.name
                
                try:
                    # ファイルサイズ確認
                    file_size = os.path.getsize(temp_filename)
                    if file_size < 1024:  # 1KB未満
                        st.error("音声ファイルが小さすぎます。有効な音声ファイルをアップロードしてください。")
                        return
                    
                    # 音声ファイルの基本情報確認
                    duration = get_audio_duration(temp_filename)
                    if duration == 0:
                        st.error("音声ファイルの読み込みに失敗しました。別のファイルを試してください。")
                        return
                    
                    st.info(f"音声ファイル情報: 長さ {duration:.1f}秒 ({duration/60:.1f}分)")
                    
                    if duration < 0.5:  # 0.5秒未満
                        st.error("音声ファイルが短すぎます（0.5秒未満）。より長い音声ファイルをアップロードしてください。")
                        return
                    
                    # 10分（600秒）以上の場合はチャンク処理を実行
                    use_chunking = duration > 600
                    if use_chunking:
                        st.warning(f"⚠️ 音声ファイルが10分を超えています（{duration/60:.1f}分）。ハルシネーション防止のため、8分ごとのチャンクに分割して処理します。")
                    
                    # モデルロード
                    load_start = time.time()
                    progress_text = st.empty()
                    progress_text.text("モデルをロード中...")
                    
                    if "Whisper" in engine_option:
                        model = load_whisper_model(model_option)
                    elif engine_option == "Voxtral Realtime":
                        model = load_voxtral_model()
                    else:  # ReazonSpeech v2.0
                        model = load_reazonspeech_model_cached()
                    
                    load_end = time.time()
                    progress_text.text(f"モデルロード完了（{load_end - load_start:.2f}秒）")
                    
                    # 文字起こし処理
                    progress_text.text("文字起こし処理中...")
                    transcribe_start = time.time()
                    
                    # 言語オプション設定
                    options = {}
                    
                    # 日本語特化モードの場合の特別設定
                    if engine_option == "Whisper (日本語特化)":
                        options.update({
                            "language": "ja",  # 日本語を強制指定
                            "task": "transcribe",  # 転写タスクを明示
                            "initial_prompt": "以下は日本語の音声です。正確に文字起こししてください。",  # 日本語プロンプト
                            "word_timestamps": True,  # 単語レベルのタイムスタンプ
                        })
                        st.info("🇯🇵 日本語特化モードで処理します。より高精度な日本語認識を行います。")
                    else:
                        # 標準モードまたはReazonSpeech
                        if language_option:
                            options["language"] = language_option
                    
                    # GPU最適化とエラー対策
                    options["verbose"] = False
                    options["fp16"] = torch.cuda.is_available()  # GPUの場合はfp16を使用
                    options["temperature"] = 0  # より安定した結果を得る
                    options["beam_size"] = 1 if torch.cuda.is_available() else 5  # GPU使用時は高速化
                        
                    # 文字起こし実行（チャンク処理の条件分岐）
                    if use_chunking:
                        # 長い音声ファイルの場合はチャンクに分割して処理
                        progress_text.text("音声を8分間のチャンクに分割中...")
                        chunk_files = split_audio_into_chunks(temp_filename, chunk_duration_seconds=480)  # 8分 = 480秒
                        
                        if not chunk_files:
                            st.error("音声分割に失敗しました。")
                            return
                        
                        st.info(f"音声を{len(chunk_files)}個のチャンクに分割しました。順次処理を開始します。")
                        
                        # プログレスバーの追加
                        progress_bar = st.progress(0)
                        
                        def progress_callback(message):
                            progress_text.text(message)
                            # プログレスバーの更新（概算）
                            current_chunk = int(message.split()[1].split('/')[0]) if 'チャンク' in message else 1
                            total_chunks = len(chunk_files)
                            progress_bar.progress(current_chunk / total_chunks)
                        
                        # チャンク単位で転写処理
                        if "Whisper" in engine_option:
                            result = transcribe_chunks(model, chunk_files, options, progress_callback, chunk_duration_seconds=480)
                        elif engine_option == "Voxtral Realtime":
                            # Voxtralは現在チャンク処理をサポートしていないため、警告を表示
                            st.warning("⚠️ Voxtral Realtimeは現在長時間音声のチャンク処理をサポートしていません。全体を一括処理します。")
                            result = transcribe_with_voxtral(model, temp_filename)
                        else:  # ReazonSpeech v2.0
                            result = transcribe_chunks_reazonspeech(model, chunk_files, progress_callback, chunk_duration_seconds=480)
                        progress_bar.progress(1.0)  # 完了
                        
                    else:
                        # 通常の処理（10分未満の場合）
<<<<<<< ours
                        if "Whisper" in engine_option:
                            result = model.transcribe(temp_filename, **options)
                        elif engine_option == "Voxtral Realtime":
                            st.info("🚀 Voxtral Realtimeモードで処理します。超低遅延音声認識を実行します。")
                            result = transcribe_with_voxtral(model, temp_filename)
                        else:  # ReazonSpeech v2.0
                            audio = audio_from_path(temp_filename)
                            result = reazonspeech_transcribe(model, audio)
                            # ReazonSpeechの結果をWhisper形式に変換
                            result = {
                                "text": result.text,
                                "segments": [{"start": seg.start_seconds, "end": seg.end_seconds, "text": seg.text} for seg in result.segments],
                                "language": "ja"
                            }
=======
                        result = model.transcribe(temp_filename, **options)

                    # 話者分離処理
                    speaker_segments = []
                    if use_diarization:
                        if not hf_token:
                            st.warning("話者分離が有効ですが、Hugging Faceトークンが未設定のためスキップします。")
                        elif diarization_min_speakers > diarization_max_speakers:
                            st.warning("話者分離設定が不正です（最小話者数 > 最大話者数）。話者分離をスキップします。")
                        else:
                            progress_text.text("話者分離処理中...")
                            diarization_pipeline = load_diarization_pipeline(hf_token)

                            diarization_options = {}
                            if diarization_num_speakers > 0:
                                diarization_options["num_speakers"] = diarization_num_speakers
                            else:
                                diarization_options["min_speakers"] = diarization_min_speakers
                                diarization_options["max_speakers"] = diarization_max_speakers

<<<<<<< ours
<<<<<<< ours
                            diarization = diarization_pipeline(temp_filename, **diarization_options)
                            speaker_segments = assign_speakers_to_segments(result["segments"], diarization)
<<<<<<< ours
>>>>>>> theirs
=======
>>>>>>> theirs
=======
                            diarization_output = diarization_pipeline(temp_filename, **diarization_options)
                            diarization = normalize_diarization_output(diarization_output)
                            speaker_segments = assign_speakers_to_segments(result["segments"], diarization)
>>>>>>> theirs
=======
                            diarization_output = diarization_pipeline(temp_filename, **diarization_options)
                            diarization = normalize_diarization_output(diarization_output)
                            speaker_segments = assign_speakers_to_segments(result["segments"], diarization)
>>>>>>> theirs
                    
                    transcribe_end = time.time()
                    progress_text.empty()
                    
                    # 処理時間計算
                    transcribe_time = transcribe_end - transcribe_start
                    total_time = transcribe_end - load_start
                    
                    # 結果表示
                    st.markdown("### 文字起こし結果")
                    st.success(f"処理完了（文字起こし: {transcribe_time:.2f}秒、合計: {total_time:.2f}秒）")
                    
                    # テキスト結果表示
                    st.markdown("#### テキスト")
                    st.text_area("文字起こし結果", value=result["text"], height=200)
                    
                    # ダウンロードボタン
                    st.download_button(
                        label="テキストをダウンロード",
                        data=result["text"],
                        file_name=f"{os.path.splitext(uploaded_file.name)[0]}_transcript.txt",
                        mime="text/plain"
                    )

                    # 話者付きテキスト表示
                    if speaker_segments:
                        st.markdown("#### 話者付きテキスト")
                        speaker_text = ""
                        for speaker_segment in speaker_segments:
                            speaker_text += f"[{speaker_segment['speaker']}] {speaker_segment['text']}\n"
                        st.text_area("話者付き文字起こし結果", value=speaker_text, height=220)
                        st.download_button(
                            label="話者付きテキストをダウンロード",
                            data=speaker_text,
                            file_name=f"{os.path.splitext(uploaded_file.name)[0]}_transcript_speakers.txt",
                            mime="text/plain"
                        )
                    
                    # タイムスタンプ付きの詳細結果
                    with st.expander("詳細（タイムスタンプ付き）"):
                        # テーブル表示用のデータ準備
                        table_data = []
                        timestamp_text = ""
                        
                        for idx, segment in enumerate(result["segments"]):
                            start_time = segment["start"]
                            end_time = segment["end"]
                            text = segment["text"]
                            speaker = speaker_segments[idx]["speaker"] if idx < len(speaker_segments) else "-"
                            
                            # 時間をフォーマット (HH:MM:SS.ms)
                            start_formatted = str(datetime.utcfromtimestamp(start_time).strftime('%H:%M:%S.%f'))[:-3]
                            end_formatted = str(datetime.utcfromtimestamp(end_time).strftime('%H:%M:%S.%f'))[:-3]
                            
                            table_data.append({
                                "開始": start_formatted,
                                "終了": end_formatted,
                                "話者": speaker,
                                "テキスト": text
                            })
                            
                            timestamp_text += f"[{start_formatted} --> {end_formatted}] [{speaker}] {text}\n"
                        
                        # テーブル表示
                        st.table(table_data)
                        
                        # タイムスタンプ付きテキストのダウンロードボタン
                        st.download_button(
                            label="タイムスタンプ付きテキストをダウンロード",
                            data=timestamp_text,
                            file_name=f"{os.path.splitext(uploaded_file.name)[0]}_transcript_timestamps.txt",
                            mime="text/plain"
                        )
                
                except Exception as e:
                    st.error(f"エラーが発生しました: {str(e)}")
                
                finally:
                    # 一時ファイルの削除
                    if os.path.exists(temp_filename):
                        os.unlink(temp_filename)
    
    else:
        # ファイルがアップロードされていない場合の表示
        st.info("👆 音声ファイルをアップロードしてください")
        
        # サンプル説明
        with st.expander("使い方"):
            st.markdown("""
            1. サイドバーでモデルサイズと言語を選択
            2. 音声ファイルをアップロード
            3. 「文字起こし開始」ボタンをクリック
            4. 結果を確認し、必要に応じてダウンロード
            
            **長時間音声の自動チャンク分割機能:**
            - 10分を超える音声ファイルは自動的に8分間のチャンクに分割されます
            - これによりハルシネーション（幻覚）を防ぎ、より正確な文字起こしを実現します
            - 各チャンクは順次処理され、結果は自動的に統合されます
            - タイムスタンプも適切に調整されます
            
            **モデルサイズについて:**
            - **turbo**: 最新最適化モデル（推奨）- 高精度&高速
            - **large**: 最高精度（処理時間が長い）
            - **medium(.en)**: 高精度 - VRAM ~5GB必要
            - **small(.en)**: 中程度の精度 - バランス型
            - **base(.en)**: 軽量・高速 - 日常用途に適している
            - **tiny(.en)**: 最小・最速（低精度）- リアルタイム処理用
            
            **.en モデル**: 英語音声専用で、該当するサイズの多言語版より高精度
            
            **VRAM使用量目安:**
            - tiny/base: ~1GB | small: ~2GB | medium: ~5GB
            - turbo: ~6GB | large: ~10GB
            """)

if __name__ == "__main__":
    main()
