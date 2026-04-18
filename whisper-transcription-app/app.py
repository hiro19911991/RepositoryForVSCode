#!/usr/bin/env python3
"""
Whisper文字起こしWebアプリ（Streamlit使用）
pyannote speaker diarization統合版
"""

import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import time
import tempfile
import whisper
import torch
import streamlit as st
from datetime import datetime, timezone
import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment

# pyannote.audio関連のインポート（オプション）
try:
    from pyannote.audio import Pipeline
    PYANNOTE_AVAILABLE = True
    print("✅ pyannote.audioが利用可能です。")
except ImportError:
    Pipeline = None
    PYANNOTE_AVAILABLE = False
    print("❌ pyannote.audioが利用できません。話者分離機能は無効です。")

# ReazonSpeech関連のインポート（オプション）
try:
    from reazonspeech.k2.asr import load_model as load_reazonspeech_model, transcribe as reazonspeech_transcribe, audio_from_path
    REAZONSPEECH_AVAILABLE = True
    print("✅ ReazonSpeechが利用可能です。")
except ImportError:
    REAZONSPEECH_AVAILABLE = False
    print("❌ ReazonSpeechが利用できません。Whisperのみを使用します。")

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
    print("❌ Voxtralが利用できません。")

# WhisperX関連のインポート（オプション）
try:
    import whisperx
    WHISPERX_AVAILABLE = True
    print("✅ WhisperXが利用可能です。")
except ImportError:
    WHISPERX_AVAILABLE = False
    print("❌ WhisperXが利用できません。pip install whisperx でインストールしてください。")

# 日本語最適化Whisperモデルの定義
JAPANESE_OPTIMIZED_MODELS = {
    "large-v3 (日本語最適化)": "large-v3",
    "large-v2 (日本語最適化)": "large-v2", 
    "medium (日本語最適化)": "medium",
    "small (日本語最適化)": "small"
}

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
def load_reazonspeech_model_cached(precision='fp32'):
    """ReazonSpeechモデルをロードする（キャッシュ使用）"""
    if not REAZONSPEECH_AVAILABLE:
        raise ImportError("ReazonSpeechが利用できません")
    # sherpa-onnxベース。GPU版は sherpa-onnx-cuda が必要なため、CPUで実行
    return load_reazonspeech_model(device='cpu', precision=precision)

@st.cache_resource
def load_diarization_pipeline(hf_token):
    """話者分離パイプラインをロードする（キャッシュ使用）"""
    if not PYANNOTE_AVAILABLE:
        raise ImportError("pyannote.audioがインストールされていません")
    
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token
    )
    
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
    return pipeline

@st.cache_resource
def load_whisperx_model(model_name, device, compute_type="float16"):
    """WhisperXモデルをロードする（キャッシュ使用）"""
    if not WHISPERX_AVAILABLE:
        raise ImportError("WhisperXがインストールされていません。pip install whisperx を実行してください。")
    return whisperx.load_model(model_name, device, compute_type=compute_type)

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
    """音声ファイルを指定された長さのチャンクに分割する"""
    try:
        audio = AudioSegment.from_file(audio_path)
        total_duration_ms = len(audio)
        chunk_duration_ms = chunk_duration_seconds * 1000
        
        chunk_files = []
        chunk_count = 0
        
        for start_time in range(0, total_duration_ms, chunk_duration_ms):
            end_time = min(start_time + chunk_duration_ms, total_duration_ms)
            chunk = audio[start_time:end_time]
            
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
    """複数のチャンクを順次転写処理する"""
    all_segments = []
    full_text = ""
    time_offset = 0.0
    detected_language = "unknown"
    
    for i, chunk_file in enumerate(chunk_files):
        chunk_duration = get_audio_duration(chunk_file) or chunk_duration_seconds
        try:
            if progress_callback:
                progress_callback(f"チャンク {i+1}/{len(chunk_files)} を処理中...")
            
            chunk_options = options.copy()
            chunk_options.update({
                "condition_on_previous_text": False,
                "compression_ratio_threshold": 2.4,
                "logprob_threshold": -1.0,
                "no_speech_threshold": 0.6,
                "temperature": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            })
            
            result = model.transcribe(chunk_file, **chunk_options)
            
            if i == 0 and result.get("language"):
                detected_language = result["language"]
            
            text = result["text"].strip()
            if text and not is_hallucination(text):
                for segment in result["segments"]:
                    adjusted_segment = segment.copy()
                    adjusted_segment["start"] += time_offset
                    adjusted_segment["end"] += time_offset
                    all_segments.append(adjusted_segment)
                
                if full_text and not full_text.endswith(" "):
                    full_text += " "
                full_text += text
            else:
                st.warning(f"チャンク {i+1} はハルシネーションまたは無音と判定されました。")
            
            time_offset += chunk_duration
                
        except Exception as e:
            st.warning(f"チャンク {i+1} の処理でエラーが発生しました: {str(e)}")
            time_offset += chunk_duration
            continue
        finally:
            if os.path.exists(chunk_file):
                try:
                    os.unlink(chunk_file)
                except:
                    pass
    
    return {
        "text": full_text.strip(),
        "segments": all_segments,
        "language": detected_language
    }

def is_hallucination(text):
    """ハルシネーション（幻覚）テキストかどうかを判定する"""
    if not text or len(text.strip()) < 5:
        return True
    
    words = text.split()
    if len(words) < 3:
        return False
    
    for phrase_length in range(2, min(5, len(words) // 3 + 1)):
        for i in range(len(words) - phrase_length * 3 + 1):
            phrase = words[i:i + phrase_length]
            phrase_str = " ".join(phrase)
            
            remaining_text = " ".join(words[i:])
            phrase_count = remaining_text.count(phrase_str)
            
            if phrase_count >= 3:
                total_repeated_length = phrase_count * len(phrase_str)
                if total_repeated_length > len(text) * 0.7:
                    return True
    
    return False

def get_audio_duration(audio_path):
    """音声ファイルの長さを秒単位で取得する"""
    try:
        audio = AudioSegment.from_file(audio_path)
        return len(audio) / 1000.0
    except Exception as e:
        st.error(f"音声ファイルの長さ取得エラー: {str(e)}")
        return 0.0

def load_audio_for_pyannote(audio_path):
    """音声をpyannote用テンソルとして読み込む（torchcodec/AudioDecoder回避）"""
    waveform, sample_rate = librosa.load(audio_path, sr=None, mono=False)
    if waveform.ndim == 1:
        waveform = waveform[np.newaxis, :]  # (1, time)
    return {
        "waveform": torch.from_numpy(waveform.astype(np.float32)),
        "sample_rate": sample_rate
    }

def assign_speakers_to_segments(whisper_segments, diarization):
    """Whisperセグメントに話者ラベルを付与する"""
    speaker_segments = []

    # pyannote.audio 4.0 は DiarizeOutput を返す。Annotation を取り出す
    if hasattr(diarization, 'speaker_diarization'):
        diarization = diarization.speaker_diarization

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

def main():
    """メイン関数"""
    st.title("🎤 Whisper文字起こしツール with Speaker Diarization")
    st.markdown("""
    OpenAIのWhisperモデルを使用して、音声ファイルからテキストへの文字起こしを行います。
    pyannoteの話者分離機能により、誰が話しているかを自動識別できます。
    """)
    
    # FFmpegの確認
    check_ffmpeg()
    
    # サイドバー設定
    st.sidebar.title("設定")
    
    # ASRエンジン選択
    available_engines = ["Whisper (標準)", "Whisper (日本語特化)"]
    if WHISPERX_AVAILABLE:
        available_engines.append("WhisperX (高速・単語精度)")
    if REAZONSPEECH_AVAILABLE:
        available_engines.append("ReazonSpeech v2.0")

    engine_option = st.sidebar.selectbox(
        "ASRエンジンを選択",
        options=available_engines,
        help="使用する音声認識エンジンを選択してください。"
    )

    # モデル選択
    if engine_option in ("Whisper (標準)", "Whisper (日本語特化)"):
        if engine_option == "Whisper (日本語特化)":
            model_option = st.sidebar.selectbox(
                "日本語特化Whisperモデルを選択",
                options=["large-v3", "large-v2", "large", "medium", "small"],
                index=0,
                help="日本語音声認識に最適化されたモデル設定。large-v3が最も高精度です。"
            )
        else:
            model_option = st.sidebar.selectbox(
                "Whisperモデルサイズを選択",
                options=get_available_models(),
                index=9,
                help="モデルの詳細: turbo(最新最適化)、.enは英語専用で高精度"
            )
    elif engine_option == "WhisperX (高速・単語精度)":
        model_option = st.sidebar.selectbox(
            "WhisperXモデルを選択",
            options=["large-v3", "large-v2", "large-v1", "medium", "small", "base", "tiny"],
            index=0,
            help="large-v2/v3が最高精度。batch_sizeを下げるとVRAM消費を抑えられます。"
        )
    else:
        model_option = None

    # WhisperX専用設定
    whisperx_batch_size = 16
    whisperx_compute_type = "float16"
    whisperx_word_align = True

    if engine_option == "WhisperX (高速・単語精度)":
        st.sidebar.markdown("---")
        st.sidebar.markdown("### ⚡ WhisperX設定")
        whisperx_compute_type = st.sidebar.selectbox(
            "計算精度",
            options=["float16", "int8"],
            index=0,
            help="float16: 高精度（VRAM多め）、int8: 省メモリ（精度やや低下）。OOMが出た場合はint8を試してください。"
        )
        whisperx_batch_size = st.sidebar.slider(
            "バッチサイズ",
            min_value=1,
            max_value=32,
            value=16,
            step=1,
            help="大きいほど高速だがVRAMを多く使用します。OOMエラーが出た場合は小さくしてください。"
        )
        whisperx_word_align = st.sidebar.checkbox(
            "単語レベルのタイムスタンプ（wav2vec2）",
            value=True,
            help="wav2vec2アライメントモデルを使って単語単位の正確なタイムスタンプを生成します。"
        )

    # ReazonSpeech専用設定
    reazonspeech_precision = 'fp32'
    if engine_option == "ReazonSpeech v2.0":
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🎌 ReazonSpeech設定")
        reazonspeech_precision = st.sidebar.selectbox(
            "モデル精度",
            options=["fp32", "int8", "int8-fp32"],
            index=0,
            help="fp32: 高精度（デフォルト）、int8: 省メモリ・高速（精度やや低下）、int8-fp32: エンコーダfp32・デコーダint8（中間）"
        )

    # 言語選択
    language_option = st.sidebar.selectbox(
        "言語を選択（自動検出する場合は空欄）",
        options=["", "en", "ja", "zh", "de", "fr", "es", "ko", "ru"],
        index=2,
        format_func=lambda x: {
            "": "自動検出", "en": "英語", "ja": "日本語", "zh": "中国語",
            "de": "ドイツ語", "fr": "フランス語", "es": "スペイン語",
            "ko": "韓国語", "ru": "ロシア語"
        }.get(x, x),
        help="音声の言語を指定します。"
    )
    
    # デバイス情報表示
    device = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
    st.sidebar.info(f"使用デバイス: {device}")
    
    if device == "CPU":
        st.sidebar.warning("GPUが検出されませんでした。処理が遅くなる可能性があります。")
    
    # 話者分離設定
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎭 話者分離（Speaker Diarization）")
    
    use_diarization = st.sidebar.checkbox(
        "話者分離を有効化",
        value=False,
        help="pyannote/speaker-diarizationを使って話者を自動識別します。"
    )
    
    hf_token = ""
    diarization_num_speakers = 0
    diarization_min_speakers = 1
    diarization_max_speakers = 8
    
    if use_diarization:
        if not PYANNOTE_AVAILABLE:
            st.sidebar.error("❌ pyannote.audioがインストールされていません。")
            st.sidebar.info("インストール: pip install pyannote.audio")
            use_diarization = False
        else:
            hf_token = st.sidebar.text_input(
                "Hugging Face Access Token",
                type="password",
                help="pyannote/speaker-diarizationの利用には Hugging Face トークンが必要です。https://huggingface.co/settings/tokens"
            ).strip()
            
            diarization_num_speakers = st.sidebar.number_input(
                "話者数（0で自動推定）",
                min_value=0,
                max_value=20,
                value=0,
                step=1,
                help="話者数が分かっている場合は指定してください。"
            )
            
            if diarization_num_speakers == 0:
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
    
    # ファイルアップロード
    uploaded_file = st.file_uploader(
        "音声ファイルをアップロード", 
        type=["mp3", "wav", "m4a", "ogg", "flac"],
        help="対応フォーマット: MP3, WAV, M4A, OGG, FLAC（最大1GB）"
    )
    
    if uploaded_file is not None:
        file_size_mb = uploaded_file.size / (1024 * 1024)
        st.info(f"ファイル: {uploaded_file.name} ({file_size_mb:.2f} MB)")
        
        st.audio(uploaded_file, format=f"audio/{uploaded_file.name.split('.')[-1]}")
        
        transcribe_button = st.button("文字起こし開始", type="primary")
        
        if transcribe_button:
            with st.spinner("文字起こし処理中..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    temp_filename = tmp_file.name
                
                try:
                    file_size = os.path.getsize(temp_filename)
                    if file_size < 1024:
                        st.error("音声ファイルが小さすぎます。")
                        return
                    
                    duration = get_audio_duration(temp_filename)
                    if duration == 0:
                        st.error("音声ファイルの読み込みに失敗しました。")
                        return
                    
                    st.info(f"音声ファイル情報: 長さ {duration:.1f}秒 ({duration/60:.1f}分)")
                    
                    if duration < 0.5:
                        st.error("音声ファイルが短すぎます（0.5秒未満）。")
                        return
                    
                    use_chunking = duration > 600
                    if use_chunking:
                        st.warning(f"⚠️ 音声ファイルが10分を超えています（{duration/60:.1f}分）。8分ごとのチャンクに分割して処理します。")
                    
                    # モデルロード
                    load_start = time.time()
                    progress_text = st.empty()
                    progress_text.text("モデルをロード中...")

                    speaker_segments = []
                    device_str = "cuda" if torch.cuda.is_available() else "cpu"

                    if engine_option == "WhisperX (高速・単語精度)":
                        # === WhisperX フロー ===
                        model = load_whisperx_model(model_option, device_str, whisperx_compute_type)
                        load_end = time.time()
                        progress_text.text(f"モデルロード完了（{load_end - load_start:.2f}秒）")

                        transcribe_start = time.time()
                        progress_text.text("WhisperXで文字起こし中...")
                        audio_array = whisperx.load_audio(temp_filename)

                        # language=None はtokenizerがNoneになりクラッシュするため、事前に言語検出
                        if language_option:
                            wx_language = language_option
                        else:
                            progress_text.text("言語を自動検出中...")
                            try:
                                detect_audio = audio_array[:int(30 * 16000)] if len(audio_array) > 30 * 16000 else audio_array
                                _, detect_info = model.model.transcribe(detect_audio, beam_size=1, without_timestamps=True)
                                wx_language = detect_info.language
                                progress_text.text(f"言語を検出しました: {wx_language}")
                            except Exception:
                                wx_language = "ja"
                                st.info("言語の自動検出に失敗しました。日本語として処理します。")

                        wx_result = model.transcribe(
                            audio_array,
                            batch_size=whisperx_batch_size,
                            language=wx_language
                        )

                        # 単語アライメント
                        if whisperx_word_align:
                            progress_text.text("単語アライメント処理中（wav2vec2）...")
                            try:
                                model_a, metadata = whisperx.load_align_model(
                                    language_code=wx_result["language"],
                                    device=device_str
                                )
                                wx_result = whisperx.align(
                                    wx_result["segments"], model_a, metadata,
                                    audio_array, device_str,
                                    return_char_alignments=False
                                )
                                del model_a
                            except Exception as e:
                                st.warning(f"⚠️ 単語アライメントに失敗しました（スキップ）: {str(e)}")

                        result = {
                            "text": " ".join(seg["text"].strip() for seg in wx_result["segments"]),
                            "segments": wx_result["segments"],
                            "language": wx_result.get("language", "unknown")
                        }

                        # WhisperX話者分離
                        if use_diarization and not hf_token:
                            st.warning("⚠️ Hugging Face Access Token が入力されていません。話者分離をスキップします。")
                        if use_diarization and hf_token:
                            try:
                                progress_text.text("話者分離処理中（WhisperX）...")
                                diarize_model = whisperx.diarize.DiarizationPipeline(token=hf_token, device=device_str)
                                diarize_kwargs = {}
                                if diarization_num_speakers > 0:
                                    diarize_kwargs["num_speakers"] = diarization_num_speakers
                                else:
                                    diarize_kwargs["min_speakers"] = diarization_min_speakers
                                    diarize_kwargs["max_speakers"] = diarization_max_speakers
                                diarize_segments = diarize_model(audio_array, **diarize_kwargs)
                                wx_with_speakers = whisperx.assign_word_speakers(diarize_segments, result)
                                result["segments"] = wx_with_speakers["segments"]
                                speaker_segments = [
                                    {
                                        "speaker": seg.get("speaker", "UNKNOWN"),
                                        "start": seg["start"],
                                        "end": seg["end"],
                                        "text": seg["text"].strip()
                                    }
                                    for seg in wx_with_speakers["segments"]
                                ]
                                st.success("✅ 話者分離が完了しました！")
                            except Exception as e:
                                err_msg = str(e)
                                if "cannot find the requested files" in err_msg or "locate the file on the Hub" in err_msg:
                                    st.error("⚠️ 話者分離エラー: HuggingFaceからモデルをダウンロードできません。\n\n**確認事項:**\n1. HF Access Token が正しく入力されているか\n2. [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) の利用規約に同意済みか\n3. [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) の利用規約に同意済みか")
                                elif "403" in err_msg or "restricted" in err_msg:
                                    st.error("⚠️ 話者分離エラー（403）")
                                    with st.expander("エラー詳細（開発者向け）"):
                                        import traceback
                                        st.code(traceback.format_exc())
                                else:
                                    st.warning(f"⚠️ 話者分離でエラーが発生しました: {err_msg}")

                    else:
                        # === 既存 Whisper / ReazonSpeech フロー ===
                        if engine_option in ("Whisper (標準)", "Whisper (日本語特化)"):
                            model = load_whisper_model(model_option)
                        else:
                            model = load_reazonspeech_model_cached(reazonspeech_precision)

                        load_end = time.time()
                        progress_text.text(f"モデルロード完了（{load_end - load_start:.2f}秒）")

                        # 文字起こし処理
                        progress_text.text("文字起こし処理中...")
                        transcribe_start = time.time()

                        if engine_option == "ReazonSpeech v2.0":
                            # === ReazonSpeech フロー ===
                            from reazonspeech.k2.asr.interface import TranscribeConfig
                            st.info("🇯🇵 ReazonSpeech v2.0 (sherpa-onnx) で文字起こし中...")
                            config = TranscribeConfig(verbose=False)
                            audio = audio_from_path(temp_filename)
                            rs_result = reazonspeech_transcribe(model, audio, config=config)

                            # サブワードから時間付きセグメントを生成（2秒間隔でグループ化）
                            segments = []
                            if rs_result.subwords:
                                GAP_THRESHOLD = 2.0
                                current_group = []
                                for sw in rs_result.subwords:
                                    if current_group and (sw.seconds - current_group[-1].seconds) > GAP_THRESHOLD:
                                        seg_start = current_group[0].seconds
                                        seg_end = current_group[-1].seconds + 0.5
                                        seg_text = "".join(s.token for s in current_group)
                                        if seg_text.strip():
                                            segments.append({"start": seg_start, "end": seg_end, "text": seg_text})
                                        current_group = []
                                    current_group.append(sw)
                                if current_group:
                                    seg_start = current_group[0].seconds
                                    seg_end = current_group[-1].seconds + 0.5
                                    seg_text = "".join(s.token for s in current_group)
                                    if seg_text.strip():
                                        segments.append({"start": seg_start, "end": seg_end, "text": seg_text})

                            if not segments:
                                segments = [{"start": 0.0, "end": duration, "text": rs_result.text}]

                            result = {
                                "text": rs_result.text,
                                "segments": segments,
                                "language": "ja"
                            }
                        else:
                            # === Whisper フロー ===
                            options = {}

                            if engine_option == "Whisper (日本語特化)":
                                options.update({
                                    "language": "ja",
                                    "task": "transcribe",
                                    "initial_prompt": "以下は日本語の音声です。正確に文字起こししてください。",
                                    "word_timestamps": True,
                                })
                                st.info("🇯🇵 日本語特化モードで処理します。")
                            else:
                                if language_option:
                                    options["language"] = language_option

                            options["verbose"] = False
                            options["fp16"] = torch.cuda.is_available()
                            options["temperature"] = 0
                            options["beam_size"] = 1 if torch.cuda.is_available() else 5

                            # 文字起こし実行
                            if use_chunking:
                                progress_text.text("音声を8分間のチャンクに分割中...")
                                chunk_files = split_audio_into_chunks(temp_filename, chunk_duration_seconds=480)

                                if not chunk_files:
                                    st.error("音声分割に失敗しました。")
                                    return

                                st.info(f"音声を{len(chunk_files)}個のチャンクに分割しました。")
                                progress_bar = st.progress(0)

                                def progress_callback(message):
                                    progress_text.text(message)
                                    current_chunk = int(message.split()[1].split('/')[0]) if 'チャンク' in message else 1
                                    progress_bar.progress(current_chunk / len(chunk_files))

                                result = transcribe_chunks(model, chunk_files, options, progress_callback, chunk_duration_seconds=480)
                                progress_bar.progress(1.0)
                            else:
                                result = model.transcribe(temp_filename, **options)

                        # 話者分離処理（pyannote直接）
                        if use_diarization and not hf_token:
                            st.warning("⚠️ Hugging Face Access Token が入力されていません。話者分離をスキップします。")
                        if use_diarization and hf_token:
                            try:
                                progress_text.text("話者分離処理中...")
                                diarization_pipeline = load_diarization_pipeline(hf_token)

                                diarization_options = {}
                                if diarization_num_speakers > 0:
                                    diarization_options["num_speakers"] = diarization_num_speakers
                                else:
                                    diarization_options["min_speakers"] = diarization_min_speakers
                                    diarization_options["max_speakers"] = diarization_max_speakers

                                audio_input = load_audio_for_pyannote(temp_filename)
                                diarization = diarization_pipeline(audio_input, **diarization_options)
                                speaker_segments = assign_speakers_to_segments(result["segments"], diarization)
                                st.success("✅ 話者分離が完了しました！")
                            except Exception as e:
                                err_msg = str(e)
                                if "cannot find the requested files" in err_msg or "locate the file on the Hub" in err_msg:
                                    st.error("⚠️ 話者分離エラー: HuggingFaceからモデルをダウンロードできません。\n\n**確認事項:**\n1. HF Access Token が正しく入力されているか\n2. [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) の利用規約に同意済みか\n3. [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) の利用規約に同意済みか")
                                elif "403" in err_msg or "restricted" in err_msg:
                                    st.error("⚠️ 話者分離エラー（403）")
                                    with st.expander("エラー詳細（開発者向け）"):
                                        import traceback
                                        st.code(traceback.format_exc())
                                else:
                                    st.warning(f"⚠️ 話者分離でエラーが発生しました: {err_msg}")

                    transcribe_end = time.time()
                    progress_text.empty()
                    
                    # 処理時間計算
                    transcribe_time = transcribe_end - transcribe_start
                    total_time = transcribe_end - load_start
                    
                    # 結果表示
                    st.markdown("### 📝 文字起こし結果")
                    st.success(f"処理完了（文字起こし: {transcribe_time:.2f}秒、合計: {total_time:.2f}秒）")
                    
                    # テキスト結果表示
                    st.markdown("#### 基本テキスト")
                    st.text_area("文字起こし結果", value=result["text"], height=200)
                    
                    st.download_button(
                        label="📥 テキストをダウンロード",
                        data=result["text"],
                        file_name=f"{os.path.splitext(uploaded_file.name)[0]}_transcript.txt",
                        mime="text/plain"
                    )
                    
                    # 話者付きテキスト表示
                    if speaker_segments:
                        st.markdown("#### 🎭 話者付きテキスト")
                        speaker_text = ""
                        for seg in speaker_segments:
                            speaker_text += f"[{seg['speaker']}] {seg['text']}\n"
                        
                        st.text_area("話者付き文字起こし結果", value=speaker_text, height=250)
                        
                        st.download_button(
                            label="📥 話者付きテキストをダウンロード",
                            data=speaker_text,
                            file_name=f"{os.path.splitext(uploaded_file.name)[0]}_transcript_speakers.txt",
                            mime="text/plain"
                        )
                    
                    # タイムスタンプ付きの詳細結果
                    with st.expander("🕐 詳細（タイムスタンプ付き）"):
                        segments_to_display = speaker_segments if speaker_segments else result["segments"]

                        for seg in segments_to_display:
                            start_time = seg["start"]
                            end_time = seg["end"]
                            text = seg["text"]

                            start_formatted = str(datetime.utcfromtimestamp(start_time).strftime('%H:%M:%S.%f'))[:-3]
                            end_formatted = str(datetime.utcfromtimestamp(end_time).strftime('%H:%M:%S.%f'))[:-3]

                            if "speaker" in seg:
                                st.write(f"**[{seg['speaker']}]** `{start_formatted}` → `{end_formatted}`: {text}")
                            else:
                                st.write(f"`{start_formatted}` → `{end_formatted}`: {text}")

                    # WhisperX 単語レベルのタイムスタンプ
                    if engine_option == "WhisperX (高速・単語精度)" and whisperx_word_align:
                        with st.expander("📝 単語レベルのタイムスタンプ（WhisperX）"):
                            for seg in result["segments"]:
                                words = seg.get("words", [])
                                if not words:
                                    continue
                                speaker_label = f"**[{seg['speaker']}]** " if "speaker" in seg else ""
                                word_parts = []
                                for w in words:
                                    w_start = datetime.fromtimestamp(w.get("start", 0), tz=timezone.utc).strftime('%H:%M:%S.%f')[:-3]
                                    word_parts.append(f"`{w_start}` {w['word']}")
                                st.markdown(f"{speaker_label}" + "　".join(word_parts))
                
                except Exception as e:
                    st.error(f"❌ エラーが発生しました: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())
                
                finally:
                    if os.path.exists(temp_filename):
                        try:
                            os.unlink(temp_filename)
                        except:
                            pass
    
    else:
        st.info("👆 音声ファイルをアップロードしてください")
        
        with st.expander("💡 使い方"):
            st.markdown("""
            ### 基本的な使い方
            1. サイドバーでモデルサイズと言語を選択
            2. 音声ファイルをアップロード
            3. 「文字起こし開始」ボタンをクリック
            4. 結果を確認し、必要に応じてダウンロード
            
            ### 🎭 話者分離機能
            - 複数人が話している音声ファイルで、誰が話しているかを自動識別
            - **Hugging Face Access Token**が必要です
            - [pyannote/speaker-diarization](https://huggingface.co/pyannote/speaker-diarization-3.1)の利用規約に同意が必要
            - トークンは[こちら](https://huggingface.co/settings/tokens)から取得できます
            
            ### モデルサイズについて
            - **turbo**: 最新最適化モデル（推奨）
            - **large**: 最高精度
            - **medium**: 高精度・バランス型
            - **small**: 中程度の精度
            - **base/tiny**: 軽量・高速
            
            ### 長時間音声の処理
            - 10分を超える音声は自動的に8分ごとに分割
            - ハルシネーション防止のための処理
            """)

if __name__ == "__main__":
    main()
