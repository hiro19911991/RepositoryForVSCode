#!/usr/bin/env python3
"""
Whisper文字起こしWebアプリ（Streamlit使用）
"""

import os
import sys
import time
import tempfile
import whisper
import torch
import streamlit as st
from datetime import datetime
import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment

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
            time_offset += chunk_duration_seconds
                
        except Exception as e:
            st.warning(f"チャンク {i+1} の処理でエラーが発生しました: {str(e)}")
            time_offset += chunk_duration_seconds  # エラーでもオフセットを進める
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
    
    # モデル選択
    model_option = st.sidebar.selectbox(
        "モデルサイズを選択",
        options=get_available_models(),
        index=9,  # turboをデフォルトに（最新の最適化モデル）
        help="モデルの詳細: turbo(最新最適化)、.enは英語専用で高精度、largeは最高精度"
    )
    
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
    
    # サイドバーにGitHubリンク
    st.sidebar.markdown("---")
    st.sidebar.markdown("[GitHubリポジトリ](https://github.com/yourusername/whisper-transcription)")
    
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
                    model = load_whisper_model(model_option)
                    load_end = time.time()
                    progress_text.text(f"モデルロード完了（{load_end - load_start:.2f}秒）")
                    
                    # 文字起こし処理
                    progress_text.text("文字起こし処理中...")
                    transcribe_start = time.time()
                    
                    # 言語オプション設定
                    options = {}
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
                        result = transcribe_chunks(model, chunk_files, options, progress_callback, chunk_duration_seconds=480)
                        progress_bar.progress(1.0)  # 完了
                        
                    else:
                        # 通常の処理（10分未満の場合）
                        result = model.transcribe(temp_filename, **options)
                    
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
                    
                    # タイムスタンプ付きの詳細結果
                    with st.expander("詳細（タイムスタンプ付き）"):
                        # テーブル表示用のデータ準備
                        table_data = []
                        timestamp_text = ""
                        
                        for segment in result["segments"]:
                            start_time = segment["start"]
                            end_time = segment["end"]
                            text = segment["text"]
                            
                            # 時間をフォーマット (HH:MM:SS.ms)
                            start_formatted = str(datetime.utcfromtimestamp(start_time).strftime('%H:%M:%S.%f'))[:-3]
                            end_formatted = str(datetime.utcfromtimestamp(end_time).strftime('%H:%M:%S.%f'))[:-3]
                            
                            table_data.append({
                                "開始": start_formatted,
                                "終了": end_formatted,
                                "テキスト": text
                            })
                            
                            timestamp_text += f"[{start_formatted} --> {end_formatted}] {text}\n"
                        
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
