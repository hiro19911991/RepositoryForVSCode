# Voxtral-Mini-4B-Realtime-2602 Integration Plan

## Model Overview
- **Model**: mistralai/Voxtral-Mini-4B-Realtime-2602
- **Type**: Multilingual realtime speech-transcription model
- **Languages**: 13 languages supported
- **Latency**: <500ms with configurable delays (240ms-2.4s, recommended 480ms)
- **Architecture**: ~3.4B Language Model + ~970M Audio Encoder
- **Hardware**: Single GPU with >=16GB memory required

## Integration Strategy

### 1. Current App Architecture
- Streamlit-based UI
- Support for: Whisper (standard/Japanese-optimized), ReazonSpeech
- File upload and batch processing
- Chunk processing for long audio files

### 2. Voxtral Integration Approach

#### A. Add as Third ASR Engine Option
```python
available_engines = [
    "Whisper (標準)", 
    "Whisper (日本語特化)", 
    "Voxtral Realtime"  # NEW
]
if REAZONSPEECH_AVAILABLE:
    available_engines.append("ReazonSpeech v2.0")
```

#### B. Dependencies Required
```bash
# Install vLLM (nightly build)
uv pip install -U vllm

# Install Mistral Common (>=1.9.0)
pip install mistral_common>=1.9.0

# Audio processing libraries
uv pip install sox librosa soundfile

# Optional: Upgrade transformers
uv pip install --upgrade transformers
```

#### C. Model Serving Setup
- Use vLLM to serve the model as an API endpoint
- Configure for optimal latency (480ms delay recommended)
- Handle BF16 format and memory requirements

### 3. Implementation Plan

#### Phase 1: Basic Integration
1. Add Voxtral dependencies to requirements.txt
2. Create Voxtral model loading function
3. Add Voxtral option to UI dropdown
4. Implement basic transcription for uploaded files

#### Phase 2: Real-time Features (Future Enhancement)
1. Add WebRTC/WebSocket support for live audio streaming
2. Implement real-time transcription display
3. Add configurable latency settings

### 4. Technical Implementation

#### A. Model Loading
```python
from vllm import LLM
from mistral_common.protocol.instruct.messages import SystemMessage, UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest

@st.cache_resource
def load_voxtral_model():
    """Load Voxtral model using vLLM"""
    model = LLM(
        model="mistralai/Voxtral-Mini-4B-Realtime-2602",
        trust_remote_code=True,
        max_model_len=131072,  # ~ca. 3h for recommended settings
        tensor_parallel_size=1,
    )
    return model
```

#### B. Transcription Function
```python
def transcribe_with_voxtral(model, audio_path, delay_ms=480):
    """Transcribe audio using Voxtral model"""
    # Implementation will use vLLM's audio processing capabilities
    # Convert audio to appropriate format
    # Process with configurable delay
    # Return transcription results
    pass
```

### 5. UI/UX Considerations

#### A. Model Selection
- Add "Voxtral Realtime" to engine dropdown
- Show hardware requirements warning (16GB GPU memory)
- Add latency configuration slider (240ms-2.4s)

#### B. Performance Indicators
- Display real-time processing status
- Show latency metrics
- Memory usage monitoring

### 6. Hardware Requirements
- **Minimum**: Single GPU with 16GB VRAM
- **Recommended**: NVIDIA RTX 3090/4090 or better
- **CPU**: Modern multi-core processor
- **RAM**: 32GB+ system memory recommended

### 7. Limitations and Considerations
- Requires significant GPU resources (16GB+ VRAM)
- Real-time streaming requires additional WebSocket/WebRTC implementation
- Model is optimized for real-time use, may be overkill for batch processing
- Limited to 13 supported languages

## Next Steps
1. Update requirements.txt with Voxtral dependencies
2. Implement basic Voxtral integration in app.py
3. Test with sample audio files
4. Add UI controls for latency configuration
5. Consider future real-time streaming features