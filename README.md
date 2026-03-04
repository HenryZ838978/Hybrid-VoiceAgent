# Hybrid Voice Agent

**A natural, ultra-fast voice conversation system — audio in, audio out, ~250ms first response.**

Built on MiniCPM-o 4.5 (Omni understanding) + VoxCPM 1.5 (TTS), served through vLLM with AWQ-Marlin quantization.

---

## Why This Architecture?

Traditional voice agent pipelines chain 4-5 separate models:

```
ASR (117ms) → RAG (4ms) → LLM (163ms) → TTS (174ms) = ~458ms first audio
```

Each model adds latency, and ASR errors propagate downstream. We asked: **what if the LLM could understand audio directly?**

MiniCPM-o 4.5 is a 9B omni-modal model with built-in Whisper-medium (audio encoder) + Qwen3-8B (LLM backbone). It accepts audio input natively — no separate ASR needed. Combined with vLLM's optimized serving, this eliminates the ASR bottleneck entirely:

```
vLLM Omni AWQ (50ms TTFT) → VoxCPM TTS (190ms TTFA) = ~250ms first audio
```

### Head-to-Head Comparison

| | Pipeline (ASR+LLM+TTS) | Hybrid (Omni+TTS) |
|---|---|---|
| **First Audio** | ~454ms | **~250ms (1.8x faster)** |
| Models | 5 (VAD + ASR + RAG + LLM + TTS) | **2 (Omni + TTS)** |
| ASR errors | Yes (propagate to LLM) | **None (end-to-end audio understanding)** |
| LLM TTFT | 163ms (MiniCPM4.1 GPTQ) | **50ms (MiniCPM-o AWQ-Marlin)** |
| Turn-taking | Signal-level (VAD threshold) | **Semantic-level (model understands audio context)** |
| VRAM | ~17.6 GB (2 GPUs) | ~17.5 GB (2 GPUs) |

### Why vLLM Matters

Running MiniCPM-o 4.5 through raw `transformers` gives 3400ms latency. Through vLLM with AWQ-Marlin:

| | Raw Transformers | vLLM AWQ-Marlin |
|---|---|---|
| TTFT | 3400ms | **50ms (68x faster)** |
| Tokens/sec | 15 | **109 (7x faster)** |
| VRAM | 19.8 GB (bf16) | **10.5 GB (int4)** |

vLLM provides PagedAttention, CUDA Graph, and native AWQ-Marlin kernels — turning a research model into a production-ready inference engine.

## Features

- **~250ms first audio response** — faster than human conversational turn-taking (~300ms)
- **Audio-native understanding** — MiniCPM-o 4.5 processes speech directly, no ASR intermediate
- **Voice cloning** — VoxCPM 1.5 with reference audio for consistent persona
- **Sentence-level streaming** — LLM streams text, TTS generates per-sentence, playback starts immediately
- **Robust barge-in** — VAD 0.85 threshold + RMS energy gate + 3-chunk confirmation window
- **AudioWorklet playback** — zero-pop queue-based audio player in the browser
- **Real-time metrics** — LLM TTFT, TTS TTFA, head-to-head comparison with pipeline baseline
- **Public demo** — one-click Cloudflare tunnel deployment

## Architecture

```
Browser (mic + speaker)
    │
    └── WebSocket ──→ FastAPI Server (:3001)
                        ├── VAD (Silero, CPU)
                        ├── vLLM API (:8200, GPU)     ← MiniCPM-o 4.5 AWQ int4
                        │     audio understanding        10.5 GB, TTFT ~50ms
                        │     + text generation          AWQ-Marlin kernel
                        │
                        └── VoxCPM TTS (GPU)           ← voice cloning
                              text → speech               TTFA ~190ms
                              44.1 kHz output             cfg=3.0, temp=0.7
```

## Quick Start

### Prerequisites

- NVIDIA GPU with 24GB+ VRAM (e.g., RTX 4090)
- Second GPU for TTS (or shared with lower `gpu_memory_utilization`)
- Conda / Python 3.10+

### 1. Setup Environment

```bash
bash setup_env.sh
```

### 2. Download Models

- **MiniCPM-o 4.5 AWQ**: `openbmb/MiniCPM-o-4_5-awq` from HuggingFace
- **VoxCPM 1.5**: from the voiceagent models directory
- **Silero VAD**: bundled with the voiceagent models

### 3. Start vLLM Server

```bash
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model /path/to/MiniCPM-o-4_5-awq \
  --served-model-name MiniCPM-o-4.5-awq \
  --trust-remote-code --dtype half --quantization awq_marlin \
  --gpu-memory-utilization 0.90 --max-model-len 4096 \
  --host 0.0.0.0 --port 8200
```

### 4. Start Hybrid Server

```bash
CUDA_VISIBLE_DEVICES=1 TTS_DEVICE=cuda:0 PORT=3001 \
  python ws_server_hybrid.py
```

### 5. Public Demo (optional)

```bash
./cloudflared tunnel --url http://localhost:3001
```

## Benchmark Results

Tested on RTX 4090 x2, MiniCPM-o 4.5 AWQ int4:

| Metric | Value |
|---|---|
| LLM TTFT (p50) | 50ms |
| TTS TTFA | ~190ms |
| First Audio | **~250ms** |
| LLM TPS | 109 tok/s |
| Omni VRAM | 10.5 GB (AWQ int4) |
| TTS VRAM | ~7 GB |

## Project Structure

```
├── ws_server_hybrid.py     # Core: vLLM Omni + VoxCPM TTS hybrid pipeline
├── ws_server.py            # Alternative: pure omni end-to-end server
├── static/index.html       # Frontend with comparison dashboard
├── engine/
│   ├── omni.py             # MiniCPM-o 4.5 native engine (for benchmarking)
│   └── vad.py              # Silero VAD wrapper
├── config.py               # Model paths, prompts, KB embedding
├── benchmark.py            # Latency measurement script
├── setup_env.sh            # Environment setup
└── start_all.sh            # One-click launch
```

## License

MIT
