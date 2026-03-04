"""
Hybrid Voice Agent v2 — Production-grade stability rewrite

Architecture:
  Browser <-WSS-> FastAPI (:3001)
                    |- VAD + State Machine (async, never blocks)
                    |- vLLM API (:8200) — audio understanding + text gen (HTTP streaming)
                    '- VoxCPM TTS (thread-isolated) — text to speech

Key design: TTS runs in a dedicated thread via run_in_executor. The async event
loop is never blocked by GPU work. Audio is sent in 50ms chunks with cancel
checks between each, enabling responsive barge-in.
"""
import os
import sys
import re
import json
import time
import base64
import asyncio
import logging
import numpy as np
from enum import Enum

VOICEAGENT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "voiceagent")
sys.path.insert(0, VOICEAGENT_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import httpx
import uvicorn
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("hybrid")

VLLM_BASE = os.environ.get("VLLM_BASE", "http://localhost:8200")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "MiniCPM-o-4.5-awq")
TTS_MODEL_DIR = os.path.join(VOICEAGENT_DIR, "models", "VoxCPM1.5")
TTS_DEVICE = os.environ.get("TTS_DEVICE", "cuda:0")
VAD_MODEL_DIR = os.path.join(VOICEAGENT_DIR, "models", "snakers4_silero-vad")
VOICE_PROMPT_WAV = os.path.join(VOICEAGENT_DIR, "data", "voice_prompt.wav")
VOICE_PROMPT_TEXT = "在国内引起了非常大的反响啊，我们也完全没有想到我们的这个工作会以这种方式出圈。"

EMBED_MODEL_DIR = os.path.join(VOICEAGENT_DIR, "models", "bge-small-zh-v1.5")
KB_DATA_PATH = os.path.join(VOICEAGENT_DIR, "data", "sample_kb.json")
RAG_DEVICE = os.environ.get("RAG_DEVICE", "cpu")
RAG_TOP_K = 3
MAX_HISTORY_TURNS = 8

SAMPLE_RATE_IN = 16000
SAMPLE_RATE_OUT = 44100
CHUNK_SAMPLES = 512
CHUNK_MS = 32
PORT = int(os.environ.get("PORT", "3001"))

TTS_SEND_CHUNK_BYTES = 8820 * 2  # ~200ms at 44.1kHz 16-bit (larger = smoother over network)

BARGE_IN_VAD_THRESHOLD = 0.85
BARGE_IN_RMS_THRESHOLD = 0.015
BARGE_IN_CONFIRM_CHUNKS = 3

ENDPOINT_FAST_CHUNKS = 6     # ~192ms for short utterances (<500ms)
ENDPOINT_DEFAULT_CHUNKS = 10  # ~320ms for normal
ENDPOINT_SLOW_CHUNKS = 20    # ~640ms for long utterances (>3s)

SYSTEM_PROMPT_BASE = (
    "你是面壁智能的专业客服代表。只使用中文普通话。"
    "语气友好、专业、沉稳，像资深客服人员。"
    "回答简洁准确，每次1-2句话。不要用编号和列举。"
    "不要使用任何XML标签。不要复述用户的话。直接回答问题。"
)

SYSTEM_PROMPT_RAG = (
    "{base}\n\n以下是知识库，优先根据知识库内容回答。如果知识库没有相关信息，坦诚告知。"
    "\n\n知识库：\n{context}"
)

SENTENCE_RE = re.compile(r'([^。？！?!.]+[。？！?!.])')
THINK_RE = re.compile(r'<think>.*?</think>', re.DOTALL)
TAG_RE = re.compile(r'<[^>]+>')
HEARD_RE = re.compile(r'\[听到[】\]][^\n]*[\n]?')


class State(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"


engine = {}
metrics_history = []


def _rms(chunk: np.ndarray) -> float:
    return float(np.sqrt(np.mean(chunk ** 2)))


async def _async_tts(text: str) -> list:
    """Collect all TTS chunks. Runs on the main event loop (nanovllm requirement).
    Returns list of numpy audio chunks.
    """
    tts_engine = engine["tts_engine"]
    pid = engine.get("tts_prompt_id")
    kwargs = {"target_text": text, "temperature": 0.7, "cfg_value": 3.0}
    if pid:
        kwargs["prompt_id"] = pid

    chunks = []
    chunk_idx = 0
    async for audio_chunk in tts_engine.generate(**kwargs):
        await asyncio.sleep(0)  # yield to event loop between GPU steps
        if isinstance(audio_chunk, np.ndarray):
            c = audio_chunk
        elif hasattr(audio_chunk, 'numpy'):
            c = audio_chunk.numpy()
        else:
            c = np.array(audio_chunk, dtype=np.float32)

        chunk_idx += 1
        if chunk_idx == 1:
            continue
        if chunk_idx == 2:
            fade = int(SAMPLE_RATE_OUT * 0.01)
            if len(c) > fade:
                c[:fade] *= np.linspace(0, 1, fade, dtype=c.dtype)
        chunks.append(c)

    # Cross-fade between TTS chunks to eliminate boundary clicks
    if len(chunks) > 1:
        xfade = min(441, min(len(c) for c in chunks))  # ~10ms at 44.1kHz
        for i in range(1, len(chunks)):
            fade_out = np.linspace(1, 0, xfade, dtype=np.float32)
            fade_in = np.linspace(0, 1, xfade, dtype=np.float32)
            chunks[i - 1][-xfade:] *= fade_out
            chunks[i][:xfade] *= fade_in

    return chunks


def _sync_tts_via_main_loop(text: str) -> list:
    """Thread-callable wrapper: schedules TTS on the main event loop, blocks until done."""
    main_loop = engine["main_loop"]
    future = asyncio.run_coroutine_threadsafe(_async_tts(text), main_loop)
    return future.result(timeout=30)


def clean_text(text):
    text = THINK_RE.sub('', text)
    text = TAG_RE.sub('', text)
    text = HEARD_RE.sub('', text)
    return text.strip()


async def load_engines():
    from engine.vad import SileroVAD
    from nanovllm_voxcpm import VoxCPM

    log.info("=== Loading Hybrid Agent v2 engines ===")

    log.info("Loading VAD...")
    vad = SileroVAD(VAD_MODEL_DIR, threshold=0.5)
    vad.load()
    engine["vad_template"] = vad

    log.info("Loading VoxCPM TTS on %s ...", TTS_DEVICE)
    gpu_idx = int(TTS_DEVICE.split(":")[-1]) if ":" in TTS_DEVICE else 0
    tts_engine = VoxCPM.from_pretrained(
        model=TTS_MODEL_DIR, gpu_memory_utilization=0.5, devices=[gpu_idx],
    )
    engine["tts_engine"] = tts_engine
    engine["tts_prompt_id"] = None

    if os.path.exists(VOICE_PROMPT_WAV):
        log.info("Registering voice clone...")
        with open(VOICE_PROMPT_WAV, "rb") as f:
            pid = await tts_engine.add_prompt(f.read(), "wav", VOICE_PROMPT_TEXT)
        engine["tts_prompt_id"] = pid
        log.info("Voice clone registered: %s", pid)

    engine["main_loop"] = asyncio.get_event_loop()

    log.info("Warming up TTS...")
    await _async_tts("你好，欢迎致电。")
    log.info("TTS warm.")

    log.info("Loading RAG (bge-small-zh-v1.5)...")
    if os.path.isdir(EMBED_MODEL_DIR) and os.path.exists(KB_DATA_PATH):
        import importlib.util
        spec = importlib.util.spec_from_file_location("rag", os.path.join(VOICEAGENT_DIR, "engine", "rag.py"))
        rag_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rag_mod)
        rag = rag_mod.RAGEngine(EMBED_MODEL_DIR, device=RAG_DEVICE, top_k=RAG_TOP_K)
        rag.load()
        with open(KB_DATA_PATH, "r", encoding="utf-8") as f:
            import json as _json
            docs = _json.load(f)
        info = rag.build_index(docs)
        engine["rag"] = rag
        log.info("RAG: %d docs, dim=%d, embed=%.1fms", info["num_docs"], info["dim"], info["encode_ms"])
    else:
        engine["rag"] = None
        log.info("RAG: skipped (model or KB not found)")

    log.info("vLLM endpoint: %s (model: %s)", VLLM_BASE, VLLM_MODEL)
    log.info("=== All engines loaded (v2.1) ===")


app = FastAPI(title="Hybrid Voice Agent v2")


@app.on_event("startup")
async def startup():
    await load_engines()


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse((Path(__file__).parent / "static" / "index.html").read_text("utf-8"))


@app.get("/api/info")
async def api_info():
    return {
        "architecture": "hybrid v2: vLLM MiniCPM-o-4.5 AWQ + VoxCPM TTS (thread-isolated)",
        "vllm_model": VLLM_MODEL,
        "vllm_endpoint": VLLM_BASE,
        "tts": "VoxCPM 1.5 (thread-isolated)",
        "sample_rate_out": SAMPLE_RATE_OUT,
    }


@app.get("/api/metrics")
async def api_metrics():
    return {"history": metrics_history[-50:]}


async def stream_vllm_text(audio_b64: str, history: list, system_prompt: str):
    """Stream text from vLLM with multi-turn history + current audio."""
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": [
        {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{audio_b64}"}}
    ]})

    payload = {
        "model": VLLM_MODEL,
        "messages": messages,
        "max_tokens": 256,
        "temperature": 0.7,
        "stream": True,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }
    async with httpx.AsyncClient(timeout=60) as client:
        async with client.stream("POST", f"{VLLM_BASE}/v1/chat/completions", json=payload) as resp:
            async for line in resp.aiter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[6:])
                delta = chunk["choices"][0]["delta"].get("content", "")
                if delta:
                    yield delta


async def send_audio_chunked(ws: WebSocket, audio: np.ndarray, cancel: asyncio.Event):
    """Send audio in 50ms chunks with cancel check between each."""
    if audio.dtype != np.int16:
        audio = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    audio_bytes = audio.tobytes()
    for i in range(0, len(audio_bytes), TTS_SEND_CHUNK_BYTES):
        if cancel.is_set():
            return
        try:
            await ws.send_bytes(audio_bytes[i:i + TTS_SEND_CHUNK_BYTES])
        except Exception:
            return
        await asyncio.sleep(0)


@app.websocket("/ws/voice")
async def ws_voice(ws: WebSocket):
    await ws.accept()
    log.info("WebSocket connected")

    from engine.vad import SileroVAD
    vad = SileroVAD(VAD_MODEL_DIR, threshold=0.5)
    vad.load()

    state = State.IDLE
    audio_buffer = []
    silence_count = 0
    turn = 0
    cancel = asyncio.Event()
    barge_confirm = 0
    speaking_task = None
    conversation_history = []  # list of {"role": "user"/"assistant", "content": str}

    async def send_json(d):
        try:
            await ws.send_json(d)
        except Exception:
            pass

    async def set_state(s: State):
        nonlocal state
        state = s
        await send_json({"type": "state", "state": s.value})

    def adaptive_endpoint_threshold() -> int:
        dur_ms = len(audio_buffer) * CHUNK_MS
        if dur_ms < 500:
            return ENDPOINT_FAST_CHUNKS
        elif dur_ms > 3000:
            return ENDPOINT_SLOW_CHUNKS
        return ENDPOINT_DEFAULT_CHUNKS

    async def run_pipeline(audio_np: np.ndarray, current_turn: int):
        nonlocal state
        loop = asyncio.get_event_loop()
        t_start = time.perf_counter()

        await set_state(State.THINKING)
        await send_json({"type": "processing", "audio_duration_s": round(len(audio_np) / SAMPLE_RATE_IN, 2)})

        import soundfile as sf, io
        buf = io.BytesIO()
        sf.write(buf, audio_np, SAMPLE_RATE_IN, format='WAV', subtype='PCM_16')
        audio_b64 = base64.b64encode(buf.getvalue()).decode()

        rag = engine.get("rag")
        rag_context = ""
        rag_ms = 0
        if rag and conversation_history:
            last_ai = next((m["content"] for m in reversed(conversation_history) if m["role"] == "assistant"), "")
            query = last_ai
            if query:
                t_rag = time.perf_counter()
                rag_result = await loop.run_in_executor(None, rag.get_context, query)
                rag_context = rag_result["context"]
                rag_ms = rag_result["total_ms"]
                await send_json({"type": "rag", "context": rag_context, "latency_ms": round(rag_ms, 1)})

        if rag_context:
            sys_prompt = SYSTEM_PROMPT_RAG.format(base=SYSTEM_PROMPT_BASE, context=rag_context)
        else:
            sys_prompt = SYSTEM_PROMPT_BASE

        llm_ttft = None
        full_text = ""
        sentence_buf = ""
        tts_started = False
        tts_ttfa = None
        sentence_count = 0

        async for token in stream_vllm_text(audio_b64, conversation_history, sys_prompt):
            if cancel.is_set():
                log.info("[Turn %d] Cancel during LLM stream", current_turn)
                break
            if llm_ttft is None:
                llm_ttft = (time.perf_counter() - t_start) * 1000

            full_text += token
            sentence_buf += clean_text(token)

            sentences = SENTENCE_RE.findall(sentence_buf)
            if sentences:
                sentence_buf = SENTENCE_RE.sub('', sentence_buf)
                for sent in sentences:
                    if cancel.is_set():
                        break
                    sent = sent.strip()
                    if not sent or len(sent) < 2:
                        continue

                    if not tts_started:
                        cancel.clear()
                        await set_state(State.SPEAKING)
                        await send_json({"type": "audio_start", "sample_rate": SAMPLE_RATE_OUT})
                        tts_started = True

                    t_tts = time.perf_counter()
                    tts_chunks = await _async_tts(sent)
                    sentence_count += 1

                    if tts_ttfa is None and tts_chunks:
                        tts_ttfa = (time.perf_counter() - t_tts) * 1000

                    if tts_chunks and not cancel.is_set():
                        audio_out = np.concatenate(tts_chunks)
                        log.info("[Turn %d] TTS sent %d chunks for: '%s'",
                                 current_turn, len(tts_chunks), sent[:30])
                        await send_audio_chunked(ws, audio_out, cancel)

        remaining = clean_text(sentence_buf).strip()
        if remaining and len(remaining) >= 2 and not cancel.is_set():
            if not tts_started:
                cancel.clear()
                await set_state(State.SPEAKING)
                await send_json({"type": "audio_start", "sample_rate": SAMPLE_RATE_OUT})
                tts_started = True

            t_tts = time.perf_counter()
            tts_chunks = await _async_tts(remaining)
            sentence_count += 1
            if tts_ttfa is None and tts_chunks:
                tts_ttfa = (time.perf_counter() - t_tts) * 1000
            if tts_chunks and not cancel.is_set():
                audio_out = np.concatenate(tts_chunks)
                log.info("[Turn %d] TTS sent %d chunks for: '%s'",
                         current_turn, len(tts_chunks), remaining[:30])
                await send_audio_chunked(ws, audio_out, cancel)

        total_ms = (time.perf_counter() - t_start) * 1000

        ai_response = clean_text(full_text).strip()

        if ai_response:
            conversation_history.append({"role": "user", "content": "(audio)"})
            conversation_history.append({"role": "assistant", "content": ai_response})
        while len(conversation_history) > MAX_HISTORY_TURNS * 2:
            conversation_history.pop(0)

        log.info("[Turn %d] History: ai='%s' (total %d msgs)",
                 current_turn, ai_response[:40], len(conversation_history))

        m = {
            "llm_ttft_ms": round(llm_ttft or 0, 1),
            "tts_ttfa_ms": round(tts_ttfa or 0, 1),
            "rag_ms": round(rag_ms, 1),
            "total_ms": round(total_ms, 1),
            "first_audio_ms": round((llm_ttft or 0) + (tts_ttfa or 0), 1),
            "text": ai_response,
            "timestamp": time.time(),
        }
        metrics_history.append(m)

        await send_json({
            "type": "response", "text": ai_response,
            "llm_ttft_ms": m["llm_ttft_ms"], "tts_ttfa_ms": m["tts_ttfa_ms"],
            "rag_ms": m["rag_ms"], "total_ms": m["total_ms"],
            "first_audio_ms": m["first_audio_ms"],
            "history_len": len(conversation_history),
        })
        await send_json({"type": "audio_end", "total_ms": m["total_ms"]})

        if state == State.SPEAKING:
            await set_state(State.IDLE)
        elif state == State.INTERRUPTED:
            log.info("[Turn %d] Pipeline done while interrupted, staying in INTERRUPTED", current_turn)

    async def safe_pipeline(audio_np, current_turn):
        try:
            await run_pipeline(audio_np, current_turn)
        except Exception as e:
            log.error("[Turn %d] Pipeline error: %s", current_turn, e, exc_info=True)
            await set_state(State.IDLE)

    try:
        await set_state(State.IDLE)
        while True:
            data = await ws.receive()
            if data.get("type") == "websocket.disconnect":
                break

            if "bytes" in data and data["bytes"]:
                samples = np.frombuffer(data["bytes"], dtype=np.int16).astype(np.float32) / 32768.0
                for i in range(0, len(samples), CHUNK_SAMPLES):
                    chunk = samples[i:i + CHUNK_SAMPLES]
                    if len(chunk) < CHUNK_SAMPLES:
                        chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))

                    vad_result = vad.process_chunk(chunk)
                    speech_prob = vad_result["speech_prob"]
                    is_speech = speech_prob >= 0.5
                    rms = _rms(chunk)

                    # --- IDLE ---
                    if state == State.IDLE:
                        if is_speech and rms > 0.01:
                            audio_buffer.clear()
                            audio_buffer.append(chunk.copy())
                            silence_count = 0
                            await set_state(State.LISTENING)

                    # --- LISTENING ---
                    elif state == State.LISTENING:
                        audio_buffer.append(chunk.copy())
                        if is_speech:
                            silence_count = 0
                        else:
                            silence_count += 1
                            if silence_count >= adaptive_endpoint_threshold():
                                if len(audio_buffer) >= 6:
                                    turn += 1
                                    full_audio = np.concatenate(audio_buffer)
                                    audio_buffer.clear()
                                    silence_count = 0
                                    cancel.clear()
                                    barge_confirm = 0
                                    speaking_task = asyncio.create_task(safe_pipeline(full_audio, turn))
                                else:
                                    audio_buffer.clear()
                                    silence_count = 0
                                    await set_state(State.IDLE)

                    # --- SPEAKING ---
                    elif state == State.SPEAKING:
                        is_real = speech_prob >= BARGE_IN_VAD_THRESHOLD and rms > BARGE_IN_RMS_THRESHOLD
                        if is_real:
                            barge_confirm += 1
                            if barge_confirm >= BARGE_IN_CONFIRM_CHUNKS:
                                log.info("Barge-in confirmed (prob=%.2f, rms=%.3f)", speech_prob, rms)
                                cancel.set()
                                barge_confirm = 0
                                audio_buffer.clear()
                                audio_buffer.append(chunk.copy())
                                silence_count = 0
                                await send_json({"type": "barge_in"})
                                await set_state(State.INTERRUPTED)
                        else:
                            barge_confirm = 0

                    # --- INTERRUPTED ---
                    elif state == State.INTERRUPTED:
                        audio_buffer.append(chunk.copy())
                        if is_speech:
                            silence_count = 0
                        else:
                            silence_count += 1
                            if silence_count >= 10:
                                if len(audio_buffer) >= 6:
                                    turn += 1
                                    full_audio = np.concatenate(audio_buffer)
                                    audio_buffer.clear()
                                    silence_count = 0
                                    cancel.clear()
                                    barge_confirm = 0
                                    speaking_task = asyncio.create_task(safe_pipeline(full_audio, turn))
                                else:
                                    audio_buffer.clear()
                                    silence_count = 0
                                    await set_state(State.IDLE)

                    # --- THINKING ---
                    elif state == State.THINKING:
                        pass  # ignore audio while processing

            elif "text" in data and data["text"]:
                try:
                    msg = json.loads(data["text"])
                except json.JSONDecodeError:
                    continue
                if msg.get("type") == "reset":
                    cancel.set()
                    vad.reset()
                    audio_buffer.clear()
                    conversation_history.clear()
                    silence_count = 0
                    barge_confirm = 0
                    turn = 0
                    await set_state(State.IDLE)
                    log.info("Session reset (history cleared)")

    except WebSocketDisconnect:
        log.info("Disconnected")
    except Exception as e:
        log.error("WS error: %s", e, exc_info=True)
    finally:
        cancel.set()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
