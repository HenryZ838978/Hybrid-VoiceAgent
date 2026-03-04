"""
Hybrid Voice Agent: vLLM MiniCPM-o-4.5 AWQ (audio->text) + VoxCPM TTS (text->audio)

Architecture:
  Browser <-WSS-> FastAPI (:3001)
                    |- VAD (Silero, CPU)
                    |- vLLM API (:8200, GPU 2) - audio understanding + text gen
                    '- VoxCPM TTS (GPU 1)      - text to speech

Sentence-level streaming: LLM streams text, each complete sentence is
immediately sent to TTS while LLM continues generating the rest.
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

VOICEAGENT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "voiceagent")
sys.path.insert(0, VOICEAGENT_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import nest_asyncio
nest_asyncio.apply()

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

SAMPLE_RATE_IN = 16000
SAMPLE_RATE_OUT = 44100
CHUNK_SAMPLES = 512
PORT = int(os.environ.get("PORT", "3001"))

SYSTEM_PROMPT = (
    "你是面壁智能的专业客服代表。只使用中文普通话。"
    "语气友好、专业、沉稳，像资深客服人员。"
    "回答简洁准确，每次1-2句话。不要用编号和列举。"
    "不要使用任何XML标签。"
)

SENTENCE_RE = re.compile(r'([^。？！?!.]+[。？！?!.])')
THINK_RE = re.compile(r'<think>.*?</think>', re.DOTALL)
TAG_RE = re.compile(r'<[^>]+>')

engine = {}
metrics_history = []


async def recover_tts():
    """Recreate TTS engine after crash."""
    from nanovllm_voxcpm import VoxCPM
    log.warning("Recovering TTS engine...")
    try:
        gpu_idx = int(TTS_DEVICE.split(":")[-1]) if ":" in TTS_DEVICE else 0
        tts_engine = VoxCPM.from_pretrained(
            model=TTS_MODEL_DIR, gpu_memory_utilization=0.5, devices=[gpu_idx],
        )
        engine["tts_engine"] = tts_engine
        if os.path.exists(VOICE_PROMPT_WAV):
            with open(VOICE_PROMPT_WAV, "rb") as f:
                pid = await tts_engine.add_prompt(f.read(), "wav", VOICE_PROMPT_TEXT)
            engine["tts_prompt_id"] = pid
        async for _ in tts_engine.generate(target_text="恢复。", temperature=0.7, cfg_value=3.0):
            pass
        log.info("TTS recovered successfully")
    except Exception as e:
        log.error("TTS recovery failed: %s", e)


async def load_engines():
    from engine.vad import SileroVAD
    from nanovllm_voxcpm import VoxCPM

    log.info("=== Loading Hybrid Agent engines ===")

    log.info("Loading VAD...")
    vad = SileroVAD(VAD_MODEL_DIR, threshold=0.5)
    vad.load()
    engine["vad_template"] = vad

    log.info("Loading VoxCPM TTS on %s ...", TTS_DEVICE)
    gpu_idx = int(TTS_DEVICE.split(":")[-1]) if ":" in TTS_DEVICE else 0
    tts_engine = VoxCPM.from_pretrained(
        model=TTS_MODEL_DIR,
        gpu_memory_utilization=0.5,
        devices=[gpu_idx],
    )
    engine["tts_engine"] = tts_engine
    engine["tts_prompt_id"] = None

    if os.path.exists(VOICE_PROMPT_WAV):
        log.info("Registering voice clone...")
        with open(VOICE_PROMPT_WAV, "rb") as f:
            wav_bytes = f.read()
        pid = await tts_engine.add_prompt(wav_bytes, "wav", VOICE_PROMPT_TEXT)
        engine["tts_prompt_id"] = pid
        log.info("Voice clone registered: %s", pid)

    log.info("Warming up TTS...")
    async for _ in tts_engine.generate(target_text="你好。", temperature=0.7, cfg_value=3.0):
        pass
    log.info("TTS warm.")

    log.info("vLLM endpoint: %s (model: %s)", VLLM_BASE, VLLM_MODEL)
    log.info("=== All engines loaded ===")


app = FastAPI(title="Hybrid Voice Agent (vLLM Omni + VoxCPM TTS)")


@app.on_event("startup")
async def startup():
    await load_engines()


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse((Path(__file__).parent / "static" / "index.html").read_text("utf-8"))


@app.get("/api/info")
async def api_info():
    return {
        "architecture": "hybrid: vLLM MiniCPM-o-4.5 AWQ (audio->text) + VoxCPM TTS (text->audio)",
        "vllm_model": VLLM_MODEL,
        "vllm_endpoint": VLLM_BASE,
        "tts": "VoxCPM 1.5",
        "tts_device": TTS_DEVICE,
        "sample_rate_out": SAMPLE_RATE_OUT,
    }


@app.get("/api/metrics")
async def api_metrics():
    return {"history": metrics_history[-50:]}


def clean_text(text):
    text = THINK_RE.sub('', text)
    text = TAG_RE.sub('', text)
    return text.strip()


async def stream_vllm_text(audio_b64: str):
    """Stream text tokens from vLLM given base64 audio."""
    payload = {
        "model": VLLM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{audio_b64}"}}
            ]},
        ],
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


async def tts_synthesize_chunks(text: str, cancel_event=None):
    """Generate TTS audio chunks. MUST fully consume the generator (nanovllm requirement).
    On cancel: continues draining but stops yielding. asyncio.sleep(0) prevents event loop starvation.
    """
    tts_engine = engine["tts_engine"]
    pid = engine.get("tts_prompt_id")
    kwargs = {"target_text": text, "temperature": 0.7, "cfg_value": 3.0}
    if pid:
        kwargs["prompt_id"] = pid

    chunk_idx = 0
    try:
        async for audio_chunk in tts_engine.generate(**kwargs):
            await asyncio.sleep(0)  # yield to event loop (critical for WS keepalive)

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

            if cancel_event and cancel_event.is_set():
                continue  # drain without yielding

            yield c
    except Exception as e:
        log.error("TTS generation error: %s", e)
        await recover_tts()


@app.websocket("/ws/voice")
async def ws_voice(ws: WebSocket):
    await ws.accept()
    log.info("WebSocket connected")

    from engine.vad import SileroVAD
    vad = SileroVAD(VAD_MODEL_DIR, threshold=0.5)
    vad.load()

    state = "idle"
    audio_buffer = []
    speech_chunks = 0
    cancel = asyncio.Event()
    barge_confirm = 0

    async def send_json(d):
        try: await ws.send_json(d)
        except: pass

    async def set_state(s):
        nonlocal state; state = s
        await send_json({"type": "state", "state": s})

    async def handle_speech(audio_np):
        nonlocal state
        t_start = time.perf_counter()

        await set_state("thinking")
        audio_dur = len(audio_np) / SAMPLE_RATE_IN
        await send_json({"type": "processing", "audio_duration_s": round(audio_dur, 2)})

        import soundfile as sf, io
        buf = io.BytesIO()
        sf.write(buf, audio_np, SAMPLE_RATE_IN, format='WAV', subtype='PCM_16')
        audio_b64 = base64.b64encode(buf.getvalue()).decode()

        llm_ttft = None
        full_text = ""
        sentence_buf = ""
        tts_started = False
        tts_ttfa = None

        async for token in stream_vllm_text(audio_b64):
            if cancel.is_set():
                log.info("Cancel during LLM stream — aborting turn")
                break
            if llm_ttft is None:
                llm_ttft = (time.perf_counter() - t_start) * 1000

            full_text += token
            cleaned = clean_text(token)
            sentence_buf += cleaned

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
                        await set_state("speaking")
                        await send_json({"type": "audio_start", "sample_rate": SAMPLE_RATE_OUT})
                        tts_started = True

                    t_tts = time.perf_counter()
                    sent_chunks = 0
                    async for c in tts_synthesize_chunks(sent, cancel):
                        if cancel.is_set():
                            continue
                        if tts_ttfa is None:
                            tts_ttfa = (time.perf_counter() - t_tts) * 1000
                        c16 = (c * 32767).clip(-32768, 32767).astype(np.int16)
                        sent_chunks += 1
                        try:
                            await ws.send_bytes(c16.tobytes())
                        except Exception as e:
                            log.error("send_bytes failed at chunk %d: %s", sent_chunks, e)
                            break
                    log.info("TTS sent %d chunks for: '%s' (%.0fms)", sent_chunks, sent[:20], (time.perf_counter()-t_tts)*1000)

        remaining = clean_text(sentence_buf).strip()
        if remaining and len(remaining) >= 2 and not cancel.is_set():
            if not tts_started:
                await set_state("speaking")
                await send_json({"type": "audio_start", "sample_rate": SAMPLE_RATE_OUT})
                tts_started = True

            t_tts = time.perf_counter()
            async for c in tts_synthesize_chunks(remaining, cancel):
                if cancel.is_set():
                    continue
                if tts_ttfa is None:
                    tts_ttfa = (time.perf_counter() - t_tts) * 1000
                c16 = (c * 32767).clip(-32768, 32767).astype(np.int16)
                try:
                    await ws.send_bytes(c16.tobytes())
                except:
                    break

        total_ms = (time.perf_counter() - t_start) * 1000
        clean_full = clean_text(full_text).strip()

        m = {
            "llm_ttft_ms": round(llm_ttft or 0, 1),
            "tts_ttfa_ms": round(tts_ttfa or 0, 1),
            "total_ms": round(total_ms, 1),
            "first_audio_ms": round((llm_ttft or 0) + (tts_ttfa or 0), 1),
            "text": clean_full,
            "timestamp": time.time(),
        }
        metrics_history.append(m)

        await send_json({
            "type": "response",
            "text": clean_full,
            "llm_ttft_ms": m["llm_ttft_ms"],
            "tts_ttfa_ms": m["tts_ttfa_ms"],
            "total_ms": m["total_ms"],
            "first_audio_ms": m["first_audio_ms"],
        })
        await send_json({"type": "audio_end", "total_ms": m["total_ms"]})

        if not cancel.is_set():
            await set_state("idle")
        else:
            log.info("Turn cancelled, resetting to idle")
            await set_state("idle")

    async def safe_handle_speech(audio_np):
        try:
            await handle_speech(audio_np)
        except Exception as e:
            log.error("handle_speech error: %s", e, exc_info=True)
            await set_state("idle")

    speaking_task = None
    try:
        await set_state("idle")
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

                    ev = vad.process_chunk(chunk)
                    rms = float(np.sqrt(np.mean(chunk ** 2)))

                    if state == "speaking":
                        is_real = ev["speech_prob"] >= 0.85 and rms > 0.015
                        if is_real:
                            barge_confirm += 1
                            if barge_confirm >= 3:
                                log.info("Barge-in confirmed (3 chunks, prob=%.2f, rms=%.3f)",
                                         ev["speech_prob"], rms)
                                cancel.set()
                                barge_confirm = 0
                                await send_json({"type": "barge_in"})
                                audio_buffer.clear()
                                speech_chunks = 0
                                await set_state("listening")
                        else:
                            barge_confirm = 0
                    elif ev["speech_start"]:
                        audio_buffer.clear()
                        speech_chunks = 0
                        await set_state("listening")

                    if state == "listening":
                        audio_buffer.append(chunk.copy())
                        speech_chunks += 1

                    if ev["speech_end"] and state == "listening":
                        if speech_chunks >= 10:
                            full_audio = np.concatenate(audio_buffer)
                            audio_buffer.clear()
                            speech_chunks = 0
                            cancel.clear()
                            speaking_task = asyncio.create_task(safe_handle_speech(full_audio))
                        else:
                            audio_buffer.clear()
                            speech_chunks = 0
                            await set_state("idle")

            elif "text" in data and data["text"]:
                try:
                    msg = json.loads(data["text"])
                except json.JSONDecodeError:
                    continue
                if msg.get("type") == "reset":
                    cancel.set()
                    vad.reset()
                    audio_buffer.clear()
                    await set_state("idle")

    except WebSocketDisconnect:
        log.info("Disconnected")
    except Exception as e:
        log.error("WS error: %s", e, exc_info=True)
    finally:
        cancel.set()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
