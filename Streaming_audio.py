# app.py — stable (Gradio stream + Gradium WS) avec queue hors gr.State
import os
import json
import base64
import asyncio
import logging
import threading
import queue
import ssl
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import gradio as gr
import websockets
import certifi
from dotenv import load_dotenv

# =========================
# LOGGING
# =========================
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

load_dotenv()

# =========================
# CONFIG
# =========================
GRADIUM_API_KEY = os.getenv("GRADIUM_API_KEY", "").strip()
if not GRADIUM_API_KEY:
    raise RuntimeError("GRADIUM_API_KEY manquante. Mets-la dans ton .env (GRADIUM_API_KEY=...).")

GRADIUM_REGION = os.getenv("GRADIUM_REGION", "eu").strip().lower()
GRADIUM_MODEL_NAME = os.getenv("GRADIUM_MODEL_NAME", "default").strip()

GRADIUM_WS_URL = (
    "wss://eu.api.gradium.ai/api/speech/asr"
    if GRADIUM_REGION == "eu"
    else "wss://us.api.gradium.ai/api/speech/asr"
)

TARGET_SR = 24000
FRAME_SAMPLES = 1920  # 80ms @ 24kHz
MIN_UTTERANCE_SECONDS = 0.25

DEFAULT_SILENCE_SECONDS = 0.70
DEFAULT_RMS_THRESHOLD = 0.015

SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())

# =========================
# GLOBAL QUEUE REGISTRY (HORS gr.State)
# =========================
_Q_REGISTRY: Dict[str, "queue.Queue[str]"] = {}
_Q_LOCK = threading.Lock()

def get_queue(session_id: str) -> "queue.Queue[str]":
    with _Q_LOCK:
        q = _Q_REGISTRY.get(session_id)
        if q is None:
            q = queue.Queue()
            _Q_REGISTRY[session_id] = q
        return q

def reset_queue(session_id: str):
    with _Q_LOCK:
        _Q_REGISTRY[session_id] = queue.Queue()

# =========================
# AUDIO UTILS
# =========================
def to_mono(x: np.ndarray) -> np.ndarray:
    if x is None:
        return np.zeros((0,), dtype=np.float32)
    if x.ndim == 1:
        return x
    if x.ndim == 2:
        return x.mean(axis=1)
    return x.reshape(-1)

def resample_linear(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x.astype(np.float32, copy=False)
    x = x.astype(np.float32, copy=False)
    n_in = x.shape[0]
    if n_in == 0:
        return x
    duration = n_in / float(sr_in)
    n_out = int(round(duration * sr_out))
    if n_out <= 1:
        return np.zeros((0,), dtype=np.float32)

    t_in = np.linspace(0.0, duration, num=n_in, endpoint=False, dtype=np.float32)
    t_out = np.linspace(0.0, duration, num=n_out, endpoint=False, dtype=np.float32)
    return np.interp(t_out, t_in, x).astype(np.float32)

def float_to_pcm16le(x: np.ndarray) -> bytes:
    x = np.clip(x, -1.0, 1.0)
    y = (x * 32767.0).astype(np.int16)
    return y.tobytes(order="C")

def chunk_pcm16(pcm_bytes: bytes, frame_samples: int = FRAME_SAMPLES) -> List[bytes]:
    frame_bytes = frame_samples * 2
    out = []
    for i in range(0, len(pcm_bytes), frame_bytes):
        j = i + frame_bytes
        if j <= len(pcm_bytes):
            out.append(pcm_bytes[i:j])
    return out

def rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    xf = x.astype(np.float32, copy=False)
    return float(np.sqrt(np.mean(xf * xf) + 1e-12))

# =========================
# GRADIUM WS (async)
# =========================
def ws_connect(url: str, headers):
    try:
        return websockets.connect(
            url,
            additional_headers=headers,  # websockets >= 15
            ssl=SSL_CONTEXT,
            ping_interval=20,
            ping_timeout=20,
        )
    except TypeError:
        return websockets.connect(
            url,
            extra_headers=headers,  # websockets < 15
            ssl=SSL_CONTEXT,
            ping_interval=20,
            ping_timeout=20,
        )

async def gradium_transcribe_and_push(pcm16le_24k_mono: bytes, out_q: "queue.Queue[str]"):
    headers = [("x-api-key", GRADIUM_API_KEY)]
    logging.info("Gradium: opening WS...")
    try:
        async with ws_connect(GRADIUM_WS_URL, headers) as ws:
            logging.info("Gradium: WS connected.")

            await ws.send(json.dumps({
                "type": "setup",
                "model_name": GRADIUM_MODEL_NAME,
                "input_format": "pcm",
            }))

            while True:
                raw = await ws.recv()
                msg = json.loads(raw)
                if msg.get("type") == "ready":
                    break
                if msg.get("type") == "error":
                    raise RuntimeError(f"Gradium error (setup): {msg.get('message')} (code={msg.get('code')})")

            frames = chunk_pcm16(pcm16le_24k_mono, FRAME_SAMPLES)
            for frame in frames:
                await ws.send(json.dumps({
                    "type": "audio",
                    "audio": base64.b64encode(frame).decode("ascii")
                }))
                await asyncio.sleep(0)

            await ws.send(json.dumps({"type": "end_of_stream"}))

            parts: List[str] = []
            last = ""

            while True:
                raw = await ws.recv()
                msg = json.loads(raw)
                t = msg.get("type")

                if t == "text":
                    txt = (msg.get("text") or "").strip()
                    if txt and txt != last:
                        parts.append(txt)
                        last = txt
                        out_q.put(" ".join(parts).strip())

                elif t == "error":
                    raise RuntimeError(f"Gradium error: {msg.get('message')} (code={msg.get('code')})")

                elif t in ("end_of_stream", "final", "done"):
                    break

            out_q.put("__DONE__")

    except Exception as e:
        logging.error(f"Gradium: transcription failed: {e}")
        out_q.put(f"[ERREUR] {e}")
        out_q.put("__DONE__")

def start_gradium_thread(pcm: bytes, out_q: "queue.Queue[str]"):
    def runner():
        asyncio.run(gradium_transcribe_and_push(pcm, out_q))
    th = threading.Thread(target=runner, daemon=True)
    th.start()
    return th

# =========================
# STATE (DEEP-COPY FRIENDLY)
# =========================
@dataclass
class StreamState:
    session_id: str = ""
    buffer_chunks: List[List[float]] = field(default_factory=list)  # store as python lists
    silence_s: float = 0.0
    saw_voice: bool = False
    in_flight: bool = False
    chunk_count: int = 0
    last_text: str = ""

def new_state() -> StreamState:
    return StreamState(session_id=str(uuid.uuid4()))

# Helpers to convert safely
def append_chunk(state: StreamState, chunk_f: np.ndarray):
    # store as list to avoid numpy object deepcopy issues
    state.buffer_chunks.append(chunk_f.astype(np.float32, copy=False).tolist())

def concat_chunks(state: StreamState) -> np.ndarray:
    if not state.buffer_chunks:
        return np.zeros((0,), dtype=np.float32)
    arrays = [np.asarray(c, dtype=np.float32) for c in state.buffer_chunks]
    return np.concatenate(arrays, axis=0)

# =========================
# AUDIO CALLBACK (FAST)
# =========================
async def on_audio(
    audio: Optional[Tuple[int, np.ndarray]],
    st: StreamState,
    silence_seconds: float,
    rms_threshold: float
):
    state = st or new_state()

    if audio is None:
        return "", state

    sr_in, chunk = audio
    chunk = to_mono(chunk)

    if np.issubdtype(chunk.dtype, np.integer):
        max_val = np.iinfo(chunk.dtype).max
        chunk_f = (chunk.astype(np.float32) / float(max_val)).astype(np.float32)
    else:
        chunk_f = np.clip(chunk.astype(np.float32, copy=False), -1.0, 1.0)

    state.chunk_count += 1
    current_rms = rms(chunk_f)
    is_voice = current_rms >= float(rms_threshold)

    logging.debug(
        f"Chunk #{state.chunk_count}: RMS={current_rms:.4f}, Voice={is_voice}, Threshold={float(rms_threshold):.4f}"
    )

    if state.in_flight:
        # on ne bufferise pas pendant la transcription
        return "", state

    append_chunk(state, chunk_f)
    dur_chunk = float(chunk_f.shape[0]) / float(sr_in) if sr_in else 0.0

    if is_voice:
        if not state.saw_voice:
            logging.info("VOIX DÉTECTÉE. Démarrage de l'énoncé.")
        state.saw_voice = True
        state.silence_s = 0.0
    elif state.saw_voice:
        state.silence_s += dur_chunk
        logging.debug(f"Silence: {state.silence_s:.3f}s / {float(silence_seconds):.3f}s")

    if not state.saw_voice:
        return "", state

    if state.silence_s >= float(silence_seconds):
        logging.info(f"Silence atteint ({state.silence_s:.3f}s). DÉCLENCHEMENT DE L'ENVOI.")

        full = concat_chunks(state)

        # reset VAD
        state.buffer_chunks = []
        state.silence_s = 0.0
        state.saw_voice = False

        full_24k = resample_linear(full, sr_in=int(sr_in), sr_out=TARGET_SR)
        utt_seconds = float(full_24k.shape[0]) / float(TARGET_SR) if full_24k.size else 0.0

        if utt_seconds < MIN_UTTERANCE_SECONDS:
            logging.warning(f"Audio trop court ({utt_seconds:.3f}s). Annulation.")
            return "", state

        pcm = float_to_pcm16le(full_24k)
        logging.info(f"Envoi de {utt_seconds:.3f}s (PCM: {len(pcm)} bytes) à Gradium.")

        # reset queue for this session utterance
        reset_queue(state.session_id)
        q = get_queue(state.session_id)

        state.in_flight = True
        start_gradium_thread(pcm, q)

    return "", state

# =========================
# POLLING (Timer) — UI streaming
# =========================
def poll_transcript(st: StreamState):
    state = st or new_state()
    q = get_queue(state.session_id)

    updated = False
    while True:
        try:
            msg = q.get_nowait()
        except queue.Empty:
            break

        if msg == "__DONE__":
            state.in_flight = False
            break

        state.last_text = msg
        updated = True

    if updated:
        return state.last_text, state

    return state.last_text, state

def on_reset(st: StreamState):
    state = new_state()
    reset_queue(state.session_id)
    return "", state

# =========================
# UI
# =========================
with gr.Blocks(title="Gradium STT (silence-triggered)") as demo:
    gr.Markdown("## Gradium Speech-to-Text (micro → silence → envoi → transcript streaming)")

    with gr.Row():
        with gr.Column(scale=1):
            audio_in = gr.Audio(
                sources=["microphone"],
                type="numpy",
                streaming=True,
                label="Microphone (streaming)"
            )
            silence_seconds = gr.Slider(
                minimum=0.30, maximum=1.50, value=DEFAULT_SILENCE_SECONDS, step=0.05,
                label="Silence (s) avant envoi"
            )
            rms_threshold = gr.Slider(
                minimum=0.003, maximum=0.060, value=DEFAULT_RMS_THRESHOLD, step=0.001,
                label="Seuil RMS (détection voix)"
            )
            reset_btn = gr.Button("Reset")

        with gr.Column(scale=1):
            transcript = gr.Textbox(label="Transcription (streaming)", lines=10)
            st = gr.State(new_state())

    audio_in.stream(
        fn=on_audio,
        inputs=[audio_in, st, silence_seconds, rms_threshold],
        outputs=[transcript, st],
        show_progress="hidden",
        concurrency_limit=32,
    )

    timer = gr.Timer(0.1)
    timer.tick(fn=poll_transcript, inputs=[st], outputs=[transcript, st], show_progress="hidden")

    reset_btn.click(fn=on_reset, inputs=[st], outputs=[transcript, st])

if __name__ == "__main__":
    demo.queue()
    demo.launch(server_port=7860)