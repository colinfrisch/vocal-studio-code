"""
Gradium.ai Speech-to-Text.
"""

import json
import asyncio
import base64

import numpy as np
import websockets

from config import GRADIUM_API_KEY, SSL_CONTEXT


async def transcribe_with_gradium(audio_data: bytes) -> str:
    if not GRADIUM_API_KEY:
        return "Erreur: GRADIUM_API_KEY non configurée"

    ws_url = "wss://eu.api.gradium.ai/api/speech/asr"

    try:
        async with websockets.connect(
            ws_url,
            additional_headers={"x-api-key": GRADIUM_API_KEY},
            ssl=SSL_CONTEXT,
        ) as websocket:
            setup_message = {
                "type": "setup",
                "model_name": "default",
                "input_format": "pcm",
                "language": "fr",
            }
            await websocket.send(json.dumps(setup_message))

            response = await websocket.recv()
            response_data = json.loads(response)
            if response_data.get("type") != "ready":
                return f"Erreur: serveur non prêt - {response_data}"

            audio_base64 = base64.b64encode(audio_data).decode("utf-8")
            await websocket.send(json.dumps({"type": "audio", "audio": audio_base64}))
            await websocket.send(json.dumps({"type": "end_of_stream"}))

            full_text = ""
            while True:
                try:
                    msg = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                    data = json.loads(msg)

                    if data.get("type") == "text":
                        text = data.get("text", "")
                        if text:
                            full_text += text + " "
                    elif data.get("type") == "final_text":
                        text = data.get("text", "")
                        if text:
                            full_text = text
                        break
                    elif data.get("type") == "end_of_stream":
                        break
                    elif data.get("type") == "error":
                        return f"Erreur Gradium: {data.get('message', 'Unknown error')}"
                except asyncio.TimeoutError:
                    break

            return full_text.strip()

    except Exception as e:
        return f"Erreur de transcription: {str(e)}"


def transcribe_audio(audio) -> str:
    if audio is None:
        return ""

    sample_rate, audio_data = audio

    if audio_data.dtype != np.int16:
        if audio_data.dtype in (np.float32, np.float64):
            audio_data = (audio_data * 32767).astype(np.int16)
        else:
            audio_data = audio_data.astype(np.int16)

    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1).astype(np.int16)

    if sample_rate != 24000:
        from scipy import signal
        num_samples = int(len(audio_data) * 24000 / sample_rate)
        audio_data = signal.resample(audio_data, num_samples).astype(np.int16)

    audio_bytes = audio_data.tobytes()

    try:
        return asyncio.run(transcribe_with_gradium(audio_bytes))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(transcribe_with_gradium(audio_bytes))
        finally:
            loop.close()
