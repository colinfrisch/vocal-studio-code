"""
Assistant Vocal de Code
Utilise Gradium.ai pour la reconnaissance vocale, OpenAI GPT pour l'édition de code,
Gradio pour l'interface et LangGraph pour l'architecture.
"""

import os
import json
import asyncio
import base64
import ssl
from datetime import datetime
from typing import TypedDict, List, Optional
from dotenv import load_dotenv

import certifi
import gradio as gr
import numpy as np
import websockets
from openai import OpenAI

from langgraph.graph import StateGraph, END

# ================== CONFIG ==================

load_dotenv()

GRADIUM_API_KEY = os.getenv("GRADIUM_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# SSL context (CA bundle) pour éviter CERTIFICATE_VERIFY_FAILED sur macOS
SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())

# ================== LANGGRAPH STATE & NODES ==================

class CodeAssistantState(TypedDict):
    code: str
    instruction: str
    modifications: List[str]
    error: Optional[str]


def parse_instruction(state: CodeAssistantState) -> CodeAssistantState:
    instruction = (state.get("instruction") or "").strip()
    if not instruction:
        return {**state, "error": "Aucune instruction reçue"}
    return {**state, "instruction": instruction, "error": None}


def generate_code_modification(state: CodeAssistantState) -> CodeAssistantState:
    try:
        if not OPENAI_API_KEY:
            return {**state, "error": "OPENAI_API_KEY non configurée"}

        client = OpenAI(api_key=OPENAI_API_KEY)

        current_code = state.get("code", "") or ""
        instruction = state.get("instruction", "") or ""

        messages = [
            {
                "role": "system",
                "content": (
                    "Tu es un assistant de programmation expert. Tu reçois du code et une instruction.\n"
                    "Tu modifies le code selon l'instruction et retournes UNIQUEMENT le code modifié, sans explication.\n"
                    "Si le code est vide et qu'on te demande de créer quelque chose, crée le code demandé.\n"
                    "Retourne uniquement le code, pas de markdown, pas de ```."
                ),
            },
            {
                "role": "user",
                "content": f"Code actuel:\n{current_code if current_code else '(vide)'}\n\n"
                           f"Instruction: {instruction}\n\n"
                           "Retourne uniquement le code modifié:",
            },
        ]

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.1,
        )

        new_code = (response.choices[0].message.content or "").strip()

        if new_code.startswith("```"):
            lines = new_code.split("\n")
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            new_code = "\n".join(lines).strip()

        timestamp = datetime.now().strftime("%H:%M:%S")
        modification_summary = f"[{timestamp}] {instruction}"

        modifications = list(state.get("modifications", []))
        modifications.append(modification_summary)

        return {**state, "code": new_code, "modifications": modifications, "error": None}

    except Exception as e:
        return {**state, "error": f"Erreur OpenAI: {str(e)}"}


def should_continue(state: CodeAssistantState) -> str:
    if state.get("error"):
        return "error"
    if not (state.get("instruction") or "").strip():
        return "error"
    return "continue"


def create_code_assistant_graph():
    workflow = StateGraph(CodeAssistantState)
    workflow.add_node("parse", parse_instruction)
    workflow.add_node("generate", generate_code_modification)
    workflow.set_entry_point("parse")

    workflow.add_conditional_edges(
        "parse",
        should_continue,
        {
            "continue": "generate",
            "error": END,
        },
    )

    workflow.add_edge("generate", END)
    return workflow.compile()


code_graph = create_code_assistant_graph()

# ================== GRADIUM.AI SPEECH-TO-TEXT ==================

async def transcribe_with_gradium(audio_data: bytes) -> str:
    if not GRADIUM_API_KEY:
        return "Erreur: GRADIUM_API_KEY non configurée"

    ws_url = "wss://eu.api.gradium.ai/api/speech/asr"

    try:
        async with websockets.connect(
            ws_url,
            additional_headers={"x-api-key": GRADIUM_API_KEY},
            ssl=SSL_CONTEXT,  # <-- FIX CERTIFICATS
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

# ================== MAIN FUNCTIONS ==================

def _append_history(history: str, entry: str) -> str:
    history = history or ""
    if not entry:
        return history
    return (history + "\n" + entry).strip() if history else entry


def process_voice_instruction(audio, current_code: str, modifications_history: str):
    if audio is None:
        return current_code, "Aucun audio reçu", modifications_history

    instruction = transcribe_audio(audio)

    if not instruction or instruction.startswith("Erreur"):
        return current_code, instruction or "Transcription vide", modifications_history

    initial_state: CodeAssistantState = {
        "code": current_code or "",
        "instruction": instruction,
        "modifications": [],
        "error": None,
    }

    try:
        result = code_graph.invoke(initial_state)

        if result.get("error"):
            return current_code, f"Erreur: {result['error']}", modifications_history

        new_code = result.get("code", current_code) or current_code
        mods = result.get("modifications", []) or []
        if mods:
            modifications_history = _append_history(modifications_history, mods[-1])

        return new_code, f"✓ {instruction}", modifications_history

    except Exception as e:
        return current_code, f"Erreur: {str(e)}", modifications_history


def apply_text_instruction(instruction: str, current_code: str, modifications_history: str):
    instruction = (instruction or "").strip()
    if not instruction:
        return current_code, "Aucune instruction", modifications_history

    initial_state: CodeAssistantState = {
        "code": current_code or "",
        "instruction": instruction,
        "modifications": [],
        "error": None,
    }

    try:
        result = code_graph.invoke(initial_state)

        if result.get("error"):
            return current_code, f"Erreur: {result['error']}", modifications_history

        new_code = result.get("code", current_code) or current_code
        mods = result.get("modifications", []) or []
        if mods:
            modifications_history = _append_history(modifications_history, mods[-1])

        return new_code, f"✓ {instruction}", modifications_history

    except Exception as e:
        return current_code, f"Erreur: {str(e)}", modifications_history


def clear_all():
    return "", "En attente d'instruction...", ""


def build_audio_component():
    try:
        return gr.Audio(
            sources=["microphone", "upload"],
            type="numpy",
            label="Cliquez pour enregistrer",
            streaming=False,
        )
    except TypeError:
        return gr.Audio(
            source="microphone",
            type="numpy",
            label="Cliquez pour enregistrer",
        )

# ================== UI (GRADIO) ==================

custom_css = """
.container { max-width: 1400px; margin: auto; }
.code-editor { font-family: 'Fira Code', 'Monaco', 'Consolas', monospace !important; font-size: 14px !important; min-height: 500px !important; }
.modifications-log { font-family: 'Fira Code', monospace; font-size: 12px; background-color: #1a1a2e; color: #00ff88; padding: 10px; border-radius: 8px; min-height: 150px; }
.status-box { padding: 10px; border-radius: 8px; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); color: #00ff88; }
"""

with gr.Blocks(title="Assistant Vocal de Code", css=custom_css) as demo:
    gr.HTML("""
        <div style="text-align: center; padding: 20px;">
            <h1 style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                       font-size: 2.5em; font-weight: bold;">
                Assistant Vocal de Code
            </h1>
            <p style="color: #888; font-size: 1.1em;">
                Parlez pour modifier votre code • Propulsé par Gradium.ai + OpenAI GPT
            </p>
        </div>
    """)

    with gr.Row():
        with gr.Column(scale=3):
            code_editor = gr.Code(
                label="Éditeur de Code",
                language="python",
                lines=25,
                value="# Parlez ou tapez une instruction pour créer/modifier du code\n",
                elem_classes=["code-editor"],
            )

        with gr.Column(scale=1):
            gr.HTML("<h3 style='color: #667eea;'>Enregistrement Vocal</h3>")
            audio_input = build_audio_component()

            status_display = gr.Textbox(
                label="Statut",
                value="En attente d'instruction...",
                lines=2,
                interactive=False,
                elem_classes=["status-box"],
            )

            mic_test_btn = gr.Button("Tester l'accès micro", variant="secondary")
            mic_test_btn.click(
                fn=lambda: "Test micro lancé…",
                outputs=status_display,
                js="""
                async () => {
                  try {
                    await navigator.mediaDevices.getUserMedia({ audio: true });
                    return "✓ Microphone autorisé par le navigateur";
                  } catch (e) {
                    return "✗ Microphone bloqué: " + (e && e.name ? e.name : e);
                  }
                }
                """,
            )

            gr.HTML("<h3 style='color: #667eea; margin-top: 20px;'>Instruction Textuelle</h3>")
            text_instruction = gr.Textbox(
                label="Tapez votre instruction",
                placeholder="Ex: Ajoute une gestion d'erreur...",
                lines=2,
            )

            apply_text_btn = gr.Button("Appliquer", variant="primary")
            clear_btn = gr.Button("Tout effacer", variant="secondary")

    gr.HTML("<h3 style='color: #667eea; margin-top: 20px;'>Historique des Modifications</h3>")
    modifications_display = gr.Textbox(
        label="",
        value="",
        lines=6,
        interactive=False,
        elem_classes=["modifications-log"],
        placeholder="Les modifications apparaîtront ici...",
    )

    audio_input.change(
        fn=process_voice_instruction,
        inputs=[audio_input, code_editor, modifications_display],
        outputs=[code_editor, status_display, modifications_display],
    )

    apply_text_btn.click(
        fn=apply_text_instruction,
        inputs=[text_instruction, code_editor, modifications_display],
        outputs=[code_editor, status_display, modifications_display],
    ).then(
        fn=lambda: "",
        outputs=[text_instruction],
    )

    clear_btn.click(
        fn=clear_all,
        outputs=[code_editor, status_display, modifications_display],
    )

if __name__ == "__main__":
    if not GRADIUM_API_KEY:
        print("⚠️  GRADIUM_API_KEY non trouvée dans .env")
    if not OPENAI_API_KEY:
        print("⚠️  OPENAI_API_KEY non trouvée dans .env")

    print("🚀 Démarrage de l'Assistant Vocal de Code...")
    demo.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7860,
    )
