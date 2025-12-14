"""
Assistant Vocal de Code
Utilise Gradium.ai pour la reconnaissance vocale, OpenAI GPT pour l'édition de code,
Gradio pour l'interface et LangGraph pour l'architecture.
"""

import os
import json
import asyncio
import base64
import threading
import queue
from datetime import datetime
from typing import TypedDict, Annotated, List, Optional
from dotenv import load_dotenv

import gradio as gr
import numpy as np
import websockets
from openai import OpenAI

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage

# Charger les variables d'environnement
load_dotenv()

# Configuration
GRADIUM_API_KEY = os.getenv("GRADIUM_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# État global pour le streaming audio
audio_queue = queue.Queue()
is_recording = False
recording_thread = None


# ============== LANGGRAPH STATE & NODES ==============

class CodeAssistantState(TypedDict):
    """État du graphe LangGraph pour l'assistant de code."""
    code: str
    instruction: str
    modifications: List[str]
    error: Optional[str]


def parse_instruction(state: CodeAssistantState) -> CodeAssistantState:
    """Analyse l'instruction vocale."""
    instruction = state.get("instruction", "").strip()
    if not instruction:
        return {**state, "error": "Aucune instruction reçue"}
    return state


def generate_code_modification(state: CodeAssistantState) -> CodeAssistantState:
    """Utilise OpenAI GPT pour modifier le code selon l'instruction."""
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        current_code = state.get("code", "")
        instruction = state.get("instruction", "")
        
        # Prompt pour GPT
        messages = [
            {
                "role": "system",
                "content": """Tu es un assistant de programmation expert. Tu reçois du code et une instruction vocale.
Tu dois modifier le code selon l'instruction et retourner UNIQUEMENT le code modifié, sans explication.
Si le code est vide et qu'on te demande de créer quelque chose, crée le code demandé.
Retourne uniquement le code, pas de markdown, pas de ```."""
            },
            {
                "role": "user", 
                "content": f"""Code actuel:
{current_code if current_code else "(vide)"}

Instruction vocale: {instruction}

Retourne uniquement le code modifié:"""
            }
        ]
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.1
        )
        
        new_code = response.choices[0].message.content.strip()
        
        # Nettoyer le code si il contient des marqueurs markdown
        if new_code.startswith("```"):
            lines = new_code.split("\n")
            # Enlever la première ligne (```python ou ```)
            lines = lines[1:]
            # Enlever la dernière ligne si c'est ```
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            new_code = "\n".join(lines)
        
        # Créer le résumé de modification
        timestamp = datetime.now().strftime("%H:%M:%S")
        modification_summary = f"[{timestamp}] {instruction}"
        
        modifications = state.get("modifications", [])
        modifications.append(modification_summary)
        
        return {
            **state,
            "code": new_code,
            "modifications": modifications,
            "error": None
        }
        
    except Exception as e:
        return {**state, "error": f"Erreur OpenAI: {str(e)}"}


def should_continue(state: CodeAssistantState) -> str:
    """Détermine si on continue ou s'arrête."""
    if state.get("error"):
        return "error"
    if not state.get("instruction"):
        return "error"
    return "continue"


# Création du graphe LangGraph
def create_code_assistant_graph():
    """Crée le graphe LangGraph pour l'assistant de code."""
    workflow = StateGraph(CodeAssistantState)
    
    # Ajouter les nœuds
    workflow.add_node("parse", parse_instruction)
    workflow.add_node("generate", generate_code_modification)
    
    # Définir le point d'entrée
    workflow.set_entry_point("parse")
    
    # Ajouter les transitions conditionnelles
    workflow.add_conditional_edges(
        "parse",
        should_continue,
        {
            "continue": "generate",
            "error": END
        }
    )
    
    workflow.add_edge("generate", END)
    
    return workflow.compile()


# Instance globale du graphe
code_graph = create_code_assistant_graph()


# ============== GRADIUM.AI SPEECH-TO-TEXT ==============

async def transcribe_with_gradium(audio_data: bytes) -> str:
    """Transcrit l'audio en texte avec Gradium.ai."""
    if not GRADIUM_API_KEY:
        return "Erreur: GRADIUM_API_KEY non configurée"
    
    ws_url = "wss://eu.api.gradium.ai/api/speech/asr"
    
    try:
        async with websockets.connect(
            ws_url,
            additional_headers={"x-api-key": GRADIUM_API_KEY}
        ) as websocket:
            # Message de configuration
            setup_message = {
                "type": "setup",
                "model_name": "default",
                "input_format": "pcm",
                "language": "fr"
            }
            await websocket.send(json.dumps(setup_message))
            
            # Attendre le message 'ready'
            response = await websocket.recv()
            response_data = json.loads(response)
            if response_data.get('type') != 'ready':
                return f"Erreur: serveur non prêt - {response_data}"
            
            # Encoder et envoyer l'audio
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            audio_message = {
                "type": "audio",
                "audio": audio_base64
            }
            await websocket.send(json.dumps(audio_message))
            
            # Signaler la fin du stream
            await websocket.send(json.dumps({"type": "end_of_stream"}))
            
            # Collecter les résultats
            full_text = ""
            while True:
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                    response_data = json.loads(response)
                    
                    if response_data.get('type') == 'text':
                        text = response_data.get('text', '')
                        if text:
                            full_text += text + " "
                    elif response_data.get('type') == 'final_text':
                        text = response_data.get('text', '')
                        if text:
                            full_text = text
                        break
                    elif response_data.get('type') == 'end_of_stream':
                        break
                    elif response_data.get('type') == 'error':
                        return f"Erreur Gradium: {response_data.get('message', 'Unknown error')}"
                except asyncio.TimeoutError:
                    break
            
            return full_text.strip()
            
    except Exception as e:
        return f"Erreur de transcription: {str(e)}"


def transcribe_audio(audio) -> str:
    """Wrapper synchrone pour la transcription."""
    if audio is None:
        return ""
    
    sample_rate, audio_data = audio
    
    # Convertir en PCM 16-bit mono
    if audio_data.dtype != np.int16:
        if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
            audio_data = (audio_data * 32767).astype(np.int16)
        else:
            audio_data = audio_data.astype(np.int16)
    
    # Si stéréo, convertir en mono
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1).astype(np.int16)
    
    # Rééchantillonner à 24kHz si nécessaire (Gradium requiert 24kHz)
    if sample_rate != 24000:
        from scipy import signal
        num_samples = int(len(audio_data) * 24000 / sample_rate)
        audio_data = signal.resample(audio_data, num_samples).astype(np.int16)
    
    audio_bytes = audio_data.tobytes()
    
    # Exécuter la transcription async
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(transcribe_with_gradium(audio_bytes))
    finally:
        loop.close()
    
    return result


# ============== FONCTIONS PRINCIPALES ==============

def process_voice_instruction(audio, current_code: str, modifications_history: str) -> tuple:
    """
    Traite une instruction vocale et modifie le code en conséquence.
    
    Returns: (nouveau_code, nouvelle_instruction, nouvel_historique)
    """
    if audio is None:
        return current_code, "Aucun audio reçu", modifications_history
    
    # 1. Transcrire l'audio
    instruction = transcribe_audio(audio)
    
    if not instruction or instruction.startswith("Erreur"):
        return current_code, instruction or "Transcription vide", modifications_history
    
    # 2. Exécuter le graphe LangGraph
    initial_state = {
        "code": current_code,
        "instruction": instruction,
        "modifications": [],
        "error": None
    }
    
    try:
        result = code_graph.invoke(initial_state)
        
        if result.get("error"):
            return current_code, f"Erreur: {result['error']}", modifications_history
        
        new_code = result.get("code", current_code)
        new_modifications = result.get("modifications", [])
        
        # Mettre à jour l'historique
        if new_modifications:
            if modifications_history:
                new_history = modifications_history + "\n" + new_modifications[-1]
            else:
                new_history = new_modifications[-1]
        else:
            new_history = modifications_history
        
        return new_code, f"✓ {instruction}", new_history
        
    except Exception as e:
        return current_code, f"Erreur: {str(e)}", modifications_history


def clear_all():
    """Efface tout et remet à zéro."""
    return "", "", ""


def apply_text_instruction(instruction: str, current_code: str, modifications_history: str) -> tuple:
    """Applique une instruction textuelle (alternative au vocal)."""
    if not instruction.strip():
        return current_code, "Aucune instruction", modifications_history
    
    initial_state = {
        "code": current_code,
        "instruction": instruction.strip(),
        "modifications": [],
        "error": None
    }
    
    try:
        result = code_graph.invoke(initial_state)
        
        if result.get("error"):
            return current_code, f"Erreur: {result['error']}", modifications_history
        
        new_code = result.get("code", current_code)
        new_modifications = result.get("modifications", [])
        
        if new_modifications:
            if modifications_history:
                new_history = modifications_history + "\n" + new_modifications[-1]
            else:
                new_history = new_modifications[-1]
        else:
            new_history = modifications_history
        
        return new_code, f"✓ {instruction}", new_history
        
    except Exception as e:
        return current_code, f"Erreur: {str(e)}", modifications_history


# ============== INTERFACE GRADIO ==============

# CSS personnalisé pour une interface moderne
custom_css = """
.container {
    max-width: 1400px;
    margin: auto;
}
.code-editor {
    font-family: 'Fira Code', 'Monaco', 'Consolas', monospace !important;
    font-size: 14px !important;
    min-height: 500px !important;
}
.modifications-log {
    font-family: 'Fira Code', monospace;
    font-size: 12px;
    background-color: #1a1a2e;
    color: #00ff88;
    padding: 10px;
    border-radius: 8px;
    min-height: 150px;
}
.status-box {
    padding: 10px;
    border-radius: 8px;
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    color: #00ff88;
}
.record-btn {
    background: linear-gradient(135deg, #ff6b6b 0%, #ee5a5a 100%) !important;
    border: none !important;
    font-size: 18px !important;
    padding: 20px 40px !important;
}
.title-text {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.5em;
    font-weight: bold;
    text-align: center;
}
"""

# Créer l'interface
with gr.Blocks(title="🎤 Assistant Vocal de Code") as demo:
    
    gr.HTML("""
        <div style="text-align: center; padding: 20px;">
            <h1 style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                       font-size: 2.5em; font-weight: bold;">
                🎤 Assistant Vocal de Code
            </h1>
            <p style="color: #888; font-size: 1.1em;">
                Parlez pour modifier votre code • Propulsé par Gradium.ai + OpenAI GPT
            </p>
        </div>
    """)
    
    with gr.Row():
        # Colonne principale - Éditeur de code
        with gr.Column(scale=3):
            code_editor = gr.Code(
                label="📝 Éditeur de Code",
                language="python",
                lines=25,
                value="# Commencez à parler pour créer ou modifier du code\n# Exemple: \"Crée une fonction qui calcule la factorielle d'un nombre\"\n",
                elem_classes=["code-editor"]
            )
        
        # Colonne latérale - Contrôles et historique
        with gr.Column(scale=1):
            # Section enregistrement vocal
            gr.HTML("<h3 style='color: #667eea;'>🎙️ Enregistrement Vocal</h3>")
            
            audio_input = gr.Audio(
                sources=["microphone"],
                type="numpy",
                label="Cliquez pour enregistrer",
                streaming=False
            )
            
            status_display = gr.Textbox(
                label="📊 Statut",
                value="En attente d'instruction...",
                lines=2,
                interactive=False,
                elem_classes=["status-box"]
            )
            
            # Section instruction textuelle (alternative)
            gr.HTML("<h3 style='color: #667eea; margin-top: 20px;'>⌨️ Instruction Textuelle</h3>")
            
            text_instruction = gr.Textbox(
                label="Tapez votre instruction",
                placeholder="Ex: Ajoute une gestion d'erreur...",
                lines=2
            )
            
            apply_text_btn = gr.Button("🚀 Appliquer", variant="primary")
            
            # Bouton reset
            clear_btn = gr.Button("🗑️ Tout effacer", variant="secondary")
    
    # Section historique des modifications
    gr.HTML("<h3 style='color: #667eea; margin-top: 20px;'>📋 Historique des Modifications</h3>")
    
    modifications_display = gr.Textbox(
        label="",
        value="",
        lines=6,
        interactive=False,
        elem_classes=["modifications-log"],
        placeholder="Les modifications apparaîtront ici..."
    )
    
    # Événements
    audio_input.change(
        fn=process_voice_instruction,
        inputs=[audio_input, code_editor, modifications_display],
        outputs=[code_editor, status_display, modifications_display]
    )
    
    apply_text_btn.click(
        fn=apply_text_instruction,
        inputs=[text_instruction, code_editor, modifications_display],
        outputs=[code_editor, status_display, modifications_display]
    ).then(
        fn=lambda: "",
        outputs=[text_instruction]
    )
    
    clear_btn.click(
        fn=clear_all,
        outputs=[code_editor, status_display, modifications_display]
    )
    
    # Instructions d'utilisation
    gr.HTML("""
        <div style="margin-top: 30px; padding: 20px; background: #f8f9fa; border-radius: 10px;">
            <h4 style="color: #667eea;">💡 Comment utiliser</h4>
            <ul style="color: #555;">
                <li>🎤 <strong>Vocal:</strong> Cliquez sur le micro, parlez, puis arrêtez l'enregistrement</li>
                <li>⌨️ <strong>Texte:</strong> Tapez votre instruction et cliquez sur "Appliquer"</li>
                <li>📋 <strong>Historique:</strong> Chaque modification est enregistrée avec son timestamp</li>
            </ul>
            <p style="color: #888; font-size: 0.9em; margin-top: 10px;">
                Exemples d'instructions: "Crée une classe User avec nom et email", 
                "Ajoute une méthode pour valider l'email", "Refactorise avec des type hints"
            </p>
        </div>
    """)


if __name__ == "__main__":
    # Vérifier les clés API
    if not GRADIUM_API_KEY:
        print("⚠️  GRADIUM_API_KEY non trouvée dans .env")
    if not OPENAI_API_KEY:
        print("⚠️  OPENAI_API_KEY non trouvée dans .env")
    
    print("🚀 Démarrage de l'Assistant Vocal de Code...")
    demo.launch(
        share=False, 
        server_name="0.0.0.0", 
        server_port=7860,
        css=custom_css,
        theme=gr.themes.Soft(
            primary_hue="purple",
            secondary_hue="blue", 
            neutral_hue="slate"
        )
    )
