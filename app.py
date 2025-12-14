"""
Voice Code Assistant - Assistant vocal de programmation
Utilise Gradium pour STT et GPT-4 pour éditer du code en temps réel
"""

import os
import asyncio
import base64
import json
import gradio as gr
from openai import OpenAI
import websockets
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GRADIUM_API_KEY = os.environ.get("GRADIUM_API_KEY", "")
GPT_MODEL = "gpt-4o"
GRADIUM_STT_URL = "wss://eu.api.gradium.ai/api/speech/asr"

# Client OpenAI
openai_client = None
if OPENAI_API_KEY:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    print("✅ Client OpenAI initialisé!")
else:
    print("⚠️ OPENAI_API_KEY non définie. Définissez-la pour utiliser GPT.")

if GRADIUM_API_KEY:
    print("✅ Gradium API Key configurée!")
else:
    print("⚠️ GRADIUM_API_KEY non définie. Définissez-la pour utiliser Gradium STT.")


async def transcribe_with_gradium(audio_data: np.ndarray, sample_rate: int) -> str:
    """Transcrit l'audio en texte avec Gradium STT"""
    if not GRADIUM_API_KEY:
        return ""
    
    # Convertir en int16 si nécessaire
    if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
        audio_data = (audio_data * 32767).astype(np.int16)
    elif audio_data.dtype != np.int16:
        audio_data = audio_data.astype(np.int16)
    
    # Si stéréo, convertir en mono
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1).astype(np.int16)
    
    # Rééchantillonner à 24kHz pour Gradium
    if sample_rate != 24000:
        from scipy import signal
        num_samples = int(len(audio_data) * 24000 / sample_rate)
        audio_data = signal.resample(audio_data, num_samples).astype(np.int16)
    
    headers = {"x-api-key": GRADIUM_API_KEY}
    transcribed_text = ""
    
    try:
        async with websockets.connect(GRADIUM_STT_URL, additional_headers=headers) as websocket:
            # Envoyer le message de setup
            setup_message = {
                "type": "setup",
                "model_name": "default",
                "input_format": "pcm"
            }
            await websocket.send(json.dumps(setup_message))
            
            # Attendre le message 'ready'
            response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            response_data = json.loads(response)
            if response_data.get("type") != "ready":
                print(f"Gradium: message inattendu: {response_data}")
                return ""
            
            # Envoyer l'audio en chunks
            audio_bytes = audio_data.tobytes()
            chunk_size = 1920  # 80ms à 24kHz
            
            for i in range(0, len(audio_bytes), chunk_size):
                chunk = audio_bytes[i:i + chunk_size]
                audio_base64 = base64.b64encode(chunk).decode('utf-8')
                audio_message = {
                    "type": "audio",
                    "audio": audio_base64
                }
                await websocket.send(json.dumps(audio_message))
            
            # Envoyer fin de stream
            await websocket.send(json.dumps({"type": "end_of_stream"}))
            
            # Recevoir les résultats
            while True:
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                    response_data = json.loads(response)
                    
                    if response_data.get("type") == "text":
                        transcribed_text += response_data.get("text", "")
                    elif response_data.get("type") == "final_text":
                        transcribed_text = response_data.get("text", transcribed_text)
                    elif response_data.get("type") == "end_of_stream":
                        break
                except asyncio.TimeoutError:
                    break
                    
    except Exception as e:
        print(f"Erreur Gradium STT: {e}")
        return ""
    
    return transcribed_text.strip()


def transcribe_audio(audio):
    """Transcrit l'audio en texte avec Gradium"""
    if audio is None:
        return ""
    
    sample_rate, audio_data = audio
    
    # Exécuter la transcription async
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(transcribe_with_gradium(audio_data, sample_rate))
        loop.close()
        return result
    except Exception as e:
        print(f"Erreur transcription: {e}")
        return ""


def edit_code_with_gpt(code: str, instruction: str) -> tuple[str, str]:
    """Utilise GPT pour éditer le code selon l'instruction"""
    if not openai_client:
        return code, "❌ Erreur: OPENAI_API_KEY non configurée"
    
    if not instruction.strip():
        return code, "⚠️ Aucune instruction fournie"
    
    system_prompt = """Tu es un assistant de programmation expert. 
Tu reçois du code et une instruction pour le modifier.
Tu dois retourner UNIQUEMENT le code modifié, sans explications, sans markdown, sans balises de code.
Si le code est vide et qu'on te demande de créer quelque chose, génère le code demandé.
Assure-toi que le code est syntaxiquement correct et bien formaté."""

    user_prompt = f"""Voici le code actuel:
```
{code if code.strip() else "(code vide)"}
```

Instruction: {instruction}

Retourne uniquement le code modifié:"""

    try:
        response = openai_client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=4096
        )
        
        new_code = response.choices[0].message.content
        
        # Nettoyer le code (enlever les balises markdown si présentes)
        if new_code.startswith("```"):
            lines = new_code.split("\n")
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            new_code = "\n".join(lines)
        
        return new_code, f"✅ Code modifié selon: \"{instruction}\""
    
    except Exception as e:
        return code, f"❌ Erreur GPT: {str(e)}"


def process_voice_instruction(audio, current_code):
    """Traite une instruction vocale et modifie le code"""
    if audio is None:
        return current_code, "⚠️ Aucun audio enregistré", ""
    
    if not GRADIUM_API_KEY:
        return current_code, "❌ Erreur: GRADIUM_API_KEY non configurée", ""
    
    # Transcrire l'audio
    instruction = transcribe_audio(audio)
    
    if not instruction.strip():
        return current_code, "⚠️ Aucune instruction détectée dans l'audio", ""
    
    # Modifier le code
    new_code, status = edit_code_with_gpt(current_code, instruction)
    
    return new_code, status, instruction


def process_text_instruction(instruction, current_code):
    """Traite une instruction textuelle et modifie le code"""
    new_code, status = edit_code_with_gpt(current_code, instruction)
    return new_code, status


# Interface Gradio
with gr.Blocks(title="🎤 Voice Code Assistant") as app:
    
    gr.Markdown("""
    # 🎤 Voice Code Assistant
    ### Éditez votre code par la voix avec Gradium STT + GPT-4
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            code_editor = gr.Code(
                label="📝 Éditeur de Code",
                language="python",
                lines=25,
                value='''# Bienvenue dans Voice Code Assistant!
# Utilisez votre voix ou le champ texte pour modifier ce code.

def hello():
    print("Hello, World!")

if __name__ == "__main__":
    hello()
'''
            )
            
            language_selector = gr.Dropdown(
                choices=["python", "javascript", "typescript", "html", "css", "java", "c", "cpp", "rust", "go"],
                value="python",
                label="🌐 Langage",
                scale=1
            )
        
        with gr.Column(scale=1):
            gr.Markdown("### 🎙️ Instruction Vocale")
            audio_input = gr.Audio(
                sources=["microphone"],
                type="numpy",
                label="Cliquez pour enregistrer"
            )
            voice_btn = gr.Button("🎤 Appliquer l'instruction vocale", variant="primary", size="lg")
            
            transcription_output = gr.Textbox(
                label="📝 Transcription (Gradium)",
                placeholder="L'instruction vocale apparaîtra ici...",
                interactive=False
            )
            
            gr.Markdown("---")
            gr.Markdown("### ⌨️ Instruction Textuelle")
            
            text_instruction = gr.Textbox(
                label="Tapez votre instruction",
                placeholder="Ex: Ajoute une fonction qui calcule la factorielle...",
                lines=2
            )
            text_btn = gr.Button("✨ Appliquer l'instruction", variant="secondary", size="lg")
            
            gr.Markdown("---")
            status_output = gr.Textbox(
                label="📊 Statut",
                interactive=False
            )
    
    gr.Markdown("""
    ---
    ### 💡 Exemples d'instructions
    - "Ajoute une fonction qui trie une liste"
    - "Corrige les bugs dans ce code"  
    - "Ajoute des commentaires explicatifs"
    - "Convertis cette fonction en async"
    - "Ajoute la gestion des erreurs"
    """)
    
    # Event handlers
    def update_language(lang):
        return gr.Code(language=lang)
    
    language_selector.change(
        fn=update_language,
        inputs=[language_selector],
        outputs=[code_editor]
    )
    
    voice_btn.click(
        fn=process_voice_instruction,
        inputs=[audio_input, code_editor],
        outputs=[code_editor, status_output, transcription_output]
    )
    
    text_btn.click(
        fn=process_text_instruction,
        inputs=[text_instruction, code_editor],
        outputs=[code_editor, status_output]
    )
    
    text_instruction.submit(
        fn=process_text_instruction,
        inputs=[text_instruction, code_editor],
        outputs=[code_editor, status_output]
    )


if __name__ == "__main__":
    print("\n🚀 Démarrage de Voice Code Assistant...")
    print("📍 Ouvrez http://localhost:7860 dans votre navigateur\n")
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
