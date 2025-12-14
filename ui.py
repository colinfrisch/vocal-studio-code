"""
Interface Gradio pour Vocal Studio Code.
Utilise l'enregistrement continu avec détection de fin de parole (VAD).
"""

import os
import base64
import uuid
import numpy as np
import gradio as gr
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from handlers import process_voice_instruction, three_way_merge

# Charger le logo en base64
APP_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(APP_DIR, "logo.png")

def load_logo_base64():
    try:
        with open(LOGO_PATH, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return ""

LOGO_BASE64 = load_logo_base64()

# =========================
# CONFIG VAD (Voice Activity Detection)
# =========================
TARGET_SR = 24000
DEFAULT_SILENCE_SECONDS = 0.70  # Silence avant envoi
DEFAULT_RMS_THRESHOLD = 0.015   # Seuil de détection de voix
MIN_UTTERANCE_SECONDS = 0.25    # Durée minimale pour envoyer


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


def rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    xf = x.astype(np.float32, copy=False)
    return float(np.sqrt(np.mean(xf * xf) + 1e-12))


# =========================
# STATE pour l'enregistrement continu
# =========================
@dataclass
class VoiceStreamState:
    session_id: str = ""
    buffer_chunks: List[List[float]] = field(default_factory=list)  # Stocké en list pour eviter les problèmes de deepcopy
    buffer_sr: int = 0
    silence_s: float = 0.0
    saw_voice: bool = False
    pending_audio_list: Optional[List[float]] = None  # Audio prêt à traiter (stocké en list)
    pending_audio_sr: int = 0
    status_message: str = "En attente..."


def new_voice_state() -> VoiceStreamState:
    return VoiceStreamState(session_id=str(uuid.uuid4()))


def append_chunk(state: VoiceStreamState, chunk_f: np.ndarray, sr: int):
    state.buffer_chunks.append(chunk_f.astype(np.float32, copy=False).tolist())
    state.buffer_sr = sr


def concat_chunks(state: VoiceStreamState) -> np.ndarray:
    if not state.buffer_chunks:
        return np.zeros((0,), dtype=np.float32)
    arrays = [np.asarray(c, dtype=np.float32) for c in state.buffer_chunks]
    return np.concatenate(arrays, axis=0)


def set_pending_audio(state: VoiceStreamState, sr: int, audio: np.ndarray):
    """Stocke l'audio prêt à traiter (converti en list pour deepcopy)."""
    state.pending_audio_list = audio.astype(np.float32, copy=False).tolist()
    state.pending_audio_sr = sr


def get_pending_audio(state: VoiceStreamState) -> Optional[Tuple[int, np.ndarray]]:
    """Récupère l'audio en attente et le convertit en numpy."""
    if state.pending_audio_list is None:
        return None
    audio = np.asarray(state.pending_audio_list, dtype=np.float32)
    return (state.pending_audio_sr, audio)


def clear_pending_audio(state: VoiceStreamState):
    """Efface l'audio en attente."""
    state.pending_audio_list = None
    state.pending_audio_sr = 0


def build_audio_component():
    """Composant audio en mode streaming continu."""
    try:
        return gr.Audio(
            sources=["microphone"],
            type="numpy",
            label="🎤 Parlez (enregistrement continu)",
            streaming=True,
        )
    except TypeError:
        return gr.Audio(
            source="microphone",
            type="numpy",
            label="🎤 Parlez (enregistrement continu)",
            streaming=True,
        )


MIC_TEST_JS = """
async () => {
  const debug = [];
  const ts = new Date().toISOString();
  debug.push(`[MicDebug ${ts}] Début du test micro`);

  if (!navigator.mediaDevices) {
    const msg = "✗ navigator.mediaDevices indisponible";
    debug.push(msg);
    console.log(debug.join("\\n"));
    return debug.join("\\n");
  }

  const permStatus = await (navigator.permissions?.query?.({ name: "microphone" }).catch(() => null));
  if (permStatus?.state) {
    debug.push(`Etat permission: ${permStatus.state}`);
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const tracks = stream.getAudioTracks();
    debug.push("✓ Autorisation micro accordée");
    debug.push(`Pistes audio: ${tracks.length}`);
    tracks.forEach((t, idx) => {
      debug.push(`  - Track ${idx}: ${t.label || "label inconnu"} (${t.readyState})`);
    });
    // Stopper les tracks pour libérer le device
    tracks.forEach((t) => t.stop());
  } catch (e) {
    debug.push("✗ getUserMedia a échoué");
    debug.push(`  Nom: ${e?.name || "inconnu"}`);
    debug.push(`  Message: ${e?.message || e}`);
  }

  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const audioInputs = devices.filter((d) => d.kind === "audioinput");
    debug.push(`Sources audio détectées: ${audioInputs.length}`);
    audioInputs.forEach((d, idx) => {
      debug.push(`  - ${idx}: ${d.label || "label masqué"} (${d.deviceId})`);
    });
  } catch (e) {
    debug.push("Impossible d'énumérer les devices audio");
    debug.push(`  ${e?.message || e}`);
  }

  console.log(debug.join("\\n"));
  return debug.join("\\n");
}
"""

CUSTOM_CSS = """
.container { max-width: 1400px; margin: auto; }
.code-editor { font-family: 'Fira Code', 'Monaco', 'Consolas', monospace !important; font-size: 14px !important; min-height: 500px !important; }
.modifications-log { font-family: 'Fira Code', monospace; font-size: 12px; background-color: #1a1a2e; color: #00ff88; padding: 10px; border-radius: 8px; min-height: 150px; }
.status-box { padding: 10px; border-radius: 8px; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); color: #00ff88; }
"""


# =========================
# CALLBACK AUDIO STREAMING (VAD)
# =========================
def on_audio_stream(
    audio: Optional[Tuple[int, np.ndarray]],
    voice_state: VoiceStreamState,
    silence_seconds: float,
    rms_threshold: float,
):
    """
    Traite chaque chunk audio en streaming.
    Détecte la voix et le silence pour déclencher l'envoi uniquement à la fin de parole.
    
    Retourne: (voice_state, status_message)
    """
    state = voice_state or new_voice_state()
    
    if audio is None:
        return state, state.status_message
    
    sr_in, chunk = audio
    chunk = to_mono(chunk)
    
    # Normaliser en float32
    if np.issubdtype(chunk.dtype, np.integer):
        max_val = np.iinfo(chunk.dtype).max
        chunk_f = (chunk.astype(np.float32) / float(max_val)).astype(np.float32)
    else:
        chunk_f = np.clip(chunk.astype(np.float32, copy=False), -1.0, 1.0)
    
    current_rms = rms(chunk_f)
    is_voice = current_rms >= float(rms_threshold)
    
    # Accumuler le chunk
    append_chunk(state, chunk_f, sr_in)
    dur_chunk = float(chunk_f.shape[0]) / float(sr_in) if sr_in else 0.0
    
    if is_voice:
        if not state.saw_voice:
            state.status_message = "🎙️ Parole détectée..."
        state.saw_voice = True
        state.silence_s = 0.0
    elif state.saw_voice:
        state.silence_s += dur_chunk
        remaining = max(0, float(silence_seconds) - state.silence_s)
        state.status_message = f"🎙️ Parole en cours... (silence: {state.silence_s:.1f}s)"
    
    if not state.saw_voice:
        state.status_message = "En attente de parole..."
        return state, state.status_message
    
    # Vérifier si le silence est suffisant pour déclencher l'envoi
    if state.silence_s >= float(silence_seconds):
        full = concat_chunks(state)
        
        # Reset l'état VAD
        state.buffer_chunks = []
        state.silence_s = 0.0
        state.saw_voice = False
        
        # Rééchantillonner à la fréquence cible
        full_resampled = resample_linear(full, sr_in=int(state.buffer_sr or sr_in), sr_out=TARGET_SR)
        utt_seconds = float(full_resampled.shape[0]) / float(TARGET_SR) if full_resampled.size else 0.0
        
        if utt_seconds < MIN_UTTERANCE_SECONDS:
            state.status_message = f"Audio trop court ({utt_seconds:.2f}s), ignoré."
            return state, state.status_message
        
        # Stocker l'audio prêt à traiter (en list pour éviter les problèmes de deepcopy)
        set_pending_audio(state, TARGET_SR, full_resampled)
        state.status_message = f"✅ Fin de parole détectée ({utt_seconds:.1f}s). Traitement..."
    
    return state, state.status_message


def process_pending_audio_and_merge(
    voice_state: VoiceStreamState,
    current_code: str,
    modifications_history: str,
):
    """
    Traite l'audio en attente si disponible ET applique le merge.
    Retourne: (code_editor, status, history, voice_state)
    
    Utilise gr.update() pour ne PAS rafraîchir l'éditeur quand il n'y a pas de changement.
    """
    state = voice_state or new_voice_state()
    
    # Récupérer l'audio en attente
    audio = get_pending_audio(state)
    if audio is None:
        # Pas d'audio à traiter - NE PAS mettre à jour l'éditeur
        return gr.update(), gr.update(), gr.update(), state
    
    # Effacer l'audio en attente
    clear_pending_audio(state)
    
    # Appeler le handler existant
    original, llm_code, status, history = process_voice_instruction(
        audio, current_code, modifications_history
    )
    
    state.status_message = status
    
    # Si pas de résultat du LLM, ne pas mettre à jour l'éditeur
    if original is None or llm_code is None:
        return gr.update(), status, history, state
    
    # Appliquer le merge
    merged, had_conflict = three_way_merge(original, llm_code, current_code)
    
    if had_conflict:
        status = status + " (⚠️ conflit résolu automatiquement)"
    elif original != current_code:
        status = status + " (✓ vos modifications conservées)"
    
    return merged, status, history, state


def reset_voice_state(voice_state: VoiceStreamState):
    """Réinitialise l'état de l'enregistrement vocal."""
    new_state = new_voice_state()
    new_state.status_message = "🔄 État audio réinitialisé. En attente..."
    return new_state, new_state.status_message


def create_ui():
    with gr.Blocks(title="Vocal Studio Code", css=CUSTOM_CSS) as demo:
        gr.HTML(f"""
            <div style="text-align: center; padding: 20px;">
                <div style="display: inline-flex; align-items: center; gap: 16px;">
                    <img src="data:image/png;base64,{LOGO_BASE64}" alt="Vocal Studio Code" style="width: 200px; height: 200px;">
                    <h1 style="color: #2180C2; font-size: 4em; font-weight: bold; margin: 0;">
                        Vocal Studio Code
                    </h1>
                </div>
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
                gr.HTML("<h3 style='color: #667eea;'>🎤 Enregistrement Vocal Continu</h3>")
                audio_input = build_audio_component()
                
                with gr.Row():
                    silence_slider = gr.Slider(
                        minimum=0.30, maximum=1.50, value=DEFAULT_SILENCE_SECONDS, step=0.05,
                        label="Silence (s) avant envoi"
                    )
                    rms_slider = gr.Slider(
                        minimum=0.003, maximum=0.060, value=DEFAULT_RMS_THRESHOLD, step=0.001,
                        label="Seuil de détection voix"
                    )

                status_display = gr.Textbox(
                    label="Statut",
                    value="En attente d'instruction...",
                    lines=2,
                    interactive=False,
                    elem_classes=["status-box"],
                )

                with gr.Row():
                    mic_test_btn = gr.Button("Tester micro", variant="secondary", size="sm")
                    reset_audio_btn = gr.Button("Reset audio", variant="secondary", size="sm")
                
                mic_test_btn.click(
                    fn=lambda: "Test micro lancé…",
                    outputs=status_display,
                    js=MIC_TEST_JS,
                )

        gr.HTML("<h3 style='color: #667eea; margin-top: 20px;'>Historique des Modifications</h3>")
        modifications_display = gr.Textbox(
            label="",
            value="",
            lines=6,
            interactive=False,
            elem_classes=["modifications-log"],
            placeholder="Les modifications apparaîtront ici...",
        )

        # States
        voice_state = gr.State(new_voice_state())

        # Streaming audio: traite chaque chunk pour la détection de voix/silence
        audio_input.stream(
            fn=on_audio_stream,
            inputs=[audio_input, voice_state, silence_slider, rms_slider],
            outputs=[voice_state, status_display],
            show_progress="hidden",
            concurrency_limit=32,
        )

        # Timer pour vérifier si un audio est prêt à être traité
        # Utilise gr.update() pour ne PAS rafraîchir l'éditeur quand il n'y a pas de changement
        audio_timer = gr.Timer(0.15)
        audio_timer.tick(
            fn=process_pending_audio_and_merge,
            inputs=[voice_state, code_editor, modifications_display],
            outputs=[code_editor, status_display, modifications_display, voice_state],
            show_progress="hidden",
        )

        # Reset de l'état audio
        reset_audio_btn.click(
            fn=reset_voice_state,
            inputs=[voice_state],
            outputs=[voice_state, status_display],
        )

    return demo
