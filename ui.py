"""
Interface Gradio pour l'assistant vocal de code.
"""

import gradio as gr

from handlers import process_voice_instruction, apply_text_instruction, clear_all


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


def create_ui():
    with gr.Blocks(title="Assistant Vocal de Code", css=CUSTOM_CSS) as demo:
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
                    js=MIC_TEST_JS,
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

    return demo
