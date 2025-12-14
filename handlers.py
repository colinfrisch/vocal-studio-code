"""
Fonctions principales de traitement des instructions.
"""

from graph import CodeAssistantState, code_graph
from transcription import transcribe_audio


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
