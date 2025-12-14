"""
Fonctions principales de traitement des instructions.
"""

import difflib
from graph import CodeAssistantState, code_graph
from transcription import transcribe_audio


def three_way_merge(original: str, llm_result: str, current: str) -> tuple[str, bool]:
    """
    Applique les modifications du LLM sur le code actuel de l'éditeur.
    
    - original: code envoyé au LLM
    - llm_result: code retourné par le LLM  
    - current: code actuel dans l'éditeur (potentiellement modifié par l'utilisateur)
    
    Retourne: (merged_code, had_conflict)
    """
    original = original or ""
    llm_result = llm_result or ""
    current = current or ""
    
    # Si l'utilisateur n'a pas modifié le code, on retourne simplement le résultat LLM
    if original == current:
        return llm_result, False
    
    # Si le LLM n'a pas modifié le code, on garde le code de l'utilisateur
    if original == llm_result:
        return current, False
    
    # Les deux ont modifié : on fait un merge
    original_lines = original.splitlines(keepends=True)
    llm_lines = llm_result.splitlines(keepends=True)
    current_lines = current.splitlines(keepends=True)
    
    # Stratégie : appliquer les changements LLM sur le code utilisateur
    sm = difflib.SequenceMatcher(None, original_lines, llm_lines)
    
    result_lines = []
    current_idx = 0
    had_conflict = False
    
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            # Bloc non modifié par LLM - utiliser la version utilisateur
            result_lines.extend(current_lines[current_idx:current_idx + (i2 - i1)])
            current_idx += (i2 - i1)
        elif tag == 'replace' or tag == 'insert':
            # LLM a modifié/ajouté - vérifier si l'utilisateur a aussi modifié cette zone
            if current_idx < len(current_lines):
                # Vérifier si la zone correspondante dans current est identique à original
                orig_section = original_lines[i1:i2] if tag == 'replace' else []
                curr_section = current_lines[current_idx:current_idx + len(orig_section)] if orig_section else []
                
                if orig_section == curr_section or not orig_section:
                    # L'utilisateur n'a pas modifié cette zone, on applique le changement LLM
                    result_lines.extend(llm_lines[j1:j2])
                    current_idx += len(orig_section)
                else:
                    # Conflit : l'utilisateur a aussi modifié cette zone
                    # On prend les changements LLM mais on marque un conflit
                    result_lines.extend(llm_lines[j1:j2])
                    current_idx += len(orig_section)
                    had_conflict = True
            else:
                result_lines.extend(llm_lines[j1:j2])
        elif tag == 'delete':
            # LLM a supprimé - avancer dans current
            current_idx += (i2 - i1)
    
    # Ajouter le reste du code utilisateur s'il en reste
    if current_idx < len(current_lines):
        result_lines.extend(current_lines[current_idx:])
    
    return "".join(result_lines), had_conflict


def _append_history(history: str, entry: str) -> str:
    history = history or ""
    if not entry:
        return history
    return (history + "\n" + entry).strip() if history else entry


def process_voice_instruction(audio, current_code: str, modifications_history: str):
    """
    Retourne: (original_code, llm_code, status, history)
    - original_code: le code tel qu'envoyé au LLM (pour le merge)
    - llm_code: le code retourné par le LLM
    """
    current_code = current_code or ""
    if audio is None:
        return None, None, "Aucun audio reçu", modifications_history

    instruction = transcribe_audio(audio)
    if not instruction or instruction.startswith("Erreur"):
        return None, None, instruction or "Transcription vide", modifications_history

    initial_state: CodeAssistantState = {
        "code": current_code,
        "instruction": instruction,
        "modifications": [],
        "error": None,
    }

    try:
        result = code_graph.invoke(initial_state)
        if result.get("error"):
            return None, None, f"Erreur: {result['error']}", modifications_history

        new_code = result.get("code", current_code) or current_code
        mods = result.get("modifications", []) or []
        if mods:
            modifications_history = _append_history(modifications_history, mods[-1])

        # Retourne original + llm_result pour le merge ultérieur
        return current_code, new_code, f"✓ {instruction}", modifications_history

    except Exception as e:
        return None, None, f"Erreur: {str(e)}", modifications_history


def apply_text_instruction(instruction: str, current_code: str, modifications_history: str):
    """
    Retourne: (original_code, llm_code, status, history)
    """
    current_code = current_code or ""
    instruction = (instruction or "").strip()
    if not instruction:
        return None, None, "Aucune instruction", modifications_history

    initial_state: CodeAssistantState = {
        "code": current_code,
        "instruction": instruction,
        "modifications": [],
        "error": None,
    }

    try:
        result = code_graph.invoke(initial_state)
        if result.get("error"):
            return None, None, f"Erreur: {result['error']}", modifications_history

        new_code = result.get("code", current_code) or current_code
        mods = result.get("modifications", []) or []
        if mods:
            modifications_history = _append_history(modifications_history, mods[-1])

        return current_code, new_code, f"✓ {instruction}", modifications_history

    except Exception as e:
        return None, None, f"Erreur: {str(e)}", modifications_history


def clear_all():
    return "", "En attente d'instruction...", ""
