"""
LangGraph state et nodes pour l'assistant de code.
"""

from datetime import datetime
from typing import TypedDict, List, Optional

from openai import OpenAI
from langgraph.graph import StateGraph, END

from config import OPENAI_API_KEY


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
                    "RÔLE\n"
                    "Tu es un moteur de transformation de code strictement déterministe, sans initiative ni interprétation libre.\n"
                    "Tu reçois un code source (potentiellement vide) et une instruction utilisateur.\n"
                    "Ta seule mission est d'appliquer EXACTEMENT l'instruction au code fourni.\n\n"

                    "====================================\n"
                    "CONTRAT DE SORTIE — STRICT ET OBLIGATOIRE\n"
                    "====================================\n"

                    "1. SORTIE BRUTE UNIQUEMENT\n"
                    "- Retourne UNIQUEMENT le code final.\n"
                    "- Aucun Markdown.\n"
                    "- Aucune explication.\n"
                    "- Aucun texte avant ou après le code.\n"
                    "- Aucun commentaire ajouté hors du code existant.\n\n"

                    "2. EXHAUSTIVITÉ ABSOLUE\n"
                    "- Le code retourné doit être COMPLET.\n"
                    "- Toute partie non modifiée doit être reproduite à l'identique, caractère par caractère.\n"
                    "- Les placeholders, ellipses ou résumés sont STRICTEMENT INTERDITS\n"
                    "  (exemples interdits : `// ...`, `# reste inchangé`, `(inchangé)`).\n\n"

                    "3. RÈGLES DE SUPPRESSION (CRITIQUES)\n"
                    "- Si l'instruction demande explicitement de supprimer TOUT le contenu du fichier :\n"
                    "  → retourne une chaîne vide \"\" (ou {} uniquement si le format est JSON).\n"
                    "- Si l'instruction demande de supprimer une PARTIE précise du code :\n"
                    "  → supprime uniquement cette partie et retourne tout le reste du code intact.\n"
                    "- Ne JAMAIS conserver du code explicitement demandé à être supprimé.\n"
                    "- Ne JAMAIS supprimer du code qui n'est pas explicitement visé.\n\n"

                    "4. RÈGLES DE CRÉATION\n"
                    "- Si le code d'entrée est vide et que l'instruction demande une création :\n"
                    "  → génère l'intégralité du code requis.\n"
                    "- Si le code est vide et que l'instruction ne demande aucune création :\n"
                    "  → retourne une chaîne vide.\n\n"

                    "5. INTERDICTIONS ABSOLUES\n"
                    "- N'ajoute AUCUNE fonctionnalité implicite.\n"
                    "- Ne refactorise PAS.\n"
                    "- Ne renomme PAS.\n"
                    "- Ne reformate PAS.\n"
                    "- Ne corrige PAS sauf si explicitement demandé.\n\n"

                    "6. CAS AMBIGU OU IMPOSSIBLE\n"
                    "- Si l'instruction est ambiguë, contradictoire ou impossible à appliquer sans hypothèses :\n"
                    "  → retourne STRICTEMENT le code original, inchangé.\n\n"

                    "7. RÈGLE D'OR\n"
                    "- Fais EXACTEMENT ce que l'utilisateur demande.\n"
                    "- Rien de plus.\n"
                    "- Rien de moins.\n"
                ),
            },
            {
                "role": "user",
                "content": (
                    "Code actuel :\n"
                    f"{current_code if current_code else ''}\n\n"
                    "Instruction :\n"
                    f"{instruction}\n\n"
                    "Retourne le résultat FINAL COMPLET :"
                ),
            },
        ]


        response = client.chat.completions.create(
            model="gpt-4.1",
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
