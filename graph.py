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
