"""
Assistant Vocal de Code
Utilise Gradium.ai pour la reconnaissance vocale, OpenAI GPT pour l'édition de code,
Gradio pour l'interface et LangGraph pour l'architecture.

Point d'entrée principal.
"""

from config import GRADIUM_API_KEY, OPENAI_API_KEY
from ui import create_ui


def main():
    if not GRADIUM_API_KEY:
        print("⚠️  GRADIUM_API_KEY non trouvée dans .env")
    if not OPENAI_API_KEY:
        print("⚠️  OPENAI_API_KEY non trouvée dans .env")

    print("🚀 Démarrage de l'Assistant Vocal de Code...")

    demo = create_ui()
    demo.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7860,
    )


if __name__ == "__main__":
    main()
