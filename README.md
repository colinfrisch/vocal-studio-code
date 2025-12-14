# 🎤 Voice Code Assistant

Assistant vocal de programmation utilisant Mistral Codestral pour éditer du code en temps réel.

## Fonctionnalités

- 🎙️ Reconnaissance vocale pour les instructions
- 💻 Édition de code en temps réel
- 🤖 Powered by Mistral Codestral
- 🎨 Interface moderne avec Gradio

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Définissez votre clé API Mistral:

```bash
export MISTRAL_API_KEY="votre-clé-api"
```

## Utilisation

```bash
python app.py
```

Puis ouvrez http://localhost:7860 dans votre navigateur.

## Comment ça marche

1. Écrivez ou collez votre code dans l'éditeur
2. Cliquez sur le bouton microphone et donnez vos instructions vocalement
3. Ou tapez vos instructions dans le champ texte
4. L'assistant modifie le code en temps réel selon vos instructions
