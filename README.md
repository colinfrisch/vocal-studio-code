# 🎙️ Assistant Vocal de Code

Un assistant de programmation intelligent qui modifie votre code via des commandes vocales, propulsé par **Gradium.ai**, **OpenAI GPT-4**, **LangGraph** et **Gradio**.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## ✨ Fonctionnalités

- 🎤 **Reconnaissance vocale en temps réel** avec Gradium.ai
- 🤖 **Édition intelligente de code** en deux étapes :
  - 📊 Analyse du code et identification des sections à modifier (avec gestion des dépendances)
  - 🔧 Génération ciblée de code pour chaque section
- 🛡️ **Protection anti-écrasement** : détecte et préserve les modifications utilisateur pendant les appels LLM
- 📐 **Préservation de l'indentation** automatique
- 🎨 **Surlignage des lignes modifiées** dans l'interface
- 📝 **Historique des modifications** avec horodatage
- 🌐 **Interface web moderne** avec Gradio 6.0

## 🏗️ Architecture

```
┌─────────────────┐
│  Commande Vocal │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│           LangGraph Workflow                    │
│  ┌───────────┐  ┌──────────┐  ┌─────────────┐ │
│  │  Parse    │→ │ Analyze  │→ │  Generate   │ │
│  │           │  │ Sections │  │ Replacement │ │
│  └───────────┘  └──────────┘  └──────┬──────┘ │
│                                       │        │
│  ┌─────────────────────────────────────────┐  │
│  │  Apply (avec détection conflits)        │  │
│  └─────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│   Code Modifié  │
└─────────────────┘
```

## 🚀 Installation

### Prérequis

- Python 3.10 ou supérieur
- Clés API :
  - [Gradium.ai](https://gradium.ai) (reconnaissance vocale)
  - [OpenAI](https://platform.openai.com) (GPT-4)

### Étapes

1. **Cloner le repository**
   ```bash
   git clone https://github.com/colinfrisch/voice-hack.git
   cd voice-hack
   ```

2. **Créer un environnement virtuel**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Sur macOS/Linux
   # ou
   venv\Scripts\activate  # Sur Windows
   ```

3. **Installer les dépendances**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configurer les clés API**
   
   Créez un fichier `.env` à la racine du projet :
   ```bash
   GRADIUM_API_KEY=votre_cle_gradium
   OPENAI_API_KEY=votre_cle_openai
   ```

   ⚠️ **IMPORTANT** : Ajoutez `.env` à votre `.gitignore` pour ne pas exposer vos clés !

5. **Lancer l'application**
   ```bash
   python3 app.py
   ```

6. **Ouvrir l'interface**
   
   Accédez à http://127.0.0.1:7860 dans votre navigateur

## 📖 Utilisation

### Mode Vocal

1. Cliquez sur **🎤 Enregistrer** (ou utilisez le raccourci `Ctrl+R`)
2. Parlez votre instruction (ex: "Ajoute une fonction pour calculer la somme de deux nombres")
3. L'application :
   - Transcrit votre commande
   - Analyse le code pour identifier les sections à modifier
   - Génère le nouveau code en préservant les dépendances
   - Applique les modifications intelligemment

### Mode Texte

Vous pouvez aussi taper directement vos instructions dans la zone de texte et cliquer sur **📝 Appliquer**.

### Exemples de commandes

- *"Ajoute une docstring à la fonction main"*
- *"Renomme la variable x en total_count"*
- *"Crée une classe User avec un attribut name"*
- *"Optimise cette boucle for"*
- *"Ajoute la gestion des erreurs avec try-except"*

## 🔧 Composants Principaux

### Fichiers

| Fichier | Description |
|---------|-------------|
| `app.py` | Point d'entrée principal |
| `graph.py` | Architecture LangGraph (analyse → génération → application) |
| `ui.py` | Interface Gradio avec éditeur de code |
| `handlers.py` | Traitement des commandes vocales et texte |
| `transcription.py` | Intégration API Gradium.ai |
| `config.py` | Configuration et variables d'environnement |

### Workflow LangGraph

1. **Parse** : Validation de l'instruction
2. **Analyze** : Identification des sections à modifier (JSON avec numéros de lignes)
3. **Generate** : Génération du code de remplacement pour chaque section
4. **Apply** : Application avec détection de conflits utilisateur

### Gestion des Conflits

Le système détecte automatiquement si l'utilisateur a modifié le code pendant l'appel LLM :

```python
# T0: Code initial envoyé au LLM
# T1: LLM analyse pendant 3s...
# T2: Utilisateur modifie une fonction
# T3: LLM termine → détection du conflit
# T4: Application seulement des sections non-conflictuelles
```

## 🎨 Interface

L'interface Gradio propose :

- **Éditeur de code** : Zone de texte avec coloration syntaxique
- **Zone d'enregistrement audio** : Capture vocale
- **Zone de texte** : Instructions alternatives
- **Statut** : Affichage des opérations en cours
- **Historique** : Liste des modifications avec horodatage
- **Lignes modifiées** : Surlignage visuel avec `→` et `✨`

## 🧪 Tests

Pour tester le workflow LangGraph isolément :

```python
from graph import code_graph

result = code_graph.invoke({
    "code": "def hello():\n    print('Hi')",
    "instruction": "Ajoute une docstring",
    "modifications": [],
    "error": None
})

print(result["code"])
```

## 🔐 Sécurité

- ✅ Ne commitez **JAMAIS** votre fichier `.env`
- ✅ Ajoutez `.env` à `.gitignore`
- ✅ Utilisez des variables d'environnement pour les secrets
- ✅ Révoquez immédiatement toute clé exposée

## 🐛 Dépannage

### Port déjà utilisé

```bash
# L'app utilise automatiquement un port disponible
# Si vous voulez forcer un port spécifique :
demo.launch(server_port=7861)
```

### Erreur de transcription

Vérifiez que votre `GRADIUM_API_KEY` est valide et active.

### Erreur OpenAI

Vérifiez que :
- Votre `OPENAI_API_KEY` est valide
- Vous avez des crédits sur votre compte OpenAI
- Le modèle `gpt-4o` est accessible

## 📊 Dépendances

- **Gradio** : Interface web interactive
- **LangGraph** : Orchestration du workflow
- **LangChain** : Intégration LLM
- **OpenAI** : API GPT-4
- **WebSockets** : Communication temps réel Gradium.ai
- **SoundDevice** : Capture audio
- **python-dotenv** : Gestion des variables d'environnement

## 🤝 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à :

1. Fork le projet
2. Créer une branche (`git checkout -b feature/amelioration`)
3. Commit vos changements (`git commit -m 'Ajout fonctionnalité'`)
4. Push vers la branche (`git push origin feature/amelioration`)
5. Ouvrir une Pull Request
