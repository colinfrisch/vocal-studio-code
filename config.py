"""
Configuration et variables d'environnement.
"""

import os
import ssl

import certifi
from dotenv import load_dotenv

load_dotenv()

GRADIUM_API_KEY = os.getenv("GRADIUM_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# SSL context (CA bundle) pour éviter CERTIFICATE_VERIFY_FAILED sur macOS
SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
