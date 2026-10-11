# backend/config/settings.py

import os
from pathlib import Path

from dotenv import load_dotenv


# ============================================================
# PROJECT ROOT
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]


# ============================================================
# ENVIRONMENT
# ============================================================

ENV_FILE = BASE_DIR / ".env"

if ENV_FILE.exists():
    load_dotenv(ENV_FILE)
else:
    load_dotenv()


# ============================================================
# SETTINGS
# ============================================================

class Settings:
    """
    Central configuration for JARVIS.

    All application configuration should be accessed
    through the `settings` object.
    """

    # --------------------------------------------------------
    # Application
    # --------------------------------------------------------

    APP_NAME = os.getenv(
        "APP_NAME",
        "JARVIS"
    )

    JARVIS_NAME = os.getenv(
        "JARVIS_NAME",
        "JARVIS"
    )

    ENVIRONMENT = os.getenv(
        "ENVIRONMENT",
        "development"
    )


    # --------------------------------------------------------
    # Server
    # --------------------------------------------------------

    HOST = os.getenv(
        "HOST",
        "127.0.0.1"
    )

    PORT = int(
        os.getenv(
            "PORT",
            "8000"
        )
    )

    DEBUG = os.getenv(
        "DEBUG",
        "true"
    ).lower() == "true"


    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    LOG_LEVEL = os.getenv(
        "LOG_LEVEL",
        "INFO"
    )


    # --------------------------------------------------------
    # AI
    # --------------------------------------------------------

    AI_MODEL = os.getenv(
        "AI_MODEL",
        "local"
    )

    AI_TEMPERATURE = float(
        os.getenv(
            "AI_TEMPERATURE",
            "0.7"
        )
    )

    OPENAI_API_KEY = os.getenv(
        "OPENAI_API_KEY"
    )


    # --------------------------------------------------------
    # Memory
    # --------------------------------------------------------

    MEMORY_ENABLED = os.getenv(
        "MEMORY_ENABLED",
        "true"
    ).lower() == "true"


    # --------------------------------------------------------
    # Voice
    # --------------------------------------------------------

    VOICE_ENABLED = os.getenv(
        "VOICE_ENABLED",
        "true"
    ).lower() == "true"


    # --------------------------------------------------------
    # Vision
    # --------------------------------------------------------

    VISION_ENABLED = os.getenv(
        "VISION_ENABLED",
        "true"
    ).lower() == "true"


    # --------------------------------------------------------
    # Proactive Interaction
    # --------------------------------------------------------

    PROACTIVE_ENABLED = os.getenv(
        "PROACTIVE_ENABLED",
        "true"
    ).lower() == "true"


# ============================================================
# GLOBAL SETTINGS INSTANCE
# ============================================================

settings = Settings()