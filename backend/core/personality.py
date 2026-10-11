from pathlib import Path
import yaml

from config.settings import BASE_DIR


class Personality:
    """
    Loads and manages JARVIS personality configuration.
    """

    def __init__(self):
        self.config = self._load_personality()

    def _load_personality(self):

        path = (
            BASE_DIR
            / "backend"
            / "config"
            / "personality.yaml"
        )

        if not path.exists():
            return {}

        with open(path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file) or {}

    def get(self, section: str, default=None):
        return self.config.get(section, default)

    def system_identity(self):
        return self.config.get("identity", {})