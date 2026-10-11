from datetime import datetime, timezone
from copy import deepcopy
from typing import Any, Dict, Optional


class ContextEngine:
    """
    Maintains the current context of JARVIS.

    This will later receive information from:
    - Voice
    - Camera
    - Screen
    - User activity
    - Memory
    - Environment
    """

    def __init__(self):
        self._context: Dict[str, Any] = {
            "time": None,
            "activity": None,
            "location": None,
            "environment": {},
            "vision": {},
            "screen": {},
            "audio": {},
            "user_state": {},
        }

    def update(self, key: str, value: Any) -> None:
        self._context[key] = value

    def update_many(self, values: Dict[str, Any]) -> None:
        self._context.update(values)

    def set_activity(self, activity: Optional[str]) -> None:
        self._context["activity"] = activity

    def set_vision(self, vision_data: Dict[str, Any]) -> None:
        self._context["vision"] = vision_data

    def set_screen(self, screen_data: Dict[str, Any]) -> None:
        self._context["screen"] = screen_data

    def set_audio(self, audio_data: Dict[str, Any]) -> None:
        self._context["audio"] = audio_data

    def set_user_state(self, state: Dict[str, Any]) -> None:
        self._context["user_state"] = state

    def snapshot(self) -> Dict[str, Any]:
        self._context["time"] = datetime.now(
            timezone.utc
        ).isoformat()

        return deepcopy(self._context)

    def get(self, key: str, default: Any = None) -> Any:
        return self._context.get(key, default)

    def clear(self) -> None:
        self._context = {
            "time": None,
            "activity": None,
            "location": None,
            "environment": {},
            "vision": {},
            "screen": {},
            "audio": {},
            "user_state": {},
        }