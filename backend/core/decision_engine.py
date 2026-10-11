from datetime import datetime, timezone
from copy import deepcopy
from typing import Any, Dict, Optional


class ContextEngine:
    """
    Maintains the current situation/context of JARVIS.

    Later this will receive information from:
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
        """
        Update one context value.
        """
        self._context[key] = value

    def update_many(self, values: Dict[str, Any]) -> None:
        """
        Update multiple context values.
        """
        self._context.update(values)

    def set_activity(self, activity: Optional[str]) -> None:
        """
        Store what the user appears to be doing.
        """
        self._context["activity"] = activity

    def set_vision(self, vision_data: Dict[str, Any]) -> None:
        """
        Store information obtained from the vision system.
        """
        self._context["vision"] = vision_data

    def set_screen(self, screen_data: Dict[str, Any]) -> None:
        """
        Store information obtained from screen analysis.
        """
        self._context["screen"] = screen_data

    def set_audio(self, audio_data: Dict[str, Any]) -> None:
        """
        Store information obtained from the audio system.
        """
        self._context["audio"] = audio_data

    def set_user_state(self, state: Dict[str, Any]) -> None:
        """
        Store temporary information about the user's current state.
        """
        self._context["user_state"] = state

    def snapshot(self) -> Dict[str, Any]:
        """
        Return the current context safely.

        A fresh timestamp is generated whenever JARVIS
        asks for the current context.
        """

        self._context["time"] = datetime.now(
            timezone.utc
        ).isoformat()

        return deepcopy(self._context)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get one context value.
        """
        return self._context.get(key, default)

    def clear(self) -> None:
        """
        Clear temporary context.
        """

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