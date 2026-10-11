import re
from typing import Any, Optional


def clean_text(text: Optional[str]) -> str:
    """
    Clean user-provided text before processing.

    Removes unnecessary whitespace while preserving
    the natural meaning of the message.
    """

    if text is None:
        return ""

    if not isinstance(text, str):
        text = str(text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def is_empty(value: Any) -> bool:
    """
    Check whether a value is empty.
    """

    if value is None:
        return True

    if isinstance(value, str):
        return not value.strip()

    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0

    return False


def safe_get(
    data: Any,
    key: Any,
    default: Any = None
) -> Any:
    """
    Safely retrieve a value from a dictionary.
    """

    if not isinstance(data, dict):
        return default

    return data.get(key, default)


def truncate_text(
    text: str,
    max_length: int = 500
) -> str:
    """
    Limit text length without failing on short strings.
    """

    if not text:
        return ""

    if len(text) <= max_length:
        return text

    return text[:max_length].rstrip() + "..."