from typing import Dict, List


class ShortTermMemory:
    """
    Stores recent information from the current interaction.

    This is temporary memory and can be cleared or replaced.
    """

    def __init__(self, max_items: int = 20):
        self.max_items = max_items
        self.items: List[Dict] = []

    def add(self, item: Dict) -> None:
        self.items.append(item)

        if len(self.items) > self.max_items:
            self.items.pop(0)

    def get_all(self) -> List[Dict]:
        return list(self.items)

    def latest(self, count: int = 5) -> List[Dict]:
        return self.items[-count:]

    def clear(self) -> None:
        self.items.clear()

    def count(self) -> int:
        return len(self.items)