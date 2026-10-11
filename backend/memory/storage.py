import json

from pathlib import Path
from typing import Any, Dict, List


class MemoryStorage:
    """
    Canonical persistent storage for JARVIS memories.

    This file is the source of truth.

    Storage:
        data/memories/memories.json
    """

    def __init__(
        self,
        file_path: str = "data/memories/memories.json",
    ):
        self.file_path = Path(file_path)

        self.file_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if not self.file_path.exists():
            self._write([])

    # =========================================================
    # READ
    # =========================================================

    def _read(self) -> List[Dict[str, Any]]:
        try:
            with self.file_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                data = json.load(file)

            if isinstance(data, list):
                return data

            return []

        except (
            json.JSONDecodeError,
            FileNotFoundError,
        ):
            return []

    # =========================================================
    # WRITE
    # =========================================================

    def _write(
        self,
        memories: List[Dict[str, Any]],
    ) -> None:

        temporary_path = (
            self.file_path.with_suffix(".tmp")
        )

        with temporary_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                memories,
                file,
                indent=4,
                ensure_ascii=False,
            )

        temporary_path.replace(
            self.file_path
        )

    # =========================================================
    # ADD
    # =========================================================

    def add(
        self,
        memory: Dict[str, Any],
    ) -> Dict[str, Any]:

        memories = self._read()

        memories.append(memory)

        self._write(memories)

        return memory

    # =========================================================
    # UPDATE
    # =========================================================

    def update(
        self,
        memory_id: str,
        updates: Dict[str, Any],
    ) -> Dict[str, Any] | None:

        memories = self._read()

        for memory in memories:

            if memory.get("memory_id") == memory_id:

                memory.update(updates)

                self._write(memories)

                return memory

        return None

    # =========================================================
    # GET BY ID
    # =========================================================

    def get(
        self,
        memory_id: str,
    ) -> Dict[str, Any] | None:

        memories = self._read()

        for memory in memories:

            if memory.get("memory_id") == memory_id:
                return memory

        return None

    # =========================================================
    # GET ALL
    # =========================================================

    def get_all(
        self,
    ) -> List[Dict[str, Any]]:

        return self._read()

    # =========================================================
    # DELETE
    # =========================================================

    def delete(
        self,
        memory_id: str,
    ) -> bool:

        memories = self._read()

        original_count = len(memories)

        memories = [
            memory
            for memory in memories
            if memory.get("memory_id") != memory_id
        ]

        if len(memories) == original_count:
            return False

        self._write(memories)

        return True

    # =========================================================
    # REPLACE ALL
    # =========================================================

    def replace_all(
        self,
        memories: List[Dict[str, Any]],
    ) -> None:

        self._write(memories)

    # =========================================================
    # CLEAR
    # =========================================================

    def clear(self) -> None:
        self._write([])

    # =========================================================
    # COUNT
    # =========================================================

    def count(self) -> int:
        return len(self._read())