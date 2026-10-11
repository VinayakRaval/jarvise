import uuid

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from memory.storage import MemoryStorage


class LongTermMemory:
    """
    Canonical long-term memory.

    The canonical memory record owns the permanent memory_id.

    Other systems such as FAISS must reference this ID instead
    of treating their internal vector position as the identity.
    """

    def __init__(self):
        self.storage = MemoryStorage()

    # =========================================================
    # ID GENERATION
    # =========================================================

    def _generate_memory_id(self) -> str:
        """
        Generate a permanent unique memory ID.

        Example:
            mem_8f4c9e2b1a...
        """

        return f"mem_{uuid.uuid4().hex}"

    # =========================================================
    # STORE
    # =========================================================

    def store(
        self,
        content: str,
        memory_type: str = "general",
        importance: float = 0.5,
        metadata: Optional[
            Dict[str, Any]
        ] = None,
    ) -> Dict[str, Any]:

        content = content.strip()

        if not content:
            raise ValueError(
                "Memory content cannot be empty."
            )

        try:
            importance = float(
                importance
            )
        except (
            TypeError,
            ValueError,
        ):
            importance = 0.5

        importance = max(
            0.0,
            min(
                1.0,
                importance,
            ),
        )

        now = datetime.now(
            timezone.utc
        ).isoformat()

        memory = {
            "memory_id": self._generate_memory_id(),

            "content": content,

            "type": memory_type,

            "importance": importance,

            "metadata": metadata or {},

            "created_at": now,

            "updated_at": None,

            "access_count": 0,

            "last_accessed_at": None,
        }

        return self.storage.add(
            memory
        )

    # =========================================================
    # GET
    # =========================================================

    def get(
        self,
        memory_id: str,
    ) -> Optional[
        Dict[str, Any]
    ]:

        return self.storage.get(
            memory_id
        )

    # =========================================================
    # GET ALL
    # =========================================================

    def get_all(
        self,
    ) -> List[
        Dict[str, Any]
    ]:

        return self.storage.get_all()

    # =========================================================
    # UPDATE
    # =========================================================

    def update(
        self,
        memory_id: str,
        content: Optional[str] = None,
        memory_type: Optional[str] = None,
        importance: Optional[float] = None,
        metadata: Optional[
            Dict[str, Any]
        ] = None,
    ) -> Optional[
        Dict[str, Any]
    ]:

        existing = self.get(
            memory_id
        )

        if existing is None:
            return None

        updates = {}

        if content is not None:

            content = content.strip()

            if not content:
                raise ValueError(
                    "Memory content cannot be empty."
                )

            updates[
                "content"
            ] = content

        if memory_type is not None:

            updates[
                "type"
            ] = memory_type

        if importance is not None:

            try:
                importance = float(
                    importance
                )
            except (
                TypeError,
                ValueError,
            ):
                importance = 0.5

            updates[
                "importance"
            ] = max(
                0.0,
                min(
                    1.0,
                    importance,
                ),
            )

        if metadata is not None:

            updates[
                "metadata"
            ] = metadata

        updates[
            "updated_at"
        ] = datetime.now(
            timezone.utc
        ).isoformat()

        return self.storage.update(
            memory_id=memory_id,
            updates=updates,
        )

    # =========================================================
    # ACCESS TRACKING
    # =========================================================

    def mark_accessed(
        self,
        memory_id: str,
    ) -> Optional[
        Dict[str, Any]
    ]:

        memory = self.get(
            memory_id
        )

        if memory is None:
            return None

        try:
            access_count = int(
                memory.get(
                    "access_count",
                    0,
                )
            )
        except (
            TypeError,
            ValueError,
        ):
            access_count = 0

        return self.storage.update(
            memory_id=memory_id,
            updates={
                "access_count": access_count + 1,
                "last_accessed_at": (
                    datetime.now(
                        timezone.utc
                    ).isoformat()
                ),
            },
        )

    # =========================================================
    # DELETE
    # =========================================================

    def delete(
        self,
        memory_id: str,
    ) -> bool:

        return self.storage.delete(
            memory_id
        )

    # =========================================================
    # CLEAR
    # =========================================================

    def clear(self) -> None:
        self.storage.clear()

    # =========================================================
    # COUNT
    # =========================================================

    def count(self) -> int:
        return self.storage.count()