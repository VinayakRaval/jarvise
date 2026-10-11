from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from memory.vector_store import VectorMemoryStore
from utils.logger import get_logger


class MemoryLifecycle:
    """
    Manages the lifecycle of JARVIS long-term memories.

    Lifecycle states:

        active
            Normal memory that can participate in retrieval.

        superseded
            Replaced by another memory.

        archived
            Kept for historical purposes but normally excluded
            from active reasoning.

    Every memory also contains:

        confidence
        status
        created_at
        updated_at
        access_count
        last_accessed_at
        superseded_by
        archived_at
        lifecycle_updated_at
        lifecycle_reason
    """

    # ---------------------------------------------------------
    # Lifecycle states
    # ---------------------------------------------------------

    ACTIVE = "active"
    SUPERSEDED = "superseded"
    ARCHIVED = "archived"

    VALID_STATUSES = {
        ACTIVE,
        SUPERSEDED,
        ARCHIVED,
    }

    # ---------------------------------------------------------
    # Defaults
    # ---------------------------------------------------------

    DEFAULT_CONFIDENCE = 0.80

    # ---------------------------------------------------------
    # Constructor
    # ---------------------------------------------------------

    def __init__(
        self,
        store: Optional[VectorMemoryStore] = None,
    ):
        self.logger = get_logger(
            "JARVIS.MemoryLifecycle"
        )

        self.store = store or VectorMemoryStore()

        self.logger.info(
            "Memory lifecycle manager initialized"
        )

    # =========================================================
    # Internal helpers
    # =========================================================

    @staticmethod
    def _now() -> str:
        """Return current UTC timestamp."""

        return datetime.now(
            timezone.utc
        ).isoformat()

    # ---------------------------------------------------------
    # Normalize memory
    # ---------------------------------------------------------

    def _normalize_memory(
        self,
        memory: Dict[str, Any],
    ) -> bool:
        """
        Ensure a memory contains the complete lifecycle schema.

        Returns:
            True if something was changed.
            False if the memory was already normalized.
        """

        changed = False

        # -----------------------------------------------------
        # Confidence
        # -----------------------------------------------------

        if "confidence" not in memory:

            memory["confidence"] = (
                self.DEFAULT_CONFIDENCE
            )

            changed = True

        else:

            try:
                confidence = float(
                    memory["confidence"]
                )
            except (
                TypeError,
                ValueError,
            ):
                confidence = (
                    self.DEFAULT_CONFIDENCE
                )

            confidence = max(
                0.0,
                min(
                    1.0,
                    confidence,
                ),
            )

            if memory["confidence"] != confidence:

                memory["confidence"] = confidence

                changed = True

        # -----------------------------------------------------
        # Status
        # -----------------------------------------------------

        if (
            "status" not in memory
            or not memory.get("status")
        ):

            memory["status"] = self.ACTIVE

            changed = True

        elif memory["status"] not in self.VALID_STATUSES:

            memory["status"] = self.ACTIVE

            changed = True

        # -----------------------------------------------------
        # Created timestamp
        # -----------------------------------------------------

        if not memory.get("created_at"):

            memory["created_at"] = self._now()

            changed = True

        # -----------------------------------------------------
        # Updated timestamp
        # -----------------------------------------------------

        if "updated_at" not in memory:

            memory["updated_at"] = None

            changed = True

        # -----------------------------------------------------
        # Access count
        # -----------------------------------------------------

        if "access_count" not in memory:

            memory["access_count"] = 0

            changed = True

        else:

            try:
                access_count = int(
                    memory["access_count"]
                )
            except (
                TypeError,
                ValueError,
            ):
                access_count = 0

            if memory["access_count"] != access_count:

                memory["access_count"] = access_count

                changed = True

        # -----------------------------------------------------
        # Last accessed
        # -----------------------------------------------------

        if "last_accessed_at" not in memory:

            memory["last_accessed_at"] = None

            changed = True

        # -----------------------------------------------------
        # Superseded by
        # -----------------------------------------------------

        if "superseded_by" not in memory:

            memory["superseded_by"] = None

            changed = True

        # -----------------------------------------------------
        # Archived timestamp
        # -----------------------------------------------------

        if "archived_at" not in memory:

            memory["archived_at"] = None

            changed = True

        # -----------------------------------------------------
        # Lifecycle update timestamp
        # -----------------------------------------------------

        if "lifecycle_updated_at" not in memory:

            memory["lifecycle_updated_at"] = None

            changed = True

        # -----------------------------------------------------
        # Lifecycle reason
        # -----------------------------------------------------

        if "lifecycle_reason" not in memory:

            memory["lifecycle_reason"] = None

            changed = True

        return changed

    # =========================================================
    # Migration
    # =========================================================

    def migrate_existing_memories(
        self,
        default_confidence: float = 0.80,
    ) -> Dict[str, Any]:
        """
        Add missing lifecycle fields to all existing memories.

        Stable memory_id values are never changed.
        """

        default_confidence = max(
            0.0,
            min(
                1.0,
                float(default_confidence),
            ),
        )

        changed_count = 0

        with self.store._lock:

            for memory in self.store.memories:

                before = dict(memory)

                if "confidence" not in memory:

                    memory["confidence"] = (
                        default_confidence
                    )

                self._normalize_memory(
                    memory
                )

                if memory != before:

                    changed_count += 1

            if changed_count > 0:

                self.store._save()

        self.logger.info(
            "Memory lifecycle migration completed | "
            f"updated={changed_count} | "
            f"total={len(self.store.memories)}"
        )

        return {
            "success": True,
            "updated": changed_count,
            "total": len(
                self.store.memories
            ),
        }

    # =========================================================
    # Get
    # =========================================================

    def get(
        self,
        memory_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Get a memory using its stable memory_id."""

        if not memory_id:
            return None

        with self.store._lock:

            for memory in self.store.memories:

                if (
                    memory.get("memory_id")
                    == memory_id
                ):

                    changed = (
                        self._normalize_memory(
                            memory
                        )
                    )

                    if changed:

                        self.store._save()

                    return dict(memory)

        return None

    # =========================================================
    # Update
    # =========================================================

    def update(
        self,
        memory_id: str,
        confidence: Optional[float] = None,
        status: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Update lifecycle properties.

        Args:
            memory_id:
                Stable memory identifier.

            confidence:
                New confidence between 0.0 and 1.0.

            status:
                active / superseded / archived.

            reason:
                Human-readable reason for the lifecycle change.
        """

        if not memory_id:
            return None

        with self.store._lock:

            for memory in self.store.memories:

                if (
                    memory.get("memory_id")
                    != memory_id
                ):
                    continue

                self._normalize_memory(
                    memory
                )

                changed = False

                # ---------------------------------------------
                # Confidence
                # ---------------------------------------------

                if confidence is not None:

                    confidence = max(
                        0.0,
                        min(
                            1.0,
                            float(confidence),
                        ),
                    )

                    if (
                        memory.get("confidence")
                        != confidence
                    ):

                        memory[
                            "confidence"
                        ] = confidence

                        changed = True

                # ---------------------------------------------
                # Status
                # ---------------------------------------------

                if status is not None:

                    if (
                        status
                        not in self.VALID_STATUSES
                    ):

                        raise ValueError(
                            "Invalid lifecycle status: "
                            f"{status}"
                        )

                    if (
                        memory.get("status")
                        != status
                    ):

                        memory[
                            "status"
                        ] = status

                        changed = True

                # ---------------------------------------------
                # Reason
                # ---------------------------------------------

                if reason is not None:

                    if (
                        memory.get(
                            "lifecycle_reason"
                        )
                        != reason
                    ):

                        memory[
                            "lifecycle_reason"
                        ] = reason

                        changed = True

                # ---------------------------------------------
                # Archive timestamp
                # ---------------------------------------------

                if (
                    status == self.ARCHIVED
                ):

                    if not memory.get(
                        "archived_at"
                    ):

                        memory[
                            "archived_at"
                        ] = self._now()

                        changed = True

                elif status is not None:

                    if memory.get(
                        "archived_at"
                    ) is not None:

                        memory[
                            "archived_at"
                        ] = None

                        changed = True

                # ---------------------------------------------
                # Lifecycle timestamp
                # ---------------------------------------------

                if changed:

                    now = self._now()

                    memory[
                        "updated_at"
                    ] = now

                    memory[
                        "lifecycle_updated_at"
                    ] = now

                    self.store._save()

                return dict(memory)

        return None

    # =========================================================
    # Confirm
    # =========================================================

    def confirm(
        self,
        memory_id: str,
        confidence: float = 0.95,
        reason: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Confirm a memory as trusted.

        The memory becomes active and receives the supplied
        confidence score.
        """

        return self.update(
            memory_id=memory_id,
            confidence=confidence,
            status=self.ACTIVE,
            reason=reason,
        )

    # =========================================================
    # Reduce confidence
    # =========================================================

    def reduce_confidence(
        self,
        memory_id: str,
        confidence: float,
        reason: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Reduce confidence in a memory.
        """

        return self.update(
            memory_id=memory_id,
            confidence=confidence,
            reason=reason,
        )

    # =========================================================
    # Supersede
    # =========================================================

    def supersede(
        self,
        memory_id: str,
        superseded_by: str,
        reason: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Mark a memory as superseded by another memory.
        """

        if not memory_id:
            return None

        if not superseded_by:
            return None

        with self.store._lock:

            for memory in self.store.memories:

                if (
                    memory.get("memory_id")
                    != memory_id
                ):
                    continue

                self._normalize_memory(
                    memory
                )

                now = self._now()

                memory[
                    "status"
                ] = self.SUPERSEDED

                memory[
                    "superseded_by"
                ] = superseded_by

                if reason is not None:

                    memory[
                        "lifecycle_reason"
                    ] = reason

                memory[
                    "updated_at"
                ] = now

                memory[
                    "lifecycle_updated_at"
                ] = now

                self.store._save()

                return dict(memory)

        return None

    # =========================================================
    # Archive
    # =========================================================

    def archive(
        self,
        memory_id: str,
        reason: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Archive a memory.
        """

        if not memory_id:
            return None

        with self.store._lock:

            for memory in self.store.memories:

                if (
                    memory.get("memory_id")
                    != memory_id
                ):
                    continue

                self._normalize_memory(
                    memory
                )

                now = self._now()

                memory[
                    "status"
                ] = self.ARCHIVED

                memory[
                    "archived_at"
                ] = now

                if reason is not None:

                    memory[
                        "lifecycle_reason"
                    ] = reason

                memory[
                    "updated_at"
                ] = now

                memory[
                    "lifecycle_updated_at"
                ] = now

                self.store._save()

                return dict(memory)

        return None

    # =========================================================
    # Forget
    # =========================================================

    def forget(
        self,
        memory_id: str,
        reason: Optional[str] = None,
    ) -> bool:
        """
        Permanently delete a memory using stable memory_id.
        """

        if not memory_id:
            return False

        result = (
            self.store.delete_by_memory_id(
                memory_id
            )
        )

        if result:

            self.logger.info(
                "Memory permanently forgotten | "
                f"memory_id={memory_id}"
                + (
                    f" | reason={reason}"
                    if reason
                    else ""
                )
            )

        return bool(result)

    # =========================================================
    # Record access
    # =========================================================

    def record_access(
        self,
        memory_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Record that a memory was accessed.
        """

        if not memory_id:
            return None

        with self.store._lock:

            for memory in self.store.memories:

                if (
                    memory.get("memory_id")
                    != memory_id
                ):
                    continue

                self._normalize_memory(
                    memory
                )

                memory[
                    "access_count"
                ] += 1

                memory[
                    "last_accessed_at"
                ] = self._now()

                self.store._save()

                return dict(memory)

        return None

    # =========================================================
    # Active memories
    # =========================================================

    def active_memories(
        self,
    ) -> List[Dict[str, Any]]:
        """
        Return only active memories.
        """

        active = []

        with self.store._lock:

            changed = False

            for memory in self.store.memories:

                if self._normalize_memory(
                    memory
                ):

                    changed = True

                if (
                    memory.get("status")
                    == self.ACTIVE
                ):

                    active.append(
                        dict(memory)
                    )

            if changed:

                self.store._save()

        return active

    # =========================================================
    # Status
    # =========================================================

    def status(self) -> Dict[str, Any]:
        """
        Return lifecycle statistics.
        """

        active = 0
        superseded = 0
        archived = 0

        confidence_total = 0.0

        with self.store._lock:

            changed = False

            for memory in self.store.memories:

                if self._normalize_memory(
                    memory
                ):

                    changed = True

                status = memory.get(
                    "status"
                )

                if status == self.ACTIVE:

                    active += 1

                elif status == self.SUPERSEDED:

                    superseded += 1

                elif status == self.ARCHIVED:

                    archived += 1

                try:

                    confidence_total += float(
                        memory.get(
                            "confidence",
                            self.DEFAULT_CONFIDENCE,
                        )
                    )

                except (
                    TypeError,
                    ValueError,
                ):

                    confidence_total += (
                        self.DEFAULT_CONFIDENCE
                    )

            if changed:

                self.store._save()

        total = (
            active
            + superseded
            + archived
        )

        average_confidence = (
            confidence_total / total
            if total > 0
            else 0.0
        )

        return {
            "total": total,
            "active": active,
            "superseded": superseded,
            "archived": archived,
            "average_confidence": round(
                average_confidence,
                4,
            ),
        }