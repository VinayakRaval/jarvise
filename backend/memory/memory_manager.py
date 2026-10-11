from typing import Any, Dict, List, Optional

from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory
from memory.retrieval import MemoryRetriever
from memory.router import SemanticMemoryRouter

from utils.logger import get_logger


class MemoryManager:

    def __init__(self):

        self.logger = get_logger(
            "JARVIS.Memory"
        )

        self.short_term = ShortTermMemory()

        self.long_term = LongTermMemory()

        self.retriever = MemoryRetriever()

        self.router = SemanticMemoryRouter(
            retriever=self.retriever,
            threshold=0.42,
        )

        self.logger.info(
            "Memory Manager initialized"
        )

    # =========================================================
    # REMEMBER
    # =========================================================

    def remember(
        self,
        content: str,
        memory_type: str = "general",
        importance: float = 0.5,
        metadata: Optional[
            Dict[str, Any]
        ] = None,
    ) -> Dict[str, Any]:

        if not content or not content.strip():

            return {
                "stored": False,
                "merged": False,
                "reason": "Empty memory",
            }

        content = content.strip()

        # -----------------------------------------------------
        # FIND SIMILAR EXISTING MEMORY
        # -----------------------------------------------------

        similar = self.retriever.search(
            query=content,
            limit=1,
            minimum_score=0.82,
        )

        if similar:

            existing = similar[0]

            similarity = float(
                existing.get(
                    "similarity",
                    0.0,
                )
            )

            # Very high similarity means this is
            # essentially the same information.
            if similarity >= 0.94:

                self.logger.info(
                    "Memory already exists | "
                    f"similarity={similarity:.3f}"
                )

                return {
                    "stored": False,
                    "merged": False,
                    "duplicate": True,
                    "reason": (
                        "Equivalent memory already exists"
                    ),
                    "memory": existing,
                }

            # -------------------------------------------------
            # RELATED MEMORY
            # -------------------------------------------------

            if similarity >= 0.82:

                merged = self._merge_memory(
                    existing=existing,
                    new_content=content,
                    new_type=memory_type,
                    new_importance=importance,
                )

                if merged:

                    self.logger.info(
                        "Existing memory consolidated"
                    )

                    return {
                        "stored": False,
                        "merged": True,
                        "duplicate": False,
                        "memory": merged,
                    }

        # -----------------------------------------------------
        # COMPLETELY NEW MEMORY
        # -----------------------------------------------------

        memory = self.long_term.store(
            content=content,
            memory_type=memory_type,
            importance=importance,
            metadata=metadata,
        )

        stored = self.retriever.add(
            memory
        )

        self.logger.info(
            "New semantic memory stored"
        )

        return {
            "stored": True,
            "merged": False,
            "duplicate": False,
            "memory": stored,
        }

    # =========================================================
    # MEMORY CONSOLIDATION
    # =========================================================

    def _merge_memory(
        self,
        existing: Dict[str, Any],
        new_content: str,
        new_type: str,
        new_importance: float,
    ) -> Optional[
        Dict[str, Any]
    ]:

        existing_content = existing.get(
            "content",
            "",
        )

        if not existing_content:
            return None

        # -----------------------------------------------------
        # Do not blindly concatenate information.
        #
        # First create a clean combined factual memory.
        # -----------------------------------------------------

        combined_parts = [
            existing_content.strip(),
            new_content.strip(),
        ]

        unique_parts = []

        for part in combined_parts:

            normalized = (
                part.lower()
                .strip()
            )

            already_present = False

            for existing_part in unique_parts:

                if (
                    normalized
                    == existing_part.lower().strip()
                ):

                    already_present = True
                    break

            if not already_present:

                unique_parts.append(
                    part
                )

        if len(unique_parts) == 1:

            merged_content = (
                unique_parts[0]
            )

        else:

            merged_content = (
                " ".join(
                    unique_parts
                )
            )

        # -----------------------------------------------------
        # Keep the strongest importance value.
        # -----------------------------------------------------

        try:

            old_importance = float(
                existing.get(
                    "importance",
                    0.5,
                )
            )

        except (
            TypeError,
            ValueError,
        ):

            old_importance = 0.5

        try:

            new_importance = float(
                new_importance
            )

        except (
            TypeError,
            ValueError,
        ):

            new_importance = 0.5

        merged_importance = max(
            old_importance,
            new_importance,
        )

        # -----------------------------------------------------
        # Preserve the most useful type.
        # -----------------------------------------------------

        old_type = existing.get(
            "type",
            "general",
        )

        if (
            old_type == "general"
            and new_type != "general"
        ):

            merged_type = new_type

        else:

            merged_type = old_type

        # -----------------------------------------------------
        # Update existing vector memory.
        # -----------------------------------------------------

        updated = (
            self.retriever.update_memory(
                vector_id=existing.get(
                    "vector_id"
                ),
                content=merged_content,
                memory_type=merged_type,
                importance=merged_importance,
            )
        )

        return updated

    # =========================================================
    # SHORT TERM
    # =========================================================

    def add_recent(
        self,
        item: Dict[str, Any],
    ) -> None:

        self.short_term.add(
            item
        )

    def recent(
        self,
        count: int = 5,
    ) -> List[Dict[str, Any]]:

        return self.short_term.latest(
            count
        )

    # =========================================================
    # LONG TERM
    # =========================================================

    def all_long_term(
        self,
    ) -> List[Dict[str, Any]]:

        return self.long_term.get_all()

    # =========================================================
    # SEARCH
    # =========================================================

    def search(
        self,
        query: str,
        limit: int = 5,
    ) -> List[
        Dict[str, Any]
    ]:

        return self.retriever.search(
            query=query,
            limit=limit,
        )

    # =========================================================
    # SEMANTIC ROUTER
    # =========================================================

    def should_retrieve_memory(
        self,
        query: str,
    ) -> bool:

        return self.router.should_retrieve(
            query
        )

    def inspect_memory_relevance(
        self,
        query: str,
    ) -> Dict[str, Any]:

        return self.router.inspect(
            query
        )

    # =========================================================
    # CLEAR
    # =========================================================

    def clear_short_term(self):

        self.short_term.clear()

    def clear_all(self):

        self.short_term.clear()

        self.long_term.clear()

        self.retriever.clear()

    # =========================================================
    # STATUS
    # =========================================================

    def status(
        self,
    ) -> Dict[str, Any]:

        return {

            "short_term_items":
                self.short_term.count(),

            "long_term_items":
                self.long_term.count(),

            "semantic_memory":
                self.retriever.status(),

            "memory_router": {

                "type":
                    "semantic",

                "threshold":
                    self.router.threshold,
            },
        }