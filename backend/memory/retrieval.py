from typing import Any, Dict, List, Optional

from memory.vector_store import VectorMemoryStore


class MemoryRetriever:
    """
    Lifecycle-aware and confidence-aware memory retrieval.

    Retrieval pipeline:

        Semantic similarity
              ↓
        Lifecycle filtering
              ↓
        Confidence weighting
              ↓
        Final ranking
    """

    # ---------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------

    DEFAULT_LIMIT = 5

    DEFAULT_MINIMUM_SCORE = 0.35

    # Memories below this confidence are still allowed to
    # participate, but their ranking is reduced.
    LOW_CONFIDENCE_THRESHOLD = 0.50

    # Confidence contribution to final ranking.
    CONFIDENCE_WEIGHT = 0.30

    # Similarity contribution to final ranking.
    SIMILARITY_WEIGHT = 0.70

    # ---------------------------------------------------------
    # Constructor
    # ---------------------------------------------------------

    def __init__(self):

        self.store = VectorMemoryStore()

    # =========================================================
    # Internal helpers
    # =========================================================

    @staticmethod
    def _normalize_confidence(
        value: Any,
    ) -> float:
        """
        Convert confidence to a safe value between 0 and 1.
        """

        try:

            confidence = float(value)

        except (
            TypeError,
            ValueError,
        ):

            confidence = 0.80

        return max(
            0.0,
            min(
                1.0,
                confidence,
            ),
        )

    # ---------------------------------------------------------

    @staticmethod
    def _is_retrievable(
        memory: Dict[str, Any],
    ) -> bool:
        """
        Determine whether a memory is allowed to participate
        in active retrieval.

        Only active memories are retrievable.
        """

        status = memory.get(
            "status",
            "active",
        )

        return status == "active"

    # ---------------------------------------------------------

    def _calculate_ranking_score(
        self,
        similarity: float,
        confidence: float,
    ) -> float:
        """
        Combine semantic similarity and memory confidence.

        Formula:

            ranking =
                similarity * 0.70
                +
                confidence * 0.30
        """

        similarity = max(
            0.0,
            min(
                1.0,
                float(similarity),
            ),
        )

        confidence = self._normalize_confidence(
            confidence
        )

        ranking_score = (
            similarity
            * self.SIMILARITY_WEIGHT
        ) + (
            confidence
            * self.CONFIDENCE_WEIGHT
        )

        return round(
            ranking_score,
            6,
        )

    # =========================================================
    # Search
    # =========================================================

    def search(
        self,
        query: str,
        limit: int = DEFAULT_LIMIT,
        minimum_score: float = DEFAULT_MINIMUM_SCORE,
    ) -> List[Dict[str, Any]]:
        """
        Search long-term memory.

        Steps:

        1. Semantic vector search.
        2. Remove non-active memories.
        3. Apply minimum semantic similarity.
        4. Calculate confidence-aware ranking.
        5. Sort by final ranking score.
        6. Return the best memories.
        """

        if not query or not query.strip():

            return []

        if limit <= 0:

            return []

        # -----------------------------------------------------
        # Request more candidates than final limit.
        #
        # This is important because some candidates may later
        # be removed because of lifecycle state.
        # -----------------------------------------------------

        candidate_limit = max(
            limit * 3,
            limit + 5,
        )

        candidates = self.store.search(
            query=query,
            limit=candidate_limit,
            minimum_score=minimum_score,
        )

        if not candidates:

            return []

        final_results = []

        # -----------------------------------------------------
        # Lifecycle + confidence processing
        # -----------------------------------------------------

        for memory in candidates:

            # ---------------------------------------------
            # Lifecycle filtering
            # ---------------------------------------------

            if not self._is_retrievable(
                memory
            ):
                continue

            # ---------------------------------------------
            # Semantic similarity
            # ---------------------------------------------

            similarity = float(
                memory.get(
                    "similarity",
                    0.0,
                )
            )

            if similarity < minimum_score:

                continue

            # ---------------------------------------------
            # Confidence
            # ---------------------------------------------

            confidence = (
                self._normalize_confidence(
                    memory.get(
                        "confidence",
                        0.80,
                    )
                )
            )

            # ---------------------------------------------
            # Confidence-aware ranking
            # ---------------------------------------------

            ranking_score = (
                self._calculate_ranking_score(
                    similarity=similarity,
                    confidence=confidence,
                )
            )

            # ---------------------------------------------
            # Attach retrieval metadata
            # ---------------------------------------------

            result = dict(memory)

            result[
                "confidence"
            ] = confidence

            result[
                "similarity"
            ] = round(
                similarity,
                6,
            )

            result[
                "ranking_score"
            ] = ranking_score

            result[
                "retrieval_status"
            ] = "active"

            result[
                "confidence_weight"
            ] = self.CONFIDENCE_WEIGHT

            result[
                "similarity_weight"
            ] = self.SIMILARITY_WEIGHT

            final_results.append(
                result
            )

        # -----------------------------------------------------
        # Rank highest first
        # -----------------------------------------------------

        final_results.sort(
            key=lambda item: item.get(
                "ranking_score",
                0.0,
            ),
            reverse=True,
        )

        # -----------------------------------------------------
        # Return requested number
        # -----------------------------------------------------

        return final_results[:limit]

    # =========================================================
    # Add
    # =========================================================

    def add(
        self,
        memory: Dict[str, Any],
    ) -> Dict[str, Any]:

        return self.store.add(
            memory
        )

    # =========================================================
    # Update
    # =========================================================

    def update_memory(
        self,
        vector_id: Optional[int],
        content: str,
        memory_type: str,
        importance: float,
    ) -> Optional[
        Dict[str, Any]
    ]:

        if vector_id is None:

            return None

        return self.store.update(
            vector_id=vector_id,
            content=content,
            memory_type=memory_type,
            importance=importance,
        )

    # =========================================================
    # Stable-ID update
    # =========================================================

    def update_by_memory_id(
        self,
        memory_id: str,
        content: Optional[str] = None,
        memory_type: Optional[str] = None,
        importance: Optional[float] = None,
    ) -> Optional[
        Dict[str, Any]
    ]:

        if not memory_id:

            return None

        if not hasattr(
            self.store,
            "update_by_memory_id",
        ):

            return None

        return self.store.update_by_memory_id(
            memory_id=memory_id,
            content=content,
            memory_type=memory_type,
            importance=importance,
        )

    # =========================================================
    # Get by stable ID
    # =========================================================

    def get_by_memory_id(
        self,
        memory_id: str,
    ) -> Optional[
        Dict[str, Any]
    ]:

        if not memory_id:

            return None

        if not hasattr(
            self.store,
            "get_by_memory_id",
        ):

            return None

        return self.store.get_by_memory_id(
            memory_id
        )

    # =========================================================
    # Delete
    # =========================================================

    def delete_by_memory_id(
        self,
        memory_id: str,
    ) -> bool:

        if not memory_id:

            return False

        if not hasattr(
            self.store,
            "delete_by_memory_id",
        ):

            return False

        return bool(
            self.store.delete_by_memory_id(
                memory_id
            )
        )

    # =========================================================
    # Clear
    # =========================================================

    def clear(self):

        self.store.clear()

    # =========================================================
    # Status
    # =========================================================

    def status(self):

        return self.store.status()