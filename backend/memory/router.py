from typing import Any, Dict

from memory.retrieval import MemoryRetriever
from utils.logger import get_logger


class SemanticMemoryRouter:
    """
    Semantic memory router.

    No keyword matching.
    No hardcoded memory phrases.

    The router uses semantic retrieval and the lifecycle-aware
    memory ranking system.
    """

    def __init__(
        self,
        retriever: MemoryRetriever,
        threshold: float = 0.42,
    ):
        self.logger = get_logger(
            "JARVIS.MemoryRouter"
        )

        self.retriever = retriever

        self.threshold = threshold

        self.logger.info(
            "Semantic memory router initialized | "
            f"threshold={threshold}"
        )

    # =========================================================
    # Should retrieve?
    # =========================================================

    def should_retrieve(
        self,
        query: str,
    ) -> bool:

        if not query or not query.strip():

            return False

        results = self.retriever.search(
            query=query,
            limit=1,
            minimum_score=self.threshold,
        )

        if not results:

            self.logger.info(
                "Semantic router: no active relevant memory"
            )

            return False

        result = results[0]

        similarity = float(
            result.get(
                "similarity",
                0.0,
            )
        )

        confidence = float(
            result.get(
                "confidence",
                0.80,
            )
        )

        ranking_score = float(
            result.get(
                "ranking_score",
                similarity,
            )
        )

        status = result.get(
            "status",
            "active",
        )

        # -----------------------------------------------------
        # Only active memories reach this point.
        # -----------------------------------------------------

        if status != "active":

            self.logger.info(
                "Semantic router: memory rejected "
                f"because status={status}"
            )

            return False

        self.logger.info(
            "Semantic router: relevant memory found | "
            f"similarity={similarity:.3f} | "
            f"confidence={confidence:.3f} | "
            f"ranking={ranking_score:.3f}"
        )

        return True

    # =========================================================
    # Inspect
    # =========================================================

    def inspect(
        self,
        query: str,
    ) -> Dict[str, Any]:

        if not query or not query.strip():

            return {
                "relevant": False,
                "similarity": 0.0,
                "confidence": 0.0,
                "ranking_score": 0.0,
                "candidates": [],
            }

        results = self.retriever.search(
            query=query,
            limit=3,
            minimum_score=0.0,
        )

        if not results:

            return {
                "relevant": False,
                "similarity": 0.0,
                "confidence": 0.0,
                "ranking_score": 0.0,
                "candidates": [],
            }

        best = results[0]

        similarity = float(
            best.get(
                "similarity",
                0.0,
            )
        )

        confidence = float(
            best.get(
                "confidence",
                0.80,
            )
        )

        ranking_score = float(
            best.get(
                "ranking_score",
                similarity,
            )
        )

        return {
            "relevant": (
                similarity >= self.threshold
            ),
            "similarity": similarity,
            "confidence": confidence,
            "ranking_score": ranking_score,
            "candidates": results,
        }