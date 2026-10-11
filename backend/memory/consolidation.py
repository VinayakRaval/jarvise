
import json
import re
from typing import Any, Dict, List, Optional

from ai.model_manager import ModelManager
from memory.vector_store import VectorMemoryStore
from utils.logger import get_logger


class MemoryConsolidator:
    """
    AI-assisted semantic memory consolidation engine.

    Architecture:

        Stored Memories
              ↓
        Semantic similarity
              ↓
        Candidate pairs
              ↓
        Local AI analysis
              ↓
        KEEP / MERGE
              ↓
        Safe consolidation
              ↓
        FAISS verification

    Stable memory_id is permanent identity.

    vector_id is only the current FAISS position.

    Consolidation is DRY-RUN by default.
    """

    def __init__(
        self,
        store: Optional[VectorMemoryStore] = None,
        model_manager: Optional[ModelManager] = None,
        similarity_threshold: float = 0.70,
    ):
        self.logger = get_logger(
            "JARVIS.MemoryConsolidator"
        )

        self.store = (
            store
            if store is not None
            else VectorMemoryStore()
        )

        self.model_manager = (
            model_manager
            if model_manager is not None
            else ModelManager()
        )

        self.similarity_threshold = (
            similarity_threshold
        )

        self.logger.info(
            "AI memory consolidation initialized | "
            f"threshold={similarity_threshold}"
        )

    # ==========================================================
    # FIND SEMANTIC CANDIDATES
    # ==========================================================

    def find_candidates(
        self,
        similarity_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:

        threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else self.similarity_threshold
        )

        memories = list(
            self.store.memories
        )

        candidates = []

        if len(memories) < 2:
            return candidates

        for memory_a in memories:

            content_a = str(
                memory_a.get(
                    "content",
                    "",
                )
            ).strip()

            memory_id_a = memory_a.get(
                "memory_id"
            )

            if not content_a or not memory_id_a:
                continue

            results = self.store.search(
                query=content_a,
                limit=len(memories),
                minimum_score=0.0,
            )

            for result in results:

                memory_id_b = result.get(
                    "memory_id"
                )

                if not memory_id_b:
                    continue

                if memory_id_a == memory_id_b:
                    continue

                # Avoid A-B and B-A duplicates.
                if str(memory_id_a) >= str(
                    memory_id_b
                ):
                    continue

                similarity = float(
                    result.get(
                        "similarity",
                        0.0,
                    )
                )

                if similarity < threshold:
                    continue

                memory_b = (
                    self.store.get_by_memory_id(
                        memory_id_b
                    )
                )

                if memory_b is None:
                    continue

                candidates.append(
                    {
                        "memory_a": dict(
                            memory_a
                        ),
                        "memory_b": dict(
                            memory_b
                        ),
                        "similarity": similarity,
                    }
                )

        candidates.sort(
            key=lambda item: float(
                item["similarity"]
            ),
            reverse=True,
        )

        self.logger.info(
            "Semantic candidates found | "
            f"count={len(candidates)}"
        )

        return candidates

    # ==========================================================
    # AI DECISION
    # ==========================================================

    def analyze_pair(
        self,
        memory_a: Dict[str, Any],
        memory_b: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Ask local Qwen to determine whether two memories
        should be merged.

        The AI does not modify memory.
        It only returns a structured decision.
        """

        content_a = str(
            memory_a.get(
                "content",
                "",
            )
        ).strip()

        content_b = str(
            memory_b.get(
                "content",
                "",
            )
        ).strip()

        instructions = """
You are the memory consolidation system of JARVIS.

Your job is to compare two long-term memories about the same human.

Determine whether they represent:

MERGE
    The memories describe the same underlying fact,
    project, preference, goal, experience, or context.

KEEP_SEPARATE
    The memories contain different facts that should remain
    separate.

Rules:

1. Do not invent information.
2. Do not infer facts that are not present.
3. Do not merge merely because two memories contain similar words.
4. Merge only when they describe the same underlying information.
5. Preserve all useful facts from both memories when merging.
6. Prefer concise factual wording.
7. Never include speculation.
8. Never include explanations outside JSON.

Return ONLY valid JSON.

Format:

{
    "decision": "MERGE",
    "confidence": 0.0,
    "reason": "short explanation",
    "merged_content": "combined factual memory"
}

or:

{
    "decision": "KEEP_SEPARATE",
    "confidence": 0.0,
    "reason": "short explanation",
    "merged_content": ""
}

Confidence must be between 0.0 and 1.0.
"""

        prompt = f"""
MEMORY A:
{content_a}

MEMORY B:
{content_b}
"""

        try:

            raw_result = (
                self.model_manager.generate(
                    instructions=instructions,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                )
            )

            parsed = self._parse_ai_json(
                raw_result
            )

            return self._validate_ai_decision(
                parsed,
                content_a,
                content_b,
            )

        except Exception as error:

            self.logger.exception(
                "AI consolidation analysis failed"
            )

            return {
                "decision": "KEEP_SEPARATE",
                "confidence": 0.0,
                "reason": (
                    f"AI analysis failed: {error}"
                ),
                "merged_content": "",
            }

    # ==========================================================
    # PARSE AI JSON
    # ==========================================================

    def _parse_ai_json(
        self,
        raw_result: str,
    ) -> Dict[str, Any]:

        if not raw_result:
            return {}

        text = raw_result.strip()

        text = re.sub(
            r"^```(?:json)?",
            "",
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(
            r"```$",
            "",
            text,
        )

        text = text.strip()

        try:
            data = json.loads(text)

        except json.JSONDecodeError:

            match = re.search(
                r"\{.*\}",
                text,
                flags=re.DOTALL,
            )

            if not match:
                return {}

            try:
                data = json.loads(
                    match.group(0)
                )
            except json.JSONDecodeError:
                return {}

        if not isinstance(data, dict):
            return {}

        return data

    # ==========================================================
    # VALIDATE AI DECISION
    # ==========================================================

    def _validate_ai_decision(
        self,
        data: Dict[str, Any],
        content_a: str,
        content_b: str,
    ) -> Dict[str, Any]:

        decision = str(
            data.get(
                "decision",
                "KEEP_SEPARATE",
            )
        ).upper().strip()

        if decision not in {
            "MERGE",
            "KEEP_SEPARATE",
        }:
            decision = "KEEP_SEPARATE"

        try:
            confidence = float(
                data.get(
                    "confidence",
                    0.0,
                )
            )
        except (
            TypeError,
            ValueError,
        ):
            confidence = 0.0

        confidence = max(
            0.0,
            min(
                1.0,
                confidence,
            ),
        )

        reason = str(
            data.get(
                "reason",
                "",
            )
        ).strip()

        merged_content = str(
            data.get(
                "merged_content",
                "",
            )
        ).strip()

        if decision == "MERGE":

            if not merged_content:
                decision = "KEEP_SEPARATE"
                confidence = 0.0
                reason = (
                    "AI requested merge but "
                    "did not provide merged content."
                )

            elif len(merged_content) > 1000:
                merged_content = (
                    merged_content[:1000]
                    .rstrip()
                )

        return {
            "decision": decision,
            "confidence": confidence,
            "reason": reason,
            "merged_content": merged_content,
        }

    # ==========================================================
    # INSPECT CANDIDATES WITH AI
    # ==========================================================

    def inspect_with_ai(
        self,
        similarity_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:

        candidates = (
            self.find_candidates(
                similarity_threshold=
                similarity_threshold
            )
        )

        analyses = []

        for candidate in candidates:

            analysis = self.analyze_pair(
                memory_a=candidate[
                    "memory_a"
                ],
                memory_b=candidate[
                    "memory_b"
                ],
            )

            analyses.append(
                {
                    "similarity": candidate[
                        "similarity"
                    ],
                    "memory_a": candidate[
                        "memory_a"
                    ],
                    "memory_b": candidate[
                        "memory_b"
                    ],
                    "analysis": analysis,
                }
            )

        return {
            "success": True,
            "dry_run": True,
            "candidate_count": len(
                candidates
            ),
            "analyses": analyses,
        }

    # ==========================================================
    # CHOOSE SURVIVING MEMORY
    # ==========================================================

    def _choose_survivor(
        self,
        memory_a: Dict[str, Any],
        memory_b: Dict[str, Any],
    ) -> Dict[str, Any]:

        try:
            importance_a = float(
                memory_a.get(
                    "importance",
                    0.5,
                )
            )
        except (
            TypeError,
            ValueError,
        ):
            importance_a = 0.5

        try:
            importance_b = float(
                memory_b.get(
                    "importance",
                    0.5,
                )
            )
        except (
            TypeError,
            ValueError,
        ):
            importance_b = 0.5

        if importance_a > importance_b:
            return memory_a

        if importance_b > importance_a:
            return memory_b

        try:
            access_a = int(
                memory_a.get(
                    "access_count",
                    0,
                )
            )
        except (
            TypeError,
            ValueError,
        ):
            access_a = 0

        try:
            access_b = int(
                memory_b.get(
                    "access_count",
                    0,
                )
            )
        except (
            TypeError,
            ValueError,
        ):
            access_b = 0

        if access_a > access_b:
            return memory_a

        if access_b > access_a:
            return memory_b

        created_a = str(
            memory_a.get(
                "created_at",
                "",
            )
        )

        created_b = str(
            memory_b.get(
                "created_at",
                "",
            )
        )

        if created_a <= created_b:
            return memory_a

        return memory_b

    # ==========================================================
    # APPLY ONE MERGE
    # ==========================================================

    def merge_pair(
        self,
        memory_a: Dict[str, Any],
        memory_b: Dict[str, Any],
        analysis: Dict[str, Any],
        dry_run: bool = True,
    ) -> Dict[str, Any]:

        decision = analysis.get(
            "decision",
            "KEEP_SEPARATE",
        )

        confidence = float(
            analysis.get(
                "confidence",
                0.0,
            )
        )

        if decision != "MERGE":

            return {
                "success": True,
                "merged": False,
                "dry_run": dry_run,
                "decision": decision,
                "confidence": confidence,
                "reason": analysis.get(
                    "reason",
                    "",
                ),
            }

        if confidence < 0.80:

            return {
                "success": True,
                "merged": False,
                "dry_run": dry_run,
                "decision": "KEEP_SEPARATE",
                "confidence": confidence,
                "reason": (
                    "Merge confidence below "
                    "safe threshold of 0.80."
                ),
            }

        survivor = self._choose_survivor(
            memory_a,
            memory_b,
        )

        survivor_id = survivor[
            "memory_id"
        ]

        redundant = (
            memory_b
            if survivor_id
            == memory_a["memory_id"]
            else memory_a
        )

        redundant_id = redundant[
            "memory_id"
        ]

        merged_content = analysis.get(
            "merged_content",
            "",
        )

        if not merged_content:
            return {
                "success": False,
                "merged": False,
                "error": (
                    "No merged content supplied."
                ),
            }

        result = {
            "success": True,
            "merged": True,
            "dry_run": dry_run,
            "decision": decision,
            "confidence": confidence,
            "survivor_id": survivor_id,
            "removed_id": redundant_id,
            "merged_content": merged_content,
        }

        if dry_run:
            return result

        importance = max(
            float(
                memory_a.get(
                    "importance",
                    0.5,
                )
            ),
            float(
                memory_b.get(
                    "importance",
                    0.5,
                )
            ),
        )

        memory_type = survivor.get(
            "type",
            "general",
        )

        updated = (
            self.store.update_by_memory_id(
                memory_id=survivor_id,
                content=merged_content,
                memory_type=memory_type,
                importance=importance,
            )
        )

        if updated is None:
            return {
                "success": False,
                "merged": False,
                "error": (
                    "Failed to update "
                    "surviving memory."
                ),
            }

        deleted = (
            self.store.delete_by_memory_id(
                redundant_id
            )
        )

        if not deleted:
            return {
                "success": False,
                "merged": False,
                "error": (
                    "Memory updated but redundant "
                    "memory could not be deleted."
                ),
                "survivor_id": survivor_id,
                "removed_id": redundant_id,
            }

        synchronized = (
            self.store.index.ntotal
            == len(
                self.store.memories
            )
        )

        return {
            **result,
            "synchronized": synchronized,
            "memory": updated,
        }

    # ==========================================================
    # FULL AI CONSOLIDATION
    # ==========================================================

    def consolidate(
        self,
        dry_run: bool = True,
        similarity_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:

        inspection = (
            self.inspect_with_ai(
                similarity_threshold=
                similarity_threshold
            )
        )

        results = []

        processed_ids = set()

        for item in inspection[
            "analyses"
        ]:

            memory_a = item[
                "memory_a"
            ]

            memory_b = item[
                "memory_b"
            ]

            id_a = memory_a.get(
                "memory_id"
            )

            id_b = memory_b.get(
                "memory_id"
            )

            if not id_a or not id_b:
                continue

            if (
                id_a in processed_ids
                or id_b in processed_ids
            ):
                continue

            result = self.merge_pair(
                memory_a=memory_a,
                memory_b=memory_b,
                analysis=item[
                    "analysis"
                ],
                dry_run=dry_run,
            )

            results.append(
                {
                    "similarity": item[
                        "similarity"
                    ],
                    **result,
                }
            )

            if result.get(
                "merged"
            ):
                processed_ids.add(
                    result.get(
                        "survivor_id"
                    )
                )

                processed_ids.add(
                    result.get(
                        "removed_id"
                    )
                )

        synchronized = (
            self.store.index.ntotal
            == len(
                self.store.memories
            )
        )

        return {
            "success": True,
            "dry_run": dry_run,
            "candidate_count": inspection[
                "candidate_count"
            ],
            "processed_count": len(
                results
            ),
            "synchronized": synchronized,
            "results": results,
        }

    # ==========================================================
    # STATUS
    # ==========================================================

    def status(self) -> Dict[str, Any]:

        return {
            "enabled": True,
            "ai_assisted": True,
            "model": self.model_manager.llm.model,
            "similarity_threshold": (
                self.similarity_threshold
            ),
            "merge_confidence_threshold": 0.80,
            "dry_run_default": True,
            "memory_count": (
                self.store.count()
            ),
            "vector_count": (
                self.store.index.ntotal
            ),
            "synchronized": (
                self.store.index.ntotal
                == len(
                    self.store.memories
                )
            ),
        }
