
import json
import math
import threading
import uuid

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np

from sentence_transformers import SentenceTransformer

from utils.logger import get_logger


class VectorMemoryStore:
    """
    Persistent semantic vector memory store for JARVIS.

    Architecture:

        Memory text
             ↓
        Sentence Transformer
             ↓
        Normalized embedding
             ↓
        FAISS IndexFlatIP
             ↓
        Semantic similarity
             ↓
        Intelligent ranking
             ↓
        Relevant memories


    Ranking:

        60% semantic similarity
        20% importance
        10% recency
         5% memory type
         5% previous usefulness


    Storage:

        data/memories/
            ├── memory.index
            └── vector_memories.json


    IMPORTANT:

    `memory_id` is the permanent identity of a memory.

    `vector_id` is only the current FAISS position.

    vector_id may change after rebuilding or deleting memories.

    memory_id must remain stable.
    """

    # =========================================================
    # INITIALIZATION
    # =========================================================

    def __init__(
        self,
        storage_dir: str = "data/memories",
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self.logger = get_logger(
            "JARVIS.VectorMemory"
        )

        # -----------------------------------------------------
        # Storage
        # -----------------------------------------------------

        self.storage_dir = Path(
            storage_dir
        )

        self.storage_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.index_path = (
            self.storage_dir
            / "memory.index"
        )

        self.data_path = (
            self.storage_dir
            / "vector_memories.json"
        )

        self.model_name = model_name

        # -----------------------------------------------------
        # Thread safety
        # -----------------------------------------------------

        self._lock = threading.RLock()

        # -----------------------------------------------------
        # Embedding model
        # -----------------------------------------------------

        self.logger.info(
            "Loading embedding model | "
            f"{model_name}"
        )

        self.encoder = SentenceTransformer(
            model_name
        )

        self.dimension = int(
            self.encoder.get_embedding_dimension()
        )

        self.logger.info(
            "Embedding model loaded | "
            f"dimension={self.dimension}"
        )

        # -----------------------------------------------------
        # Load metadata
        # -----------------------------------------------------

        self.memories = (
            self._load_memories()
        )

        # -----------------------------------------------------
        # Load FAISS
        # -----------------------------------------------------

        self.index = (
            self._load_index()
        )

        # -----------------------------------------------------
        # Validate synchronization
        # -----------------------------------------------------

        if (
            self.index.ntotal
            != len(self.memories)
        ):
            self.logger.warning(
                "FAISS index and memory metadata "
                "are out of sync. Rebuilding index."
            )

            self._rebuild_index()

        self.logger.info(
            "Vector memory initialized | "
            f"memories={len(self.memories)} | "
            f"vectors={self.index.ntotal}"
        )

    # =========================================================
    # FAISS INDEX
    # =========================================================

    def _create_index(self):
        """
        Create FAISS inner-product index.

        Because embeddings are normalized,
        inner product equals cosine similarity.
        """

        return faiss.IndexFlatIP(
            self.dimension
        )

    def _load_index(self):
        """
        Load existing FAISS index.

        If unavailable or invalid,
        create a new empty index.
        """

        if not self.index_path.exists():
            return self._create_index()

        try:
            index = faiss.read_index(
                str(self.index_path)
            )

            if index.d != self.dimension:

                self.logger.warning(
                    "FAISS index dimension mismatch. "
                    "Creating a new index."
                )

                return self._create_index()

            return index

        except Exception as error:

            self.logger.warning(
                "Could not load FAISS index: "
                f"{error}"
            )

            return self._create_index()

    # =========================================================
    # MEMORY METADATA
    # =========================================================

    def _generate_memory_id(self) -> str:
        """
        Generate a stable memory ID.

        Normally LongTermMemory creates the ID.
        This fallback is useful when loading legacy
        vector records that do not have one.
        """

        return (
            f"mem_{uuid.uuid4().hex}"
        )

    def _load_memories(
        self,
    ) -> List[Dict[str, Any]]:
        """
        Load vector memory metadata.

        Legacy records without memory_id receive
        a generated stable ID.
        """

        if not self.data_path.exists():
            return []

        try:

            with self.data_path.open(
                "r",
                encoding="utf-8",
            ) as file:

                data = json.load(
                    file
                )

            if not isinstance(
                data,
                list,
            ):

                self.logger.warning(
                    "Vector memory file does not "
                    "contain a list."
                )

                return []

            cleaned = []

            used_memory_ids = set()

            for index, memory in enumerate(
                data
            ):

                if not isinstance(
                    memory,
                    dict,
                ):
                    continue

                record = dict(
                    memory
                )

                # -------------------------------------------------
                # Stable memory ID
                # -------------------------------------------------

                memory_id = record.get(
                    "memory_id"
                )

                if not memory_id:

                    memory_id = (
                        self._generate_memory_id()
                    )

                memory_id = str(
                    memory_id
                )

                # Prevent duplicate IDs.
                if memory_id in used_memory_ids:

                    memory_id = (
                        self._generate_memory_id()
                    )

                used_memory_ids.add(
                    memory_id
                )

                record[
                    "memory_id"
                ] = memory_id

                # -------------------------------------------------
                # Vector ID
                #
                # This is NOT permanent.
                # It represents the current position.
                # -------------------------------------------------

                record[
                    "vector_id"
                ] = index

                # -------------------------------------------------
                # Defaults
                # -------------------------------------------------

                record.setdefault(
                    "access_count",
                    0,
                )

                record.setdefault(
                    "last_accessed_at",
                    None,
                )

                record.setdefault(
                    "updated_at",
                    None,
                )

                record.setdefault(
                    "created_at",
                    datetime.now(
                        timezone.utc
                    ).isoformat(),
                )

                record.setdefault(
                    "type",
                    "general",
                )

                record.setdefault(
                    "importance",
                    0.5,
                )

                cleaned.append(
                    record
                )

            return cleaned

        except Exception as error:

            self.logger.warning(
                "Could not load vector memories: "
                f"{error}"
            )

            return []

    # =========================================================
    # SAVE
    # =========================================================

    def _save(self) -> None:
        """
        Persist metadata and FAISS index.
        """

        self.storage_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # -----------------------------------------------------
        # Save metadata
        # -----------------------------------------------------

        temporary_data_path = (
            self.data_path.with_suffix(
                ".tmp"
            )
        )

        with temporary_data_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                self.memories,
                file,
                indent=4,
                ensure_ascii=False,
            )

        temporary_data_path.replace(
            self.data_path
        )

        # -----------------------------------------------------
        # Save FAISS index
        # -----------------------------------------------------

        temporary_index_path = (
            self.index_path.with_suffix(
                ".tmp"
            )
        )

        faiss.write_index(
            self.index,
            str(temporary_index_path),
        )

        temporary_index_path.replace(
            self.index_path
        )

    # =========================================================
    # EMBEDDINGS
    # =========================================================

    def _embed(
        self,
        text: str,
    ) -> np.ndarray:
        """
        Generate normalized embedding.
        """

        if not text or not text.strip():

            raise ValueError(
                "Cannot create embedding "
                "from empty text."
            )

        vector = self.encoder.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        vector = np.asarray(
            vector,
            dtype="float32",
        )

        if vector.ndim != 2:

            raise ValueError(
                "Embedding must be a "
                "2D matrix."
            )

        if vector.shape != (
            1,
            self.dimension,
        ):

            raise ValueError(
                "Unexpected embedding dimension: "
                f"{vector.shape}"
            )

        return vector

    # =========================================================
    # REBUILD INDEX
    # =========================================================

    def _rebuild_index(
        self,
    ) -> None:
        """
        Rebuild the entire FAISS index.

        vector_id values are regenerated based on
        the current vector position.

        memory_id values NEVER change.
        """

        with self._lock:

            new_index = (
                self._create_index()
            )

            valid_memories = []

            for memory in self.memories:

                content = memory.get(
                    "content",
                    "",
                )

                if not isinstance(
                    content,
                    str,
                ):
                    continue

                content = content.strip()

                if not content:
                    continue

                memory_id = memory.get(
                    "memory_id"
                )

                if not memory_id:

                    memory_id = (
                        self._generate_memory_id()
                    )

                    memory[
                        "memory_id"
                    ] = memory_id

                try:

                    vector = self._embed(
                        content
                    )

                    new_index.add(
                        vector
                    )

                    valid_memories.append(
                        memory
                    )

                except Exception as error:

                    self.logger.warning(
                        "Skipping invalid memory "
                        f"during index rebuild: {error}"
                    )

            # -------------------------------------------------
            # Replace memory list
            # -------------------------------------------------

            self.memories = (
                valid_memories
            )

            # -------------------------------------------------
            # Reassign ONLY vector IDs
            # -------------------------------------------------

            for index, memory in enumerate(
                self.memories
            ):

                memory[
                    "vector_id"
                ] = index

            # -------------------------------------------------
            # Replace FAISS index
            # -------------------------------------------------

            self.index = new_index

            # -------------------------------------------------
            # Persist
            # -------------------------------------------------

            self._save()

            self.logger.info(
                "FAISS index rebuilt | "
                f"vectors={self.index.ntotal}"
            )

    # =========================================================
    # ADD MEMORY
    # =========================================================

    def add(
        self,
        memory: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Add a memory to the vector store.

        The memory must contain a stable memory_id.

        LongTermMemory normally creates this ID first.
        """

        if not isinstance(
            memory,
            dict,
        ):

            raise TypeError(
                "Memory must be a dictionary."
            )

        memory_id = memory.get(
            "memory_id"
        )

        if not memory_id:

            raise ValueError(
                "Memory must contain "
                "a stable memory_id."
            )

        memory_id = str(
            memory_id
        )

        content = memory.get(
            "content",
            "",
        )

        if not isinstance(
            content,
            str,
        ):

            raise ValueError(
                "Memory content must be "
                "a string."
            )

        content = content.strip()

        if not content:

            raise ValueError(
                "Memory content cannot "
                "be empty."
            )

        with self._lock:

            # -------------------------------------------------
            # Prevent duplicate stable IDs
            # -------------------------------------------------

            for existing in self.memories:

                if (
                    existing.get(
                        "memory_id"
                    )
                    == memory_id
                ):

                    self.logger.warning(
                        "Memory already exists | "
                        f"memory_id={memory_id}"
                    )

                    return dict(
                        existing
                    )

            # -------------------------------------------------
            # Create embedding
            # -------------------------------------------------

            vector = self._embed(
                content
            )

            # -------------------------------------------------
            # FAISS position
            # -------------------------------------------------

            vector_id = (
                self.index.ntotal
            )

            # -------------------------------------------------
            # Create record
            # -------------------------------------------------

            memory_record = dict(
                memory
            )

            memory_record[
                "memory_id"
            ] = memory_id

            memory_record[
                "vector_id"
            ] = vector_id

            memory_record.setdefault(
                "content",
                content,
            )

            memory_record.setdefault(
                "type",
                "general",
            )

            memory_record.setdefault(
                "importance",
                0.5,
            )

            memory_record.setdefault(
                "created_at",
                datetime.now(
                    timezone.utc
                ).isoformat(),
            )

            memory_record.setdefault(
                "updated_at",
                None,
            )

            memory_record.setdefault(
                "access_count",
                0,
            )

            memory_record.setdefault(
                "last_accessed_at",
                None,
            )

            # -------------------------------------------------
            # Add vector
            # -------------------------------------------------

            self.index.add(
                vector
            )

            # -------------------------------------------------
            # Add metadata
            # -------------------------------------------------

            self.memories.append(
                memory_record
            )

            # -------------------------------------------------
            # Persist
            # -------------------------------------------------

            self._save()

            self.logger.info(
                "Semantic memory stored | "
                f"memory_id={memory_id} | "
                f"vector_id={vector_id}"
            )

            return dict(
                memory_record
            )

    # =========================================================
    # RECENCY
    # =========================================================

    def _recency_score(
        self,
        memory: Dict[str, Any],
    ) -> float:
        """
        Calculate recency score.

        30-day exponential decay.
        """

        timestamp = (
            memory.get(
                "last_accessed_at"
            )
            or memory.get(
                "updated_at"
            )
            or memory.get(
                "created_at"
            )
        )

        if not timestamp:
            return 0.5

        try:

            timestamp = (
                str(timestamp)
                .replace(
                    "Z",
                    "+00:00",
                )
            )

            created = (
                datetime.fromisoformat(
                    timestamp
                )
            )

            if created.tzinfo is None:

                created = (
                    created.replace(
                        tzinfo=timezone.utc
                    )
                )

            now = datetime.now(
                timezone.utc
            )

            age_days = max(
                0.0,
                (
                    now - created
                ).total_seconds()
                / 86400.0,
            )

            return float(
                math.exp(
                    -age_days / 30.0
                )
            )

        except Exception:

            return 0.5

    # =========================================================
    # MEMORY TYPE SCORE
    # =========================================================

    def _type_score(
        self,
        memory: Dict[str, Any],
    ) -> float:
        """
        Give useful memory categories a
        small ranking advantage.
        """

        memory_type = str(
            memory.get(
                "type",
                "general",
            )
        ).lower()

        weights = {
            "project": 1.00,
            "goal": 1.00,
            "important_context": 1.00,
            "preference": 0.95,
            "decision": 0.95,
            "personal_context": 0.90,
            "relationship": 0.85,
            "skill": 0.85,
            "experience": 0.80,
            "routine": 0.75,
            "general": 0.60,
        }

        return float(
            weights.get(
                memory_type,
                0.60,
            )
        )

    # =========================================================
    # IMPORTANCE SCORE
    # =========================================================

    def _importance_score(
        self,
        memory: Dict[str, Any],
    ) -> float:
        """
        Normalize importance to 0..1.
        """

        try:

            importance = float(
                memory.get(
                    "importance",
                    0.5,
                )
            )

        except (
            TypeError,
            ValueError,
        ):

            importance = 0.5

        return max(
            0.0,
            min(
                1.0,
                importance,
            ),
        )

    # =========================================================
    # USAGE SCORE
    # =========================================================

    def _usage_score(
        self,
        memory: Dict[str, Any],
    ) -> float:
        """
        Convert previous memory usage
        into a small ranking signal.
        """

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

        return min(
            1.0,
            max(
                0,
                access_count,
            )
            / 10.0,
        )

    # =========================================================
    # INTELLIGENT RANKING
    # =========================================================

    def _rank_memory(
        self,
        memory: Dict[str, Any],
        similarity: float,
    ) -> float:
        """
        Calculate final ranking score.

        Formula:

            similarity × 0.60
            importance × 0.20
            recency × 0.10
            type × 0.05
            usage × 0.05
        """

        similarity = max(
            0.0,
            min(
                1.0,
                float(similarity),
            ),
        )

        importance = (
            self._importance_score(
                memory
            )
        )

        recency = (
            self._recency_score(
                memory
            )
        )

        type_score = (
            self._type_score(
                memory
            )
        )

        usage_score = (
            self._usage_score(
                memory
            )
        )

        final_score = (
            similarity * 0.60
            + importance * 0.20
            + recency * 0.10
            + type_score * 0.05
            + usage_score * 0.05
        )

        return float(
            final_score
        )

    # =========================================================
    # SEARCH
    # =========================================================

    def search(
        self,
        query: str,
        limit: int = 5,
        minimum_score: float = 0.35,
    ) -> List[
        Dict[str, Any]
    ]:
        """
        Search semantic memories.

        Process:

            Query
              ↓
            Embedding
              ↓
            FAISS candidates
              ↓
            Similarity threshold
              ↓
            Intelligent ranking
              ↓
            Top memories
        """

        if not query or not query.strip():
            return []

        if limit <= 0:
            return []

        with self._lock:

            if self.index.ntotal == 0:
                return []

            if len(self.memories) == 0:
                return []

            # -------------------------------------------------
            # Query embedding
            # -------------------------------------------------

            query_vector = self._embed(
                query
            )

            # -------------------------------------------------
            # Candidate count
            # -------------------------------------------------

            candidate_limit = min(
                max(
                    limit * 3,
                    10,
                ),
                self.index.ntotal,
            )

            # -------------------------------------------------
            # FAISS search
            # -------------------------------------------------

            scores, indices = (
                self.index.search(
                    query_vector,
                    candidate_limit,
                )
            )

            results = []

            # -------------------------------------------------
            # Process candidates
            # -------------------------------------------------

            for score, index in zip(
                scores[0],
                indices[0],
            ):

                if index < 0:
                    continue

                if index >= len(
                    self.memories
                ):
                    continue

                similarity = float(
                    score
                )

                if (
                    similarity
                    < minimum_score
                ):
                    continue

                memory = dict(
                    self.memories[
                        index
                    ]
                )

                memory[
                    "similarity"
                ] = similarity

                memory[
                    "ranking_score"
                ] = self._rank_memory(
                    memory=memory,
                    similarity=similarity,
                )

                results.append(
                    memory
                )

            # -------------------------------------------------
            # Final ranking
            # -------------------------------------------------

            results.sort(
                key=lambda item: float(
                    item.get(
                        "ranking_score",
                        0.0,
                    )
                ),
                reverse=True,
            )

            # -------------------------------------------------
            # Update usage statistics
            # -------------------------------------------------

            now = datetime.now(
                timezone.utc
            ).isoformat()

            changed = False

            for result in results:

                memory_id = result.get(
                    "memory_id"
                )

                if not memory_id:
                    continue

                # Find using stable ID.
                for current in self.memories:

                    if (
                        current.get(
                            "memory_id"
                        )
                        == memory_id
                    ):

                        try:

                            current[
                                "access_count"
                            ] = int(
                                current.get(
                                    "access_count",
                                    0,
                                )
                            ) + 1

                        except (
                            TypeError,
                            ValueError,
                        ):

                            current[
                                "access_count"
                            ] = 1

                        current[
                            "last_accessed_at"
                        ] = now

                        changed = True

                        break

            if changed:
                self._save()

            return results[:limit]

    # =========================================================
    # GET BY VECTOR ID
    # =========================================================

    def get(
        self,
        vector_id: int,
    ) -> Optional[
        Dict[str, Any]
    ]:
        """
        Get memory using its current FAISS position.

        NOTE:

        vector_id is NOT permanent.

        Prefer get_by_memory_id() when possible.
        """

        try:

            vector_id = int(
                vector_id
            )

        except (
            TypeError,
            ValueError,
        ):

            return None

        with self._lock:

            if (
                vector_id < 0
                or vector_id >= len(
                    self.memories
                )
            ):
                return None

            return dict(
                self.memories[
                    vector_id
                ]
            )

    # =========================================================
    # GET BY STABLE MEMORY ID
    # =========================================================

    def get_by_memory_id(
        self,
        memory_id: str,
    ) -> Optional[
        Dict[str, Any]
    ]:
        """
        Get a memory using its permanent memory_id.
        """

        if not memory_id:
            return None

        memory_id = str(
            memory_id
        )

        with self._lock:

            for memory in self.memories:

                if (
                    memory.get(
                        "memory_id"
                    )
                    == memory_id
                ):

                    return dict(
                        memory
                    )

        return None

    # =========================================================
    # FIND VECTOR ID
    # =========================================================

    def _find_vector_id(
        self,
        memory_id: str,
    ) -> Optional[int]:
        """
        Find the current FAISS position
        using permanent memory_id.
        """

        if not memory_id:
            return None

        for index, memory in enumerate(
            self.memories
        ):

            if (
                memory.get(
                    "memory_id"
                )
                == memory_id
            ):
                return index

        return None

    # =========================================================
    # UPDATE BY VECTOR ID
    # =========================================================

    def update(
        self,
        vector_id: int,
        content: str,
        memory_type: str = "general",
        importance: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Backward-compatible update method.

        Prefer update_by_memory_id() for new code.
        """

        try:

            vector_id = int(
                vector_id
            )

        except (
            TypeError,
            ValueError,
        ):

            raise ValueError(
                "Invalid vector_id."
            )

        if vector_id < 0:

            raise ValueError(
                "Invalid vector_id."
            )

        content = (
            content.strip()
            if isinstance(
                content,
                str,
            )
            else ""
        )

        if not content:

            raise ValueError(
                "Memory content cannot "
                "be empty."
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

        memory_type = (
            str(
                memory_type
            ).strip().lower()
            or "general"
        )

        with self._lock:

            if (
                vector_id
                >= len(self.memories)
            ):

                raise ValueError(
                    "Memory does not exist."
                )

            memory = self.memories[
                vector_id
            ]

            memory[
                "content"
            ] = content

            memory[
                "type"
            ] = memory_type

            memory[
                "importance"
            ] = importance

            memory[
                "updated_at"
            ] = datetime.now(
                timezone.utc
            ).isoformat()

            memory.setdefault(
                "access_count",
                0,
            )

            memory.setdefault(
                "last_accessed_at",
                None,
            )

            # Keep current vector position.
            memory[
                "vector_id"
            ] = vector_id

            # -------------------------------------------------
            # Rebuild because the embedding changed.
            # -------------------------------------------------

            self._rebuild_index()

            updated = (
                self.get_by_memory_id(
                    memory.get(
                        "memory_id"
                    )
                )
            )

            self.logger.info(
                "Memory updated | "
                f"memory_id={memory.get('memory_id')} | "
                f"vector_id={vector_id}"
            )

            return (
                updated
                if updated is not None
                else dict(memory)
            )

    # =========================================================
    # UPDATE BY STABLE MEMORY ID
    # =========================================================

    def update_by_memory_id(
        self,
        memory_id: str,
        content: str,
        memory_type: str = "general",
        importance: float = 0.5,
    ) -> Optional[
        Dict[str, Any]
    ]:
        """
        Update memory using permanent memory_id.

        This is the preferred method.

        The current vector_id is looked up dynamically.
        """

        if not memory_id:
            return None

        memory_id = str(
            memory_id
        )

        content = (
            content.strip()
            if isinstance(
                content,
                str,
            )
            else ""
        )

        if not content:

            raise ValueError(
                "Memory content cannot "
                "be empty."
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

        memory_type = (
            str(
                memory_type
            ).strip().lower()
            or "general"
        )

        with self._lock:

            vector_id = (
                self._find_vector_id(
                    memory_id
                )
            )

            if vector_id is None:

                self.logger.warning(
                    "Vector memory not found | "
                    f"memory_id={memory_id}"
                )

                return None

            memory = self.memories[
                vector_id
            ]

            memory[
                "content"
            ] = content

            memory[
                "type"
            ] = memory_type

            memory[
                "importance"
            ] = importance

            memory[
                "updated_at"
            ] = datetime.now(
                timezone.utc
            ).isoformat()

            memory[
                "memory_id"
            ] = memory_id

            memory.setdefault(
                "access_count",
                0,
            )

            memory.setdefault(
                "last_accessed_at",
                None,
            )

            # -------------------------------------------------
            # Rebuild embeddings.
            # -------------------------------------------------

            self._rebuild_index()

            updated = (
                self.get_by_memory_id(
                    memory_id
                )
            )

            self.logger.info(
                "Vector memory updated | "
                f"memory_id={memory_id}"
            )

            return updated

    # =========================================================
    # DELETE BY VECTOR ID
    # =========================================================

    def delete(
        self,
        vector_id: int,
    ) -> bool:
        """
        Backward-compatible delete.

        Prefer delete_by_memory_id().
        """

        try:

            vector_id = int(
                vector_id
            )

        except (
            TypeError,
            ValueError,
        ):

            return False

        with self._lock:

            if (
                vector_id < 0
                or vector_id >= len(
                    self.memories
                )
            ):
                return False

            memory = self.memories[
                vector_id
            ]

            memory_id = memory.get(
                "memory_id"
            )

            self.memories.pop(
                vector_id
            )

            self._rebuild_index()

            self.logger.info(
                "Memory deleted | "
                f"memory_id={memory_id}"
            )

            return True

    # =========================================================
    # DELETE BY STABLE MEMORY ID
    # =========================================================

    def delete_by_memory_id(
        self,
        memory_id: str,
    ) -> bool:
        """
        Delete a vector memory using stable memory_id.
        """

        if not memory_id:
            return False

        memory_id = str(
            memory_id
        )

        with self._lock:

            vector_id = (
                self._find_vector_id(
                    memory_id
                )
            )

            if vector_id is None:

                return False

            self.memories.pop(
                vector_id
            )

            # vector_id values are regenerated,
            # but memory_id values remain unchanged.
            self._rebuild_index()

            self.logger.info(
                "Vector memory deleted | "
                f"memory_id={memory_id}"
            )

            return True

    # =========================================================
    # CLEAR
    # =========================================================

    def clear(
        self,
    ) -> None:
        """
        Delete all vector memories.
        """

        with self._lock:

            self.index = (
                self._create_index()
            )

            self.memories = []

            self._save()

            self.logger.info(
                "Vector memory cleared"
            )

    # =========================================================
    # COUNT
    # =========================================================

    def count(
        self,
    ) -> int:
        """
        Return number of vector memories.
        """

        with self._lock:

            return len(
                self.memories
            )

    # =========================================================
    # STATUS
    # =========================================================

    def status(
        self,
    ) -> Dict[str, Any]:
        """
        Return vector memory system status.
        """

        with self._lock:

            return {
                "embedding_model": (
                    self.model_name
                ),

                "dimensions": (
                    self.dimension
                ),

                "memories": (
                    len(
                        self.memories
                    )
                ),

                "vector_count": (
                    self.index.ntotal
                ),

                "index_type": (
                    "FAISS IndexFlatIP"
                ),

                "normalized_embeddings": True,

                "stable_memory_ids": True,
            }
