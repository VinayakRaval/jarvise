from pathlib import Path

from memory.long_term import LongTermMemory
from memory.vector_store import VectorMemoryStore


def main():
    print("=" * 60)
    print("JARVIS SEMANTIC MEMORY REBUILD")
    print("=" * 60)

    long_term = LongTermMemory()
    vector_store = VectorMemoryStore()

    memories = long_term.get_all()

    print(f"Existing memories found: {len(memories)}")

    if not memories:
        print("No existing memories to migrate.")
        return

    # Start with a completely fresh vector index.
    vector_store.clear()

    migrated = 0

    for memory in memories:

        content = memory.get(
            "content",
            ""
        ).strip()

        if not content:
            continue

        vector_store.add(memory)

        migrated += 1

        print(
            f"[{migrated}/{len(memories)}] "
            f"{content[:80]}"
        )

    print()
    print("=" * 60)
    print("REBUILD COMPLETE")
    print("=" * 60)

    print(
        f"Memories migrated: {migrated}"
    )

    print(
        f"Vector count: {vector_store.count()}"
    )

    print(
        f"Embedding model: "
        f"{vector_store.model_name}"
    )

    print(
        f"Embedding dimensions: "
        f"{vector_store.dimension}"
    )


if __name__ == "__main__":
    main()