from memory.vector_store import VectorMemoryStore


def check(condition, message):
    if not condition:
        raise AssertionError(message)

    print(f"[PASS] {message}")


def main():
    print("=" * 60)
    print("JARVIS MEMORY SYNCHRONIZATION TEST")
    print("=" * 60)

    store = VectorMemoryStore()

    print("\nInitial status:")
    print(store.status())

    # ---------------------------------------------------------
    # 1. Check existing memories
    # ---------------------------------------------------------

    memories = list(store.memories)

    check(
        store.index.ntotal == len(memories),
        "FAISS vector count matches metadata count",
    )

    # ---------------------------------------------------------
    # 2. Check stable IDs
    # ---------------------------------------------------------

    memory_ids = [
        memory.get("memory_id")
        for memory in memories
    ]

    check(
        all(memory_ids),
        "Every memory has a memory_id",
    )

    check(
        len(memory_ids) == len(set(memory_ids)),
        "All memory_id values are unique",
    )

    # ---------------------------------------------------------
    # 3. Check vector IDs
    # ---------------------------------------------------------

    vector_ids = [
        memory.get("vector_id")
        for memory in memories
    ]

    expected_vector_ids = list(
        range(len(memories))
    )

    check(
        vector_ids == expected_vector_ids,
        "vector_id values match FAISS positions",
    )

    # ---------------------------------------------------------
    # 4. Save stable IDs
    # ---------------------------------------------------------

    original_ids = list(memory_ids)

    print("\nOriginal stable IDs:")

    for memory in memories:
        print(
            f"  {memory['memory_id']} "
            f"-> vector_id={memory['vector_id']}"
        )

    # ---------------------------------------------------------
    # 5. Rebuild index
    # ---------------------------------------------------------

    print("\nRebuilding FAISS index...")

    store._rebuild_index()

    rebuilt_memories = list(
        store.memories
    )

    rebuilt_ids = [
        memory.get("memory_id")
        for memory in rebuilt_memories
    ]

    # ---------------------------------------------------------
    # 6. Stable ID test
    # ---------------------------------------------------------

    check(
        set(original_ids) == set(rebuilt_ids),
        "memory_id values survive index rebuild",
    )

    # ---------------------------------------------------------
    # 7. Count synchronization
    # ---------------------------------------------------------

    check(
        store.index.ntotal
        == len(store.memories),
        "FAISS and metadata remain synchronized",
    )

    # ---------------------------------------------------------
    # 8. Vector positions
    # ---------------------------------------------------------

    rebuilt_vector_ids = [
        memory.get("vector_id")
        for memory in rebuilt_memories
    ]

    check(
        rebuilt_vector_ids
        == list(range(len(rebuilt_memories))),
        "vector_id values are rebuilt correctly",
    )

    # ---------------------------------------------------------
    # 9. Lookup using stable ID
    # ---------------------------------------------------------

    if rebuilt_memories:

        test_id = rebuilt_memories[0][
            "memory_id"
        ]

        found = store.get_by_memory_id(
            test_id
        )

        check(
            found is not None,
            "Memory can be retrieved using memory_id",
        )

        check(
            found["memory_id"] == test_id,
            "Retrieved memory has correct stable ID",
        )

    print("\n" + "=" * 60)
    print("SYNCHRONIZATION TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()