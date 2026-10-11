
from memory.lifecycle import MemoryLifecycle


def check(condition, message):
    if not condition:
        raise AssertionError(message)

    print(f"[PASS] {message}")


def main():

    print("=" * 60)
    print("JARVIS MEMORY LIFECYCLE TEST")
    print("=" * 60)

    lifecycle = MemoryLifecycle()

    # ----------------------------------------------------------
    # 1. Existing IDs
    # ----------------------------------------------------------

    original_ids = [
        memory.get("memory_id")
        for memory in lifecycle.store.memories
    ]

    print(
        f"\nExisting memories: {len(original_ids)}"
    )

    check(
        all(original_ids),
        "Every existing memory has a memory_id",
    )

    check(
        len(original_ids)
        == len(set(original_ids)),
        "Existing memory IDs are unique",
    )

    # ----------------------------------------------------------
    # 2. Migrate
    # ----------------------------------------------------------

    print(
        "\nMigrating lifecycle fields..."
    )

    migration = (
        lifecycle.migrate_existing_memories(
            default_confidence=0.80
        )
    )

    print(migration)

    # ----------------------------------------------------------
    # 3. Required fields
    # ----------------------------------------------------------

    required_fields = {
        "memory_id",
        "content",
        "confidence",
        "status",
        "created_at",
        "updated_at",
        "access_count",
        "last_accessed_at",
        "superseded_by",
        "archived_at",
        "lifecycle_updated_at",
    }

    print(
        "\nChecking lifecycle schema..."
    )

    for memory in lifecycle.store.memories:

        memory_id = memory.get(
            "memory_id",
            "<missing>",
        )

        missing = sorted(
            required_fields
            - set(memory.keys())
        )

        print(
            f"\nMemory: {memory_id}"
        )

        if missing:
            print(
                "MISSING FIELDS:"
            )

            for field in missing:
                print(
                    f"  - {field}"
                )

        else:
            print(
                "All lifecycle fields present."
            )

        check(
            not missing,
            (
                f"Lifecycle fields exist for "
                f"{memory_id}"
            ),
        )

        # ------------------------------------------------------
        # Print lifecycle state
        # ------------------------------------------------------

        print(
            "  confidence:",
            memory.get(
                "confidence"
            ),
        )

        print(
            "  status:",
            memory.get(
                "status"
            ),
        )

        print(
            "  created_at:",
            memory.get(
                "created_at"
            ),
        )

        print(
            "  updated_at:",
            memory.get(
                "updated_at"
            ),
        )

        print(
            "  access_count:",
            memory.get(
                "access_count"
            ),
        )

        print(
            "  last_accessed_at:",
            memory.get(
                "last_accessed_at"
            ),
        )

        print(
            "  superseded_by:",
            memory.get(
                "superseded_by"
            ),
        )

        print(
            "  archived_at:",
            memory.get(
                "archived_at"
            ),
        )

        print(
            "  lifecycle_updated_at:",
            memory.get(
                "lifecycle_updated_at"
            ),
        )

    # ----------------------------------------------------------
    # 4. Stable ID preservation
    # ----------------------------------------------------------

    current_ids = [
        memory.get("memory_id")
        for memory in lifecycle.store.memories
    ]

    check(
        set(original_ids)
        == set(current_ids),
        "Stable memory IDs survived lifecycle migration",
    )

    # ----------------------------------------------------------
    # 5. Test confidence
    # ----------------------------------------------------------

    if current_ids:

        test_id = current_ids[0]

        print(
            "\nTesting confidence update..."
        )

        confirmed = lifecycle.confirm(
            memory_id=test_id,
            confidence=0.95,
            reason="Lifecycle test",
        )

        check(
            confirmed is not None,
            "Memory confirmation works",
        )

        check(
            float(
                confirmed["confidence"]
            )
            == 0.95,
            "Confidence can be increased",
        )

        reduced = (
            lifecycle.reduce_confidence(
                memory_id=test_id,
                confidence=0.60,
                reason="Lifecycle test",
            )
        )

        check(
            reduced is not None,
            "Confidence reduction works",
        )

        check(
            float(
                reduced["confidence"]
            )
            == 0.60,
            "Confidence can be reduced",
        )

        # Restore the test memory.
        restored = lifecycle.confirm(
            memory_id=test_id,
            confidence=0.80,
            reason="Lifecycle test completed",
        )

        check(
            restored is not None,
            "Confidence restored",
        )

    # ----------------------------------------------------------
    # 6. FAISS synchronization
    # ----------------------------------------------------------

    check(
        lifecycle.store.index.ntotal
        == len(
            lifecycle.store.memories
        ),
        "FAISS and metadata remain synchronized",
    )

    # ----------------------------------------------------------
    # 7. Final lifecycle status
    # ----------------------------------------------------------

    print(
        "\nLifecycle status:"
    )

    print(
        lifecycle.status()
    )

    # ----------------------------------------------------------
    # 8. Final stable ID check
    # ----------------------------------------------------------

    final_ids = [
        memory.get("memory_id")
        for memory in lifecycle.store.memories
    ]

    check(
        set(original_ids)
        == set(final_ids),
        "Stable memory IDs remain unchanged",
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "MEMORY LIFECYCLE TEST PASSED"
    )

    print(
        "=" * 60
    )


if __name__ == "__main__":
    main()
