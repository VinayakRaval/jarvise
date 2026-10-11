
from memory.consolidation import MemoryConsolidator


def main():
    print("=" * 60)
    print("JARVIS MEMORY CONSOLIDATION TEST")
    print("=" * 60)

    consolidator = MemoryConsolidator()

    print("\nEngine status:")
    print(
        consolidator.status()
    )

    print("\nScanning for consolidation candidates...")

    result = consolidator.inspect()

    print(
        f"\nCandidates found: "
        f"{result['candidate_count']}"
    )

    for index, candidate in enumerate(
        result["candidates"],
        start=1,
    ):
        memory_a = candidate[
            "memory_a"
        ]

        memory_b = candidate[
            "memory_b"
        ]

        print(
            f"\nCandidate {index}"
        )

        print(
            f"Similarity: "
            f"{candidate['similarity']:.4f}"
        )

        print(
            f"A [{memory_a['memory_id']}]: "
            f"{memory_a['content']}"
        )

        print(
            f"B [{memory_b['memory_id']}]: "
            f"{memory_b['content']}"
        )

    print("\n" + "=" * 60)
    print("DRY-RUN CONSOLIDATION")
    print("=" * 60)

    consolidation = (
        consolidator.consolidate(
            dry_run=True
        )
    )

    print(
        f"\nCandidates: "
        f"{consolidation['candidate_count']}"
    )

    print(
        f"Processed: "
        f"{consolidation['processed_count']}"
    )

    print(
        f"Synchronized: "
        f"{consolidation['synchronized']}"
    )

    print("\nNo memories were modified.")

    print("\n" + "=" * 60)
    print("CONSOLIDATION TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
