
from memory.consolidation import MemoryConsolidator


def main():
    print("=" * 60)
    print("JARVIS MEMORY SIMILARITY INSPECTION")
    print("=" * 60)

    consolidator = MemoryConsolidator()

    result = consolidator.inspect(
        similarity_threshold=0.0
    )

    candidates = result["candidates"]

    print(
        f"\nTotal relationships found: "
        f"{len(candidates)}"
    )

    for index, candidate in enumerate(
        candidates,
        start=1,
    ):
        memory_a = candidate["memory_a"]
        memory_b = candidate["memory_b"]

        similarity = candidate[
            "similarity"
        ]

        print("\n" + "-" * 60)

        print(
            f"Relationship {index}"
        )

        print(
            f"Similarity: {similarity:.4f}"
        )

        print(
            f"\nA [{memory_a['memory_id']}]"
        )

        print(
            f"  {memory_a['content']}"
        )

        print(
            f"\nB [{memory_b['memory_id']}]"
        )

        print(
            f"  {memory_b['content']}"
        )

    print("\n" + "=" * 60)
    print("INSPECTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
