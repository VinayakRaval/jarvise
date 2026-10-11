from memory.retrieval import MemoryRetriever


def check(condition, message):
    if not condition:
        raise AssertionError(message)

    print(f"[PASS] {message}")


def main():

    print("=" * 60)
    print("JARVIS LIFECYCLE + CONFIDENCE RETRIEVAL TEST")
    print("=" * 60)

    retriever = MemoryRetriever()

    # ---------------------------------------------------------
    # Basic search
    # ---------------------------------------------------------

    print("\nTesting semantic retrieval...")

    results = retriever.search(
        query="JARVIS AI project",
        limit=5,
        minimum_score=0.0,
    )

    check(
        isinstance(results, list),
        "Retrieval returns a list",
    )

    print(
        f"Retrieved memories: {len(results)}"
    )

    # ---------------------------------------------------------
    # Inspect results
    # ---------------------------------------------------------

    for index, memory in enumerate(
        results,
        start=1,
    ):

        print(
            f"\nMemory {index}:"
        )

        print(
            f"  memory_id: "
            f"{memory.get('memory_id')}"
        )

        print(
            f"  content: "
            f"{memory.get('content')}"
        )

        print(
            f"  similarity: "
            f"{memory.get('similarity')}"
        )

        print(
            f"  confidence: "
            f"{memory.get('confidence')}"
        )

        print(
            f"  ranking_score: "
            f"{memory.get('ranking_score')}"
        )

        print(
            f"  status: "
            f"{memory.get('status')}"
        )

        # -----------------------------------------------------
        # Required fields
        # -----------------------------------------------------

        check(
            "memory_id" in memory,
            "Retrieved memory has memory_id",
        )

        check(
            "similarity" in memory,
            "Retrieved memory has similarity",
        )

        check(
            "confidence" in memory,
            "Retrieved memory has confidence",
        )

        check(
            "ranking_score" in memory,
            "Retrieved memory has ranking_score",
        )

        check(
            memory.get("status")
            == "active",
            "Only active memories are retrieved",
        )

        # -----------------------------------------------------
        # Ranking validation
        # -----------------------------------------------------

        similarity = float(
            memory.get(
                "similarity",
                0.0,
            )
        )

        confidence = float(
            memory.get(
                "confidence",
                0.0,
            )
        )

        expected = (
            similarity * 0.70
            +
            confidence * 0.30
        )

        actual = float(
            memory.get(
                "ranking_score",
                0.0,
            )
        )

        check(
            abs(
                expected - actual
            ) < 0.00001,
            "Confidence-aware ranking calculation is correct",
        )

    # ---------------------------------------------------------
    # Ranking order
    # ---------------------------------------------------------

    if len(results) > 1:

        scores = [
            float(
                item.get(
                    "ranking_score",
                    0.0,
                )
            )
            for item in results
        ]

        check(
            scores == sorted(
                scores,
                reverse=True,
            ),
            "Memories are sorted by ranking score",
        )

    # ---------------------------------------------------------
    # Search status
    # ---------------------------------------------------------

    status = retriever.status()

    print(
        "\nVector store status:"
    )

    print(status)

    print("\n" + "=" * 60)
    print(
        "LIFECYCLE + CONFIDENCE RETRIEVAL TEST PASSED"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()