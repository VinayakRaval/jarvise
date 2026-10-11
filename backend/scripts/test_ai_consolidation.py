
from memory.consolidation import MemoryConsolidator


def main():
    print("=" * 60)
    print("JARVIS AI MEMORY CONSOLIDATION TEST")
    print("=" * 60)

    consolidator = MemoryConsolidator(
        similarity_threshold=0.70
    )

    print("\nEngine status:")
    print(
        consolidator.status()
    )

    print("\nRunning AI-assisted inspection...")

    result = (
        consolidator.inspect_with_ai()
    )

    print(
        f"\nSemantic candidates: "
        f"{result['candidate_count']}"
    )

    for index, item in enumerate(
        result["analyses"],
        start=1,
    ):

        memory_a = item[
            "memory_a"
        ]

        memory_b = item[
            "memory_b"
        ]

        analysis = item[
            "analysis"
        ]

        print("\n" + "-" * 60)

        print(
            f"Candidate {index}"
        )

        print(
            f"Similarity: "
            f"{item['similarity']:.4f}"
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

        print(
            "\nAI decision:"
        )

        print(
            f"  Decision: "
            f"{analysis['decision']}"
        )

        print(
            f"  Confidence: "
            f"{analysis['confidence']:.2f}"
        )

        print(
            f"  Reason: "
            f"{analysis['reason']}"
        )

        if analysis[
            "merged_content"
        ]:

            print(
                "\n  Proposed merged memory:"
            )

            print(
                f"  {analysis['merged_content']}"
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

    print(
        "\nIMPORTANT:"
    )

    print(
        "No memories were modified."
    )

    print("\n" + "=" * 60)
    print("AI CONSOLIDATION TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
