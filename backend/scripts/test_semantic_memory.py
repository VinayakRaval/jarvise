import sys
from pathlib import Path

# Add the backend directory to Python's import path.
BACKEND_DIR = Path(__file__).resolve().parents[1]

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


from memory.retrieval import MemoryRetriever


def main():

    retriever = MemoryRetriever()

    print("=" * 60)
    print("JARVIS SEMANTIC MEMORY TEST")
    print("=" * 60)

    query = input(
        "\nAsk something about your memories: "
    ).strip()

    if not query:
        print("No query entered.")
        return

    results = retriever.search(
        query=query,
        limit=5
    )

    print()

    if not results:
        print("No relevant memories found.")
        return

    print(
        f"Found {len(results)} relevant memories:\n"
    )

    for number, memory in enumerate(
        results,
        start=1
    ):

        print(
            f"{number}. "
            f"{memory.get('content', '')}"
        )

        print(
            f"   Type: "
            f"{memory.get('type', 'general')}"
        )

        print(
            f"   Similarity: "
            f"{memory.get('similarity', 0):.3f}"
        )

        print()


if __name__ == "__main__":
    main()