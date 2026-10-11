from ai.llm import LLM

llm = LLM()

try:
    result = llm.generate(
        instructions=(
            "You are a helpful assistant. "
            "Write natural English with correct spaces and punctuation."
        ),
        messages=[
            {
                "role": "user",
                "content": (
                    "Explain why comparing market prices helps farmers. "
                    "Answer in two sentences."
                )
            }
        ]
    )

    print("\n--- RAW RESPONSE ---")
    print(repr(result))

    print("\n--- DISPLAYED RESPONSE ---")
    print(result)

finally:
    llm.close()
