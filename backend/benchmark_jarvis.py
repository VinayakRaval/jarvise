import time
from core.jarvis import Jarvis

jarvis = Jarvis(session_id="llm-benchmark")

prompts = [
    "Hello JARVIS, how are you?",
    "Explain Docker in two short sentences.",
    "What is Kubernetes? Answer briefly."
]

for prompt in prompts:
    print("\nUSER:", prompt)

    start = time.perf_counter()

    try:
        response = jarvis.receive(prompt)
        elapsed = time.perf_counter() - start

        print("JARVIS:", response)
        print(f"TOTAL TIME: {elapsed:.2f} seconds")

    except Exception as error:
        print("ERROR:", error)

