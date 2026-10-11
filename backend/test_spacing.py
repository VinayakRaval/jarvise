import requests
import time

url = "http://127.0.0.1:11434/api/chat"

payload = {
    "model": "qwen3:4b-instruct-2507-q4_K_M",
    "stream": False,
    "think": False,
    "keep_alive": "10m",
    "messages": [
        {
            "role": "system",
            "content": (
                "Write fluent, correctly spaced English. "
                "Separate every word with normal spaces. "
                "Check spacing before returning your final answer. "
                "Do not join adjacent words."
            )
        },
        {
            "role": "user",
            "content": (
                "Explain why comparing market prices helps farmers "
                "in exactly two sentences."
            )
        }
    ],
    "options": {
        "temperature": 0.2,
        "num_predict": 160
    }
}

start = time.perf_counter()

try:
    response = requests.post(url, json=payload, timeout=(5, 180))
    response.raise_for_status()

    data = response.json()
    answer = data.get("message", {}).get("content", "")

    print("\n--- TEST RESPONSE ---")
    print(repr(answer))
    print("\n--- READABLE RESPONSE ---")
    print(answer)
    print(f"\nTIME: {time.perf_counter() - start:.2f}s")

except requests.RequestException as error:
    print("OLLAMA ERROR:", error)
