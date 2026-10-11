import requests
import time

url = "http://127.0.0.1:11434/api/chat"

models = [
    "qwen3:4b-instruct-2507-q4_K_M",
    "qwen3:4b",
    "qwen3:8b",
]

messages = [
    {
        "role": "system",
        "content": (
            "You are a fluent English assistant. "
            "Use correct word spacing and punctuation. "
            "Answer naturally in exactly two sentences."
        ),
    },
    {
        "role": "user",
        "content": (
            "Explain why comparing mandi prices helps farmers "
            "decide where to sell their crops."
        ),
    },
]

for model in models:
    print("\n" + "=" * 65)
    print("MODEL:", model)

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": False,
        "keep_alive": "10m",
        "options": {
            "temperature": 0.2,
            "num_predict": 160,
        },
    }

    start = time.perf_counter()

    try:
        response = requests.post(url, json=payload, timeout=(5, 180))
        response.raise_for_status()

        answer = response.json().get("message", {}).get("content", "")
        print("RAW:", repr(answer))
        print("TEXT:", answer)
        print(f"TIME: {time.perf_counter() - start:.2f}s")

    except requests.RequestException as error:
        print("ERROR:", error)

