import requests
import time

url = "http://127.0.0.1:11434/api/chat"

for attempt in range(1, 3):
    payload = {
        "model": "qwen3:8b",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are JARVIS. Reply naturally in two short "
                    "sentences with correct English spacing."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Why does comparing mandi prices help farmers?"
                ),
            },
        ],
        "stream": False,
        "think": False,
        "keep_alive": "10m",
        "options": {
            "temperature": 0.2,
            "num_predict": 100,
        },
    }

    start = time.perf_counter()
    response = requests.post(url, json=payload, timeout=(5, 180))
    response.raise_for_status()
    data = response.json()

    print(f"\n--- ATTEMPT {attempt} ---")
    print("RESPONSE:", repr(data.get("message", {}).get("content", "")))
    print(f"TOTAL TIME: {time.perf_counter() - start:.2f}s")

    for key in ("load_duration", "prompt_eval_duration", "eval_duration"):
        value = data.get(key)
        if isinstance(value, (int, float)):
            print(f"{key}: {value / 1e9:.2f}s")

