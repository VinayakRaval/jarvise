import requests

models = [
    "qwen3:4b-instruct-2507-q4_K_M",
    "qwen3:8b",
    "jarvis-qwen-test:latest",
]

prompt = (
    "Write exactly two natural sentences explaining how comparing "
    "mandi prices helps farmers. Use normal spaces between every word. "
    "Do not join words together."
)

for model in models:
    print(f"\n{'=' * 60}\nMODEL: {model}")

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0,
            "num_predict": 160,
        },
    }

    try:
        result = requests.post(
            "http://127.0.0.1:11434/api/chat",
            json=payload,
            timeout=180,
        )
        result.raise_for_status()
        data = result.json()
        answer = data.get("message", {}).get("content", "")

        print("RAW:", repr(answer))
        print("ANSWER:", answer)
    except Exception as error:
        print("ERROR:", repr(error))
