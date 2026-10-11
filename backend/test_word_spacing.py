import requests

payload = {
    "model": "qwen3:4b-instruct-2507-q4_K_M",
    "messages": [
        {
            "role": "system",
            "content": (
                "Answer naturally in two short sentences. "
                "Use correct English spelling, grammar, and spaces "
                "between every word. Return only the answer."
            ),
        },
        {
            "role": "user",
            "content": "Explain Docker in two short sentences.",
        },
    ],
    "stream": False,
    "think": False,
    "options": {
        "temperature": 0.2,
        "num_predict": 100,
    },
}

response = requests.post(
    "http://127.0.0.1:11434/api/chat",
    json=payload,
    timeout=180,
)
response.raise_for_status()

answer = response.json().get("message", {}).get("content", "")
print("RAW:", repr(answer))
print("ANSWER:", answer)
