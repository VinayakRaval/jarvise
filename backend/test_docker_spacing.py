import requests

payload = {
    "model": "qwen3:4b-instruct-2507-q4_K_M",
    "messages": [
        {
            "role": "user",
            "content": "Explain Docker in two short sentences."
        }
    ],
    "stream": False,
    "think": False,
    "options": {
        "temperature": 0.7,
        "num_predict": 256
    }
}

response = requests.post(
    "http://127.0.0.1:11434/api/chat",
    json=payload,
    timeout=180
)
response.raise_for_status()

data = response.json()
answer = data.get("message", {}).get("content", "")

print("RAW:", repr(answer))
print("ANSWER:", answer)
print("WORD COUNT:", len(answer.split()))
