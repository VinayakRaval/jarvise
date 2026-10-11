
from core.jarvis import Jarvis

jarvis = Jarvis(session_id="context-test")

prompts = [
    "I am building a project called Form2Feature for farmers.",
    "It helps farmers compare mandi prices.",
    "What is the name of my project?",
    "Who is the project designed to help?",
    "Why would comparing mandi prices be useful for them?",
]

for prompt in prompts:
    print(f"\nUSER: {prompt}")

    try:
        result = jarvis.receive(prompt)

        if isinstance(result, dict):
            print("JARVIS:", result.get("response", result))
        else:
            print("JARVIS:", result)

    except Exception as error:
        print("ERROR:", error)
        break
