from ai.llm import repair_response_spacing

tests = [
    "returns.This information is useful.",
    "Compare prices,then choose.",
    "Docker is a tool thatlets apps run.",
    "The result is consistencyacross environments.",
]

for text in tests:
    print("BEFORE:", text)
    print("AFTER: ", repair_response_spacing(text))
    print()
