import re

def normalize_spacing(text):
    # Add a space after sentence-ending punctuation when missing.
    text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)

    # Add a space after commas when missing.
    text = re.sub(r',(?=\S)', ', ', text)

    # Do not guess where ordinary words should be separated.
    return text

examples = [
    "Comparing prices helps farmers. This improves decisions.",
    "The best location,maximizing income.",
    "It helps them maximizetheir income.",
    "The result is correct.",
]

for example in examples:
    print("BEFORE:", example)
    print("AFTER: ", normalize_spacing(example))
    print()
