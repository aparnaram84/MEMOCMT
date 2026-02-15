"""
Text preprocessing for MELD
- Lowercasing
- Basic cleanup
"""

import re

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

if __name__ == "__main__":
    print("Text preprocessing module ready.")
