"""
Select utterances for qualitative explainability analysis.

Criteria:
- Correct predictions with high confidence
- One confident failure case
"""

import json
import pandas as pd
from collections import defaultdict

CONF_THRESHOLD = 0.90
NUM_PER_EMOTION = 1  # keep figures minimal

INFERENCE_JSON = "experiments/attention_dump.json"
CSV_PATH = "data/MELD/dev/dev_sent_emo.csv"

EMOTION_MAP = {
    "neutral": 0,
    "joy": 1,
    "sadness": 2,
    "anger": 3,
    "fear": 4,
    "disgust": 5,
    "surprise": 6
}
INV_EMOTION_MAP = {v: k for k, v in EMOTION_MAP.items()}


def main():
    with open(INFERENCE_JSON, "r") as f:
        results = json.load(f)

    df = pd.read_csv(CSV_PATH)
    gt = {
        f"dia{r.Dialogue_ID}_utt{r.Utterance_ID}": r.Emotion.lower()
        for _, r in df.iterrows()
    }

    correct = defaultdict(list)
    failures = []

    for r in results:
        utt = r["utterance_id"]
        if utt not in gt:
            continue

        pred = r["pred_label"]
        conf = r["confidence"]
        true = gt[utt]

        if pred == true and conf >= CONF_THRESHOLD:
            correct[pred].append((utt, conf))

        if pred != true and conf >= CONF_THRESHOLD:
            failures.append((utt, true, pred, conf))

    print("\n=== SELECTED CORRECT CASES ===")
    selected = []
    for emo, items in correct.items():
        items.sort(key=lambda x: -x[1])
        if items:
            selected.append(items[0])
            print(f"{emo}: {items[0]}")

    print("\n=== SELECTED FAILURE CASE ===")
    if failures:
        failures.sort(key=lambda x: -x[3])
        print(failures[0])

    print("\nUse these utterances for explainability figures.")


if __name__ == "__main__":
    main()
