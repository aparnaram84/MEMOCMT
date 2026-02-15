
"""
text_preprocess.py (Validator-only)

This script:
- Validates MELD text annotations
- Checks CSV presence for train/dev/test
- Verifies required columns exist
- Reports basic statistics
- DOES NOT modify or clean text

This is OPTIONAL and does not affect model training.
"""

import os
import pandas as pd

MELD_ROOT = "data/MELD"
SPLITS = ["train", "dev", "test"]
REQUIRED_COLUMNS = ["Utterance_ID", "Utterance"]

def run():
    print("Running MELD text validation...")

    for split in SPLITS:
        csv_path = os.path.join(MELD_ROOT, split, f"{split}_sent_emo.csv")

        if not os.path.exists(csv_path):
            print(f"[ERROR] Missing CSV: {csv_path}")
            continue

        df = pd.read_csv(csv_path)

        missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing_cols:
            print(f"[ERROR] {split}: Missing columns {missing_cols}")
            continue

        total = len(df)
        empty_text = df['Utterance'].isna().sum()
        avg_len = df['Utterance'].astype(str).apply(len).mean()

        print(f"[{split}] OK")
        print(f"  Utterances      : {total}")
        print(f"  Empty utterances: {empty_text}")
        print(f"  Avg text length : {avg_len:.1f} characters")

    print("Text validation completed.")

if __name__ == "__main__":
    run()
