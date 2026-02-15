"""
extract_text_features_dataset.py

Dataset-level text feature extraction for MELD

This script:
- Iterates over MELD train/dev/test splits
- Reads human-annotated transcripts from MELD CSV files
- Uses BERT-base to extract text embeddings
- Produces ONE fixed-length embedding per utterance
- Saves features to disk (.pt)

INPUT:
data/MELD/<split>/<split>_sent_emo.csv

OUTPUT:
features/text/{train,dev,test}.pt

NOTE:
- Text is NOT cleaned or modified
- Tokenization is handled by BERT
- Projection to common dimension is done INSIDE the model
"""

import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

# ---------------- Configuration ----------------
MELD_ROOT = "data/MELD"
SPLITS = ["train", "dev", "test"]
OUT_DIR = "features/text"
MAX_LEN = 128
# ------------------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
model.to(device)
model.eval()

def extract_text_embedding(text):
    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS]

    return cls_embedding.squeeze(0).cpu()

def run():
    for split in SPLITS:
        csv_path = os.path.join(MELD_ROOT, split, f"{split}_sent_emo.csv")
        out_path = os.path.join(OUT_DIR, f"{split}.pt")

        if not os.path.exists(csv_path):
            print(f"[WARN] Missing CSV: {csv_path}")
            continue

        df = pd.read_csv(csv_path)

        assert "Utterance_ID" in df.columns
        assert "Utterance" in df.columns

        features = {}

        print(f"[{split}] Extracting text features ({len(df)} utterances)")

        for _, row in tqdm(df.iterrows(), total=len(df)):
            dialogue_id = row["Dialogue_ID"]
            utterance_id = row["Utterance_ID"]
            utt_id = f"dia{dialogue_id}_utt{utterance_id}"

            text = row["Utterance"]

            if not isinstance(text, str) or text.strip() == "":
                continue

            try:
                features[utt_id] = extract_text_embedding(text)
            except Exception as e:
                print(f"[SKIP] {utt_id}: {e}")

        torch.save(features, out_path)
        print(f"[{split}] Saved text features to {out_path}")

if __name__ == "__main__":
    run()
