# datasets/meld_dataset.py

import torch
import pandas as pd
from torch.utils.data import Dataset

EMOTION_MAP = {
    "neutral": 0,
    "joy": 1,
    "sadness": 2,
    "anger": 3,
    "fear": 4,
    "disgust": 5,
    "surprise": 6
}

class MELDFeatureDataset(Dataset):
    def __init__(self, split):
        self.audio = torch.load(f"features/audio/{split}.pt")
        self.visual = torch.load(f"features/visual/{split}.pt")
        self.text = torch.load(f"features/text/{split}.pt")

        csv_path = f"data/MELD/{split}/{split}_sent_emo.csv"
        df = pd.read_csv(csv_path)

        # 🔴 FIX: build keys like diaX_uttY
        self.labels = {}
        for _, row in df.iterrows():
            utt_key = f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}"
            self.labels[utt_key] = EMOTION_MAP[row["Emotion"].lower()]

        # 🔴 FIX: intersect aligned keys
        self.keys = sorted(
            set(self.audio.keys())
            & set(self.visual.keys())
            & set(self.text.keys())
            & set(self.labels.keys())
        )

        print(f"[{split}] Loaded {len(self.keys)} aligned samples")

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        utt_id = self.keys[idx]
        return (
            self.audio[utt_id],
            self.visual[utt_id],
            self.text[utt_id],
            torch.tensor(self.labels[utt_id], dtype=torch.long),
        )
