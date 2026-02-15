import torch
import pandas as pd

audio = torch.load("features/audio/train.pt")
visual = torch.load("features/visual/train.pt")
text = torch.load("features/text/train.pt")

df = pd.read_csv("data/MELD/train/train_sent_emo.csv")

print("Audio keys sample:", list(audio.keys())[:5])
print("Visual keys sample:", list(visual.keys())[:5])
print("Text keys sample:", list(text.keys())[:5])
print("CSV keys sample:", df["Utterance_ID"].head().tolist())

print("Audio ∩ Text:", len(set(audio) & set(text)))
print("Audio ∩ Visual:", len(set(audio) & set(visual)))
print("All ∩ CSV:", len(set(audio) & set(visual) & set(text) & set(df["Utterance_ID"])))
