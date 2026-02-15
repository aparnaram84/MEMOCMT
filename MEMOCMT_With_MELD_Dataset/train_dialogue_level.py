import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from models.avt_memocmt import AVT_MemoCMT

# ---------------- Device ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Paths ----------------
FEATURE_ROOT = "features"
MELD_ROOT = "data/MELD"

# ---------------- Hyperparams ----------------
BATCH_SIZE = 1          # one dialogue per batch
EPOCHS = 50#10
LR = 1e-4
NUM_CLASSES = 7

EMOTION_MAP = {
    "neutral": 0,
    "joy": 1,
    "anger": 2,
    "sadness": 3,
    "fear": 4,
    "disgust": 5,
    "surprise": 6
}


# ---------------- Dataset ----------------
class MELDDialogueDataset(Dataset):
    def __init__(self, split):
        self.audio = torch.load(f"{FEATURE_ROOT}/audio/{split}.pt")
        self.visual = torch.load(f"{FEATURE_ROOT}/visual/{split}.pt")
        self.text = torch.load(f"{FEATURE_ROOT}/text/{split}.pt")

        df = pd.read_csv(f"{MELD_ROOT}/{split}/{split}_sent_emo.csv")

        self.dialogues = {}

        for _, row in df.iterrows():
            d_id = row["Dialogue_ID"]
            u_id = row["Utterance_ID"]
            utt_key = f"dia{d_id}_utt{u_id}"

            if utt_key not in self.audio:
                continue

            label = EMOTION_MAP[row["Emotion"].lower()]
            self.dialogues.setdefault(d_id, []).append((u_id, utt_key, label))

        # sort utterances temporally
        for d_id in self.dialogues:
            self.dialogues[d_id].sort(key=lambda x: x[0])

        self.dialogue_ids = list(self.dialogues.keys())
        print(f"[{split}] Loaded {len(self.dialogue_ids)} dialogues")

    def __len__(self):
        return len(self.dialogue_ids)

    def __getitem__(self, idx):
        d_id = self.dialogue_ids[idx]
        items = self.dialogues[d_id]

        audio, visual, text, labels = [], [], [], []

        for _, utt_key, lbl in items:
            audio.append(self.audio[utt_key])
            visual.append(self.visual[utt_key])
            text.append(self.text[utt_key])
            labels.append(lbl)

        return (
            torch.stack(audio),                # (T, D)
            torch.stack(visual),               # (T, D)
            torch.stack(text),                 # (T, D)
            torch.tensor(labels, dtype=torch.long)
        )


# ---------------- Training ----------------
def train_epoch(model, loader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0.0
    correct, total = 0, 0

    for step, (audio, visual, text, labels) in enumerate(tqdm(loader)):
        audio = audio.squeeze(0).to(DEVICE)     # (T, D)
        visual = visual.squeeze(0).to(DEVICE)
        text = text.squeeze(0).to(DEVICE)
        labels = labels.squeeze(0).to(DEVICE)

        # -------- DEBUG PRINT (T > 1 CHECK) --------
        if epoch == 0 and step < 3:   # print only first 3 dialogues of epoch 1
            print(f"[DEBUG] Dialogue length T = {audio.shape[0]}")
        # -------------------------------------------

        logits = model(audio, visual, text)     # (T, C)
        loss = criterion(logits, labels)

        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct, total = 0, 0

    with torch.no_grad():
        for audio, visual, text, labels in loader:
            audio = audio.squeeze(0).to(DEVICE)
            visual = visual.squeeze(0).to(DEVICE)
            text = text.squeeze(0).to(DEVICE)
            labels = labels.squeeze(0).to(DEVICE)

            logits = model(audio, visual, text)
            loss = criterion(logits, labels)

            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            total_loss += loss.item()

    return total_loss / len(loader), correct / total


# ---------------- Main ----------------
def main():
    print("Using device:", DEVICE)

    train_ds = MELDDialogueDataset("train")
    dev_ds = MELDDialogueDataset("dev")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = AVT_MemoCMT(d_model=256, num_classes=NUM_CLASSES).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, epoch
        )
        dev_loss, dev_acc = evaluate(model, dev_loader, criterion)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Dev   Loss: {dev_loss:.4f} | Dev Acc: {dev_acc:.4f}")

    torch.save(model.state_dict(), "avt_memocmt_dialogue.pt")
    print("Dialogue-level model saved.")


if __name__ == "__main__":
    main()
