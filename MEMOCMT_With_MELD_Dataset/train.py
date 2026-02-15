import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.avt_memocmt import AVT_MemoCMT
from dataset.meld_dataset import MELDFeatureDataset
from metrics import compute_metrics, plot_confusion_matrix

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 7
D_MODEL = 256
BATCH_SIZE = 32
EPOCHS = 30#10
LR = 1e-4


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0

    all_preds, all_labels = [], []

    for audio, visual, text, labels in tqdm(loader):
        audio, visual, text, labels = (
            audio.to(DEVICE),
            visual.to(DEVICE),
            text.to(DEVICE),
            labels.to(DEVICE),
        )

        optimizer.zero_grad()
        logits = model(audio, visual, text)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = logits.argmax(dim=1)
        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())

    metrics = compute_metrics(all_labels, all_preds)
    avg_loss = total_loss / len(loader)

    return avg_loss, metrics


def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for audio, visual, text, labels in loader:
            audio, visual, text, labels = (
                audio.to(DEVICE),
                visual.to(DEVICE),
                text.to(DEVICE),
                labels.to(DEVICE),
            )

            logits = model(audio, visual, text)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    metrics = compute_metrics(all_labels, all_preds)
    avg_loss = total_loss / len(loader)

    return avg_loss, metrics, all_labels, all_preds


def main():
    print("Using device:", DEVICE)

    train_dataset = MELDFeatureDataset("train")
    dev_dataset = MELDFeatureDataset("dev")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE)

    model = AVT_MemoCMT(d_model=D_MODEL, num_classes=NUM_CLASSES).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_f1 = 0.0

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")

        train_loss, train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion
        )

        dev_loss, dev_metrics, dev_labels, dev_preds = evaluate(model, dev_loader)

        plot_confusion_matrix(
            y_true=dev_labels,
            y_pred=dev_preds,
            save_path=f"experiments/confusion_matrix_epoch_{epoch}.png",
            normalize=False
        )

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Acc: {train_metrics['accuracy']:.4f} | "
            f"P: {train_metrics['macro_precision']:.4f} | "
            f"R: {train_metrics['macro_recall']:.4f} | "
            f"F1: {train_metrics['macro_f1']:.4f}"
        )

        print(
            f"Dev   Loss: {dev_loss:.4f} | "
            f"Acc: {dev_metrics['accuracy']:.4f} | "
            f"P: {dev_metrics['macro_precision']:.4f} | "
            f"R: {dev_metrics['macro_recall']:.4f} | "
            f"F1: {dev_metrics['macro_f1']:.4f}"
        )

        if dev_metrics["macro_f1"] > best_f1:
            best_f1 = dev_metrics["macro_f1"]
            torch.save(model.state_dict(), "avt_memocmt_best.pt")
            print("✅ Best model saved")


if __name__ == "__main__":
    main()
