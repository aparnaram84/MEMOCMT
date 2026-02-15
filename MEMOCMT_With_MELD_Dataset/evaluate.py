import torch
from torch.utils.data import DataLoader
from models.avt_memocmt import AVT_MemoCMT
from dataset.meld_dataset import MELDFeatureDataset
from metrics import compute_metrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 7
D_MODEL = 256
BATCH_SIZE = 32
CHECKPOINT = "avt_memocmt_best.pt"


def main():
    print("Evaluating on DEV set")

    dataset = MELDFeatureDataset("dev")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    model = AVT_MemoCMT(d_model=D_MODEL, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for audio, visual, text, labels in loader:
            audio, visual, text = (
                audio.to(DEVICE),
                visual.to(DEVICE),
                text.to(DEVICE),
            )

            logits = model(audio, visual, text)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())

    metrics = compute_metrics(all_labels, all_preds)

    print("\nEvaluation Metrics:")
    print(f"Accuracy        : {metrics['accuracy']:.4f}")
    print(f"Macro Precision : {metrics['macro_precision']:.4f}")
    print(f"Macro Recall    : {metrics['macro_recall']:.4f}")
    print(f"Macro F1        : {metrics['macro_f1']:.4f}")
    print(f"Weighted F1     : {metrics['weighted_f1']:.4f}")


if __name__ == "__main__":
    main()
