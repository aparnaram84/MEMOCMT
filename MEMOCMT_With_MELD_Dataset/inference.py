"""
Inference module for AVT-Improved MemoCMT (MELD)

Features:
- Dev/Test inference
- Utterance-level attention dumping (normalized)
- Optional explainability (Attention, Grad-CAM, Integrated Gradients)
- Compatible with utterance-level (T=1) models
"""

import os
import json
import argparse
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm

from models.avt_memocmt import AVT_MemoCMT
from explainability.attention_visualization import plot_attention
from explainability.gradcam import compute_gradcam
from explainability.integrated_gradients import integrated_gradients


# ---------------- CONFIG ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 7
D_MODEL = 256

EMOTION_NAMES = [
    "neutral", "joy", "sadness",
    "anger", "fear", "disgust", "surprise"
]

LABEL_MAP = {
    "neutral": 0,
    "joy": 1,
    "sadness": 2,
    "anger": 3,
    "fear": 4,
    "disgust": 5,
    "surprise": 6
}

FEATURE_ROOT = "features"
EXPLAIN_OUT = "experiments/explainability"
ATTN_DUMP_PATH = "experiments/attention_dump.json"

os.makedirs(EXPLAIN_OUT, exist_ok=True)
os.makedirs("experiments", exist_ok=True)
# --------------------------------------


def normalize(a, v, t):
    s = a + v + t + 1e-8
    return a / s, v / s, t / s


def load_model(checkpoint_path):
    model = AVT_MemoCMT(d_model=D_MODEL, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def load_labels_from_csv(split):
    """
    Loads ground-truth labels directly from MELD CSV.
    """
    csv_path = f"data/MELD/{split}/{split}_sent_emo.csv"
    df = pd.read_csv(csv_path)

    labels = {}
    for _, row in df.iterrows():
        utt_id = f"dia{row.Dialogue_ID}_utt{row.Utterance_ID}"
        labels[utt_id] = LABEL_MAP[row.Emotion.lower()]

    return labels


def run_inference(
    model,
    audio_feat,
    visual_feat,
    text_feat,
    label,
    utt_id,
    explain=False,
    attention_records=None,
):
    """
    Runs inference on a single utterance and optionally performs explainability.
    """

    audio = audio_feat.unsqueeze(0).to(DEVICE)
    visual = visual_feat.unsqueeze(0).to(DEVICE)
    text = text_feat.unsqueeze(0).to(DEVICE)

    # ---------- Forward pass ----------
    with torch.no_grad():
        logits = model(audio, visual, text)
        probs = F.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred].item()

    pred_label = EMOTION_NAMES[pred]
    true_label = EMOTION_NAMES[label]

    # ---------- Attention extraction ----------
    if attention_records is not None:
        # Modality contribution proxies (utterance-level)
        attn_audio = audio.norm(dim=1).mean().item()
        attn_visual = visual.norm(dim=1).mean().item()
        attn_text = text.norm(dim=1).mean().item()

        attn_audio, attn_visual, attn_text = normalize(
            attn_audio, attn_visual, attn_text
        )

        attention_records.append({
            "utterance_id": utt_id,
            "pred_label": pred_label,
            "true_label": true_label,
            "confidence": confidence,
            "attn_audio": attn_audio,
            "attn_visual": attn_visual,
            "attn_text": attn_text
        })

    # ---------- Explainability (limited examples) ----------
    if explain and confidence >= 0.5:
        print(f"🔍 Explainability for {utt_id} ({pred_label}, {confidence:.2f})")

        if hasattr(model.cmt, "last_attn") and model.cmt.last_attn is not None:
            plot_attention(
                model.cmt.last_attn,
                os.path.join(EXPLAIN_OUT, f"{utt_id}_attention.png")
            )

        compute_gradcam(
            visual,
            os.path.join(EXPLAIN_OUT, f"{utt_id}_gradcam.png")
        )

        integrated_gradients(
            model,
            audio,
            visual,
            text,
            os.path.join(EXPLAIN_OUT, f"{utt_id}_ig.png")
        )


def run_split_inference(
    model,
    split="dev",
    explain=False,
    max_explain=5
):
    print(f"\n🔍 Running inference on {split} split | explain={explain}")

    audio_feats = torch.load(f"{FEATURE_ROOT}/audio/{split}.pt")
    visual_feats = torch.load(f"{FEATURE_ROOT}/visual/{split}.pt")
    text_feats = torch.load(f"{FEATURE_ROOT}/text/{split}.pt")
    labels = load_labels_from_csv(split)

    common_keys = (
        audio_feats.keys()
        & visual_feats.keys()
        & text_feats.keys()
        & labels.keys()
    )

    print(f"[INFO] Using {len(common_keys)} aligned utterances")

    attention_records = []
    explain_count = 0

    for utt_id in tqdm(sorted(common_keys)):
        run_inference(
            model,
            audio_feats[utt_id],
            visual_feats[utt_id],
            text_feats[utt_id],
            labels[utt_id],
            utt_id=utt_id,
            explain=explain and explain_count < max_explain,
            attention_records=attention_records,
        )

        if explain and explain_count < max_explain:
            explain_count += 1

    with open(ATTN_DUMP_PATH, "w") as f:
        json.dump(attention_records, f, indent=2)

    print(f"📁 Attention dump saved to {ATTN_DUMP_PATH}")
    print("✅ Inference completed")


# ---------------- CLI ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--checkpoint", default="avt_memocmt_final.pt")
    parser.add_argument("--explain", action="store_true")
    parser.add_argument("--max_explain", type=int, default=5)

    args = parser.parse_args()

    model = load_model(args.checkpoint)
    run_split_inference(
        model,
        split=args.split,
        explain=args.explain,
        max_explain=args.max_explain
    )
