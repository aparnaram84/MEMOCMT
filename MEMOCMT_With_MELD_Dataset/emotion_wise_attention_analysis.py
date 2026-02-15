import json
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

ATTN_FILE = "experiments/attention_dump.json"
OUT_DIR = "experiments/emotion_analysis"
os.makedirs(OUT_DIR, exist_ok=True)

EMOTIONS = [
    "neutral", "joy", "sadness",
    "anger", "fear", "disgust", "surprise"
]

# -------------------------
# Load data
# -------------------------
with open(ATTN_FILE, "r") as f:
    records = json.load(f)

# -------------------------
# Aggregation
# -------------------------
def aggregate(records):
    agg = defaultdict(lambda: {
        "audio": [], "visual": [], "text": [],
        "conf_audio": [], "conf_visual": [], "conf_text": [],
        "correct": [], "incorrect": []
    })

    for r in records:
        emo = r["true_label"]
        conf = r["confidence"]
        correct = (r["pred_label"] == r["true_label"])

        agg[emo]["audio"].append(r["attn_audio"])
        agg[emo]["visual"].append(r["attn_visual"])
        agg[emo]["text"].append(r["attn_text"])

        agg[emo]["conf_audio"].append(r["attn_audio"] * conf)
        agg[emo]["conf_visual"].append(r["attn_visual"] * conf)
        agg[emo]["conf_text"].append(r["attn_text"] * conf)

        (agg[emo]["correct"] if correct else agg[emo]["incorrect"]).append(
            r["attn_visual"]
        )

    return agg

agg = aggregate(records)

# -------------------------
# Plot helpers
# -------------------------
def plot_bar(values, title, fname):
    plt.figure(figsize=(4, 3))
    plt.bar(["Audio", "Visual", "Text"], values)
    plt.ylim(0, 1)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()

# -------------------------
# 1️⃣ Emotion-wise modality contribution
# -------------------------
for emo in EMOTIONS:
    if emo not in agg:
        continue

    plot_bar(
        [
            np.mean(agg[emo]["audio"]),
            np.mean(agg[emo]["visual"]),
            np.mean(agg[emo]["text"])
        ],
        f"Modality Contribution – {emo}",
        f"{OUT_DIR}/{emo}_modality_contribution.png"
    )

# -------------------------
# 2️⃣ Confidence-weighted contribution
# -------------------------
for emo in EMOTIONS:
    if emo not in agg:
        continue

    plot_bar(
        [
            np.mean(agg[emo]["conf_audio"]),
            np.mean(agg[emo]["conf_visual"]),
            np.mean(agg[emo]["conf_text"])
        ],
        f"Confidence-Weighted Attention – {emo}",
        f"{OUT_DIR}/{emo}_confidence_weighted.png"
    )

# -------------------------
# 3️⃣ Correct vs Incorrect comparison
# -------------------------
for emo in EMOTIONS:
    if emo not in agg:
        continue
    if not agg[emo]["incorrect"]:
        continue

    plt.figure(figsize=(4, 3))
    plt.boxplot(
        [agg[emo]["correct"], agg[emo]["incorrect"]],
        tick_labels=["Correct", "Incorrect"]
    )
    plt.title(f"Visual Attention vs Accuracy – {emo}")
    plt.ylabel("Visual Attention")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/{emo}_correct_vs_incorrect.png", dpi=300)
    plt.close()

print("✅ Emotion-wise explainability analysis completed.")
