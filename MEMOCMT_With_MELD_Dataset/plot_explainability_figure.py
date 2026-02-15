"""
Generate FINAL publication-quality explainability figures
for selected utterances only.
"""

import json
import os
import matplotlib.pyplot as plt

ATTN_FILE = "experiments/attention_dump.json"
OUT_DIR = "experiments/final_explainability_figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ===== EXACT utterances to plot =====
SELECTED = {
    "Correct – Anger": "dia5_utt6",
    "Correct – Joy": "dia82_utt13",
    "Correct – Sadness": "dia30_utt3",
    "Failure Case": "dia94_utt5"
}

# -----------------------------------
with open(ATTN_FILE, "r") as f:
    records = {r["utterance_id"]: r for r in json.load(f)}

def plot_single(r, title):
    plt.figure(figsize=(4.5, 3.5))

    values = [r["attn_audio"], r["attn_visual"], r["attn_text"]]
    plt.bar(["Audio", "Visual", "Text"], values)

    plt.ylim(0, 1)
    plt.ylabel("Attention Weight")
    plt.title(title)

    caption = (
        f"UTT: {r['utterance_id']}\n"
        f"Pred: {r['pred_label']} | True: {r['true_label']}\n"
        f"Confidence: {r['confidence']:.2f}"
    )
    plt.xlabel(caption)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/{title.replace(' ', '_')}.png", dpi=300)
    plt.close()

# -----------------------------------
for title, utt_id in SELECTED.items():
    plot_single(records[utt_id], title)

print("✅ Final explainability figures saved to:", OUT_DIR)
