import os
import json
import matplotlib.pyplot as plt
import librosa.display

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.data_resolver import (
    get_video_frame,
    get_audio_spectrogram,
    get_transcript
)

# ---------------- CONFIG ----------------
ATTN_JSON = "experiments/attention_dump.json"
SPLIT = "dev"

OUT_DIR = "experiments/composite_figures"
os.makedirs(OUT_DIR, exist_ok=True)

# EXACT utterances selected for thesis
SELECTED_UTTS = {
    "dia5_utt6",     # Correct – Anger
    "dia82_utt13",   # Correct – Joy
    "dia30_utt3",    # Correct – Sadness
    "dia94_utt5",    # Failure case
}
# --------------------------------------


def plot_composite(record, split):
    utt = record["utterance_id"]

    fig, axes = plt.subplots(
        4, 1,
        figsize=(3.5, 8.0),   # IEEE single-column width
        gridspec_kw={"height_ratios": [2, 2, 1, 1]}
    )

    # ========== VIDEO ==========
    frame = get_video_frame(utt, split)
    axes[0].axis("off")
    if frame is not None:
        axes[0].imshow(frame)
        axes[0].set_title("Visual Modality", fontsize=10)
    else:
        axes[0].text(
            0.5, 0.5, "Video unavailable",
            ha="center", va="center", fontsize=9
        )

    # ========== AUDIO ==========
    spec = get_audio_spectrogram(utt, split)
    axes[1].axis("off")
    if spec is not None:
        librosa.display.specshow(
            spec,
            x_axis=None,
            y_axis=None,
            ax=axes[1]
        )
        axes[1].set_title("Acoustic Modality (Mel-Spectrogram)", fontsize=10)
    else:
        axes[1].text(
            0.5, 0.5, "Audio unavailable",
            ha="center", va="center", fontsize=9
        )

    # ========== TEXT ==========
    text = get_transcript(utt, split)
    axes[2].axis("off")
    axes[2].text(
        0.01, 0.6,
        f'Text: "{text}"',
        fontsize=11,
        wrap=True,
        ha="left",
        va="top"
    )

    # ========== PREDICTION ==========
    axes[3].axis("off")
    axes[3].text(
        0.01, 0.65,
        f'Predicted: {record["pred_label"]} | '
        f'True: {record["true_label"]} | '
        f'Confidence: {record["confidence"]:.2f}',
        fontsize=9
    )
    axes[3].text(
        0.01, 0.30,
        f'Attention → '
        f'Audio: {record["attn_audio"]:.2f} | '
        f'Visual: {record["attn_visual"]:.2f} | '
        f'Text: {record["attn_text"]:.2f}',
        fontsize=9
    )

    plt.tight_layout()
    save_path = os.path.join(OUT_DIR, f"{utt}_composite.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"✅ Saved composite figure: {save_path}")


def main():
    assert os.path.exists(ATTN_JSON), f"Missing {ATTN_JSON}"

    with open(ATTN_JSON, "r") as f:
        records = json.load(f)

    for r in records:
        if r["utterance_id"] in SELECTED_UTTS:
            plot_composite(r, split=SPLIT)


if __name__ == "__main__":
    main()
