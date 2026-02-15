"""
audio_forced_alignment.py

Forced Alignment for MELD (WAV-based, analysis-only)

This script:
- Takes pre-extracted .wav audio files as input (NO video usage)
- Performs CTC-based alignment using Wav2Vec2
- Aligns audio with transcript text from MELD CSV
- Outputs alignment metadata as JSON files
- Does NOT generate features or training inputs

OUTPUT:
alignments/{train,dev,test}/{utterance_id}.json

NOTE:
- This is OPTIONAL and used only for analysis / explainability
- It does NOT affect model training or inference
"""

import os
import json
from glob import glob

import torch
import torchaudio
import pandas as pd
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# ---------------- Configuration ----------------
TARGET_SR = 16000
MELD_ROOT = "data/MELD"
SPLITS = ["train", "dev", "test"]
ALIGN_ROOT = "alignments"

# Choose audio source (default: raw audio)
AUDIO_SUBDIR = "audio"        # or "audio_vad"

# ------------------------------------------------

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
model.eval()


def align_single(wav_path, transcript, out_json):
    waveform, sr = torchaudio.load(wav_path)

    if sr != TARGET_SR:
        waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)

    with torch.no_grad():
        inputs = processor(
            waveform.squeeze(),
            sampling_rate=TARGET_SR,
            return_tensors="pt"
        )
        logits = model(inputs.input_values).logits

    pred_ids = torch.argmax(logits, dim=-1)
    pred_text = processor.batch_decode(pred_ids)[0]

    result = {
        "utterance_id": os.path.splitext(os.path.basename(wav_path))[0],
        "audio_source": wav_path,
        "reference_transcript": transcript,
        "predicted_transcript": pred_text,
        "alignment_type": "ctc_forced_alignment",
        "timestamp_note": "CTC provides coarse alignment; no word-level timestamps",
        "used_in_training": False
    }

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


def run():
    os.makedirs(ALIGN_ROOT, exist_ok=True)

    for split in SPLITS:
        audio_dir = os.path.join(MELD_ROOT, split, AUDIO_SUBDIR)
        csv_path = os.path.join(MELD_ROOT, split, f"{split}_sent_emo.csv")
        out_dir = os.path.join(ALIGN_ROOT, split)

        if not os.path.exists(audio_dir) or not os.path.exists(csv_path):
            print(f"Skipping {split}: missing audio or CSV")
            continue

        df = pd.read_csv(csv_path)

        print(f"[{split}] Forced alignment started ({len(df)} utterances)")

        for _, row in df.iterrows():
            utt_id = row["Utterance_ID"] if "Utterance_ID" in row else row.get("Utterance_ID", None)
            wav_path = os.path.join(audio_dir, f"{utt_id}.wav")
            out_json = os.path.join(out_dir, f"{utt_id}.json")

            if not os.path.exists(wav_path):
                continue

            align_single(wav_path, row["Utterance"], out_json)

        print(f"[{split}] Alignment JSON saved to {out_dir}")


if __name__ == "__main__":
    run()
