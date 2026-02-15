"""
extract_audio_features_dataset.py

Dataset-level audio feature extraction for MELD

This script:
- Iterates over MELD train/dev/test splits
- Loads pre-extracted WAV files (robust to codec issues)
- Uses HuBERT to extract audio embeddings
- Produces ONE fixed-length embedding per utterance
- Saves features to disk (.pt)

OUTPUT:
features/audio/{train,dev,test}.pt

NOTE:
- This is REQUIRED before training
- Features are NOT computed on-the-fly during training
"""

import os
import torch
import torchaudio
import soundfile as sf
from tqdm import tqdm
from transformers import HubertModel

# ---------------- Configuration ----------------
MELD_ROOT = "data/MELD"
AUDIO_SUBDIR = "audio"
SPLITS = ["train", "dev", "test"]
OUT_DIR = "features/audio"
TARGET_SR = 16000
# ------------------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = HubertModel.from_pretrained("facebook/hubert-base-ls960")
model.to(device)
model.eval()

def load_audio_safe(wav_path):
    """
    Robust audio loader:
    - First tries torchaudio
    - Falls back to soundfile for problematic WAVs
    """
    try:
        return torchaudio.load(wav_path)
    except Exception:
        audio, sr = sf.read(wav_path)
        if audio.ndim == 1:
            audio = audio[None, :]
        else:
            audio = audio.T
        return torch.tensor(audio, dtype=torch.float32), sr

def extract_audio_embedding(wav_path):
    waveform, sr = load_audio_safe(wav_path)

    if sr != TARGET_SR:
        waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)

    waveform = waveform.to(device)

    with torch.no_grad():
        outputs = model(waveform)
        hidden_states = outputs.last_hidden_state
        embedding = hidden_states.mean(dim=1).squeeze(0)

    return embedding.cpu()

def run():
    for split in SPLITS:
        audio_dir = os.path.join(MELD_ROOT, split, AUDIO_SUBDIR)
        out_path = os.path.join(OUT_DIR, f"{split}.pt")

        if not os.path.exists(audio_dir):
            print(f"[WARN] Missing audio dir: {audio_dir}")
            continue

        features = {}
        wav_files = sorted(f for f in os.listdir(audio_dir) if f.endswith(".wav"))

        print(f"[{split}] Extracting audio features ({len(wav_files)} files)")

        for wav_file in tqdm(wav_files):
            utt_id = os.path.splitext(wav_file)[0]
            wav_path = os.path.join(audio_dir, wav_file)

            try:
                emb = extract_audio_embedding(wav_path)
                features[utt_id] = emb
            except Exception as e:
                print(f"[SKIP] {utt_id}: {e}")

        torch.save(features, out_path)
        print(f"[{split}] Saved audio features to {out_path}")

if __name__ == "__main__":
    run()
