"""
audio_preprocess_vad.py

Optional Audio Preprocessing with VAD for MELD (MP4-based)

This script:
- Extracts audio from MELD .mp4 files
- Applies Voice Activity Detection (WebRTC VAD)
- Skips corrupted videos safely
- Logs failures instead of crashing

IMPORTANT:
- This script is OPTIONAL
- It is NOT required by the AVT-Improved MemoCMT architecture
- Default experiments can run without this step
"""

import os
import subprocess
import shutil
from glob import glob
import sys
import librosa
import soundfile as sf
import webrtcvad
import numpy as np

TARGET_SR = 16000
FRAME_DURATION_MS = 30
MELD_ROOT = "data/MELD"
SPLITS = ["train", "dev", "test"]

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

vad = webrtcvad.Vad(2)

def check_ffmpeg():
    if shutil.which("ffmpeg") is None:
        print("\n❌ ERROR: ffmpeg not found in PATH")
        print("👉 Install ffmpeg and add it to PATH")
        print("👉 Windows guide: https://www.gyan.dev/ffmpeg/builds/")
        sys.exit(1)

def extract_audio_from_mp4(video_path, wav_path):
    os.makedirs(os.path.dirname(wav_path), exist_ok=True)
    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vn",
            "-ac", "1",
            "-ar", str(TARGET_SR),
            wav_path
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    return result.returncode == 0

def apply_vad(wav_path, output_wav):
    audio, _ = librosa.load(wav_path, sr=TARGET_SR)
    audio_int16 = (audio * 32768).astype(np.int16)

    frame_len = int(TARGET_SR * FRAME_DURATION_MS / 1000)
    voiced_audio = []

    for i in range(0, len(audio_int16), frame_len):
        frame = audio_int16[i:i + frame_len]
        if len(frame) == frame_len and vad.is_speech(frame.tobytes(), TARGET_SR):
            voiced_audio.extend(frame)

    if len(voiced_audio) == 0:
        return False

    voiced_audio = np.array(voiced_audio, dtype=np.int16) / 32768.0
    sf.write(output_wav, voiced_audio, TARGET_SR)
    return True

def run():
    check_ffmpeg()

    for split in SPLITS:
        video_dir = os.path.join(MELD_ROOT, split, "videos")
        audio_dir = os.path.join(MELD_ROOT, split, "audio_vad")
        temp_dir = os.path.join(audio_dir, "_tmp")
        log_file = os.path.join(LOG_DIR, f"audio_vad_failures_{split}.txt")

        if not os.path.exists(video_dir):
            print(f"⚠️ Skipping {split}: videos folder not found")
            continue

        os.makedirs(audio_dir, exist_ok=True)
        os.makedirs(temp_dir, exist_ok=True)

        video_files = glob(os.path.join(video_dir, "*.mp4"))
        print(f"[{split}] Found {len(video_files)} videos (VAD optional)")

        failed = 0

        with open(log_file, "w") as log:
            for video_path in video_files:
                name = os.path.splitext(os.path.basename(video_path))[0]
                temp_wav = os.path.join(temp_dir, f"{name}.wav")
                out_wav = os.path.join(audio_dir, f"{name}.wav")

                ok = extract_audio_from_mp4(video_path, temp_wav)
                if not ok:
                    failed += 1
                    log.write(f"{video_path} | audio extract failed\n")
                    continue

                ok = apply_vad(temp_wav, out_wav)
                if not ok:
                    failed += 1
                    log.write(f"{video_path} | VAD produced empty audio\n")

        shutil.rmtree(temp_dir, ignore_errors=True)

        print(f"✅ {split}: VAD audio generated")
        print(f"⚠️ {failed} files skipped (see {log_file})")

if __name__ == "__main__":
    run()
