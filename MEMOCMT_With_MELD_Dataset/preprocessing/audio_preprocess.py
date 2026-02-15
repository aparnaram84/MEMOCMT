"""
Robust Audio preprocessing for MELD (MP4-based)

- Extracts audio from .mp4 files
- Saves .wav files under data/MELD/<split>/audio/
- Skips corrupted videos safely
- Logs failed files
"""

import os
import subprocess
import shutil
from glob import glob
import sys

TARGET_SR = 16000
MELD_ROOT = "data/MELD"
SPLITS = ["train", "dev", "test"]

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

def check_ffmpeg():
    if shutil.which("ffmpeg") is None:
        print("\n❌ ERROR: ffmpeg not found in PATH")
        print("Install ffmpeg and add it to PATH")
        sys.exit(1)

def extract_audio_from_mp4(video_path, output_wav):
    os.makedirs(os.path.dirname(output_wav), exist_ok=True)

    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vn",
            "-ac", "1",
            "-ar", str(TARGET_SR),
            output_wav
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    return result.returncode == 0

def run():
    check_ffmpeg()

    for split in SPLITS:
        video_dir = os.path.join(MELD_ROOT, split, "videos")
        audio_dir = os.path.join(MELD_ROOT, split, "audio")
        log_file = os.path.join(LOG_DIR, f"audio_failures_{split}.txt")

        if not os.path.exists(video_dir):
            print(f"⚠️ Skipping {split}: videos folder not found")
            continue

        os.makedirs(audio_dir, exist_ok=True)

        video_files = glob(os.path.join(video_dir, "*.mp4"))
        print(f"[{split}] Found {len(video_files)} videos")

        failed = 0

        with open(log_file, "w") as log:
            for video_path in video_files:
                fname = os.path.splitext(os.path.basename(video_path))[0]
                out_wav = os.path.join(audio_dir, f"{fname}.wav")

                success = extract_audio_from_mp4(video_path, out_wav)

                if not success:
                    failed += 1
                    log.write(video_path + "\n")

        print(f"✅ {split}: Audio extracted")
        print(f"⚠️ {failed} videos skipped (see {log_file})")

if __name__ == "__main__":
    run()
