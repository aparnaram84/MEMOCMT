"""
video_preprocess.py

Video preprocessing for MELD (MP4-based)

This script:
- Iterates over MELD train/dev/test splits
- Reads .mp4 video files
- Samples frames at a fixed FPS
- Saves frames per utterance

OUTPUT:
data/MELD/<split>/frames/<utterance_id>/frame_XXXX.jpg

NOTE:
- This is REQUIRED for the visual modality
- Face detection is NOT done here (handled separately)
"""

import os
import cv2
from glob import glob

MELD_ROOT = "data/MELD"
SPLITS = ["train", "dev", "test"]
FRAME_FPS = 1
IMAGE_EXT = ".jpg"

def extract_frames(video_path, out_dir, fps=1):
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 25

    frame_interval = max(1, int(round(video_fps / fps)))
    frame_count = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            out_path = os.path.join(out_dir, f"frame_{saved:04d}{IMAGE_EXT}")
            cv2.imwrite(out_path, frame)
            saved += 1

        frame_count += 1

    cap.release()
    return saved

def run():
    for split in SPLITS:
        video_dir = os.path.join(MELD_ROOT, split, "videos")
        frame_root = os.path.join(MELD_ROOT, split, "frames")

        if not os.path.exists(video_dir):
            print(f"[WARN] {split}: videos folder not found")
            continue

        os.makedirs(frame_root, exist_ok=True)

        video_files = glob(os.path.join(video_dir, "*.mp4"))
        print(f"[{split}] Found {len(video_files)} videos")

        total_frames = 0
        skipped = 0

        for video_path in video_files:
            utt_id = os.path.splitext(os.path.basename(video_path))[0]
            out_dir = os.path.join(frame_root, utt_id)

            saved = extract_frames(video_path, out_dir, FRAME_FPS)
            if saved == 0:
                skipped += 1
            total_frames += saved

        print(f"[{split}] Frames saved: {total_frames}")
        print(f"[{split}] Videos skipped (unreadable): {skipped}")

    print("Video preprocessing completed successfully.")

if __name__ == "__main__":
    run()
