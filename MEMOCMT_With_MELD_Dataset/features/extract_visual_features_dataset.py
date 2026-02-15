"""
extract_visual_features_dataset.py

Dataset-level visual feature extraction for MELD

This script:
- Iterates over MELD train/dev/test splits
- Loads pre-extracted video frames
- Uses ResNet-50 to extract visual embeddings
- Aggregates frame-level features to ONE utterance-level embedding
- Saves features to disk (.pt)

INPUT:
data/MELD/<split>/frames/<utterance_id>/frame_XXXX.jpg

OUTPUT:
features/visual/{train,dev,test}.pt

NOTE:
- REQUIRED before training
- Does NOT read raw video files
- To use face-based frames, set FRAME_DIR_NAME = "face_frames"
"""

import os
import torch
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

# ---------------- Configuration ----------------
MELD_ROOT = "data/MELD"
SPLITS = ["train", "dev", "test"]
FRAME_DIR_NAME = "frames"     # change to "face_frames" if needed
OUT_DIR = "features/visual"
IMAGE_EXT = ".jpg"
# ------------------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load ResNet-50 backbone
model = models.resnet50(pretrained=True)
model.fc = torch.nn.Identity()   # remove classification head
model = model.to(device)
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def extract_frame_feature(img_path):
    img = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        feat = model(tensor).squeeze(0)

    return feat.cpu()

def run():
    for split in SPLITS:
        frame_root = os.path.join(MELD_ROOT, split, FRAME_DIR_NAME)
        out_path = os.path.join(OUT_DIR, f"{split}.pt")

        if not os.path.exists(frame_root):
            print(f"[WARN] Missing frames dir: {frame_root}")
            continue

        features = {}
        utterances = [
            d for d in os.listdir(frame_root)
            if os.path.isdir(os.path.join(frame_root, d))
        ]

        print(f"[{split}] Extracting visual features ({len(utterances)} utterances)")

        for utt_id in tqdm(utterances):
            utt_dir = os.path.join(frame_root, utt_id)
            frame_files = sorted(
                f for f in os.listdir(utt_dir)
                if f.endswith(IMAGE_EXT)
            )

            if not frame_files:
                continue

            frame_feats = []
            for frame in frame_files:
                frame_path = os.path.join(utt_dir, frame)
                try:
                    frame_feats.append(extract_frame_feature(frame_path))
                except Exception as e:
                    print(f"[SKIP] {utt_id}/{frame}: {e}")

            if frame_feats:
                utterance_feat = torch.stack(frame_feats).mean(dim=0)
                features[utt_id] = utterance_feat

        torch.save(features, out_path)
        print(f"[{split}] Saved visual features to {out_path}")

if __name__ == "__main__":
    run()
