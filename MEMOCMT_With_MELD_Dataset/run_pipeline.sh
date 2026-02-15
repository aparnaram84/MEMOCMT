#!/bin/bash
# =====================================================
# Full Execution Pipeline for AVT-Improved MemoCMT (MELD)
# =====================================================
# This script runs the complete pipeline:
# MP4 -> Audio/Video/Text preprocessing -> Feature extraction
# -> Training -> Evaluation -> Explainability
#
# Prerequisites:
# - conda environment activated
# - ffmpeg installed
# - MELD dataset placed under data/MELD/
# =====================================================

set -e

echo "=========================================="
echo "AVT-Improved MemoCMT : Full Pipeline Start"
echo "=========================================="

# -------- STEP 0: Check environment --------
command -v ffmpeg >/dev/null 2>&1 || { echo >&2 "ffmpeg not installed. Aborting."; exit 1; }

# -------- STEP 1: Audio extraction from MP4 --------
echo "[1/8] Extracting audio from MP4 videos..."
python preprocessing/audio_preprocess.py

# -------- STEP 2: Audio cleaning with VAD --------
echo "[2/8] Applying Voice Activity Detection (VAD)..."
python preprocessing/audio_preprocess_vad.py

# -------- STEP 3: Video frame extraction --------
echo "[3/8] Extracting video frames..."
python preprocessing/video_preprocess.py

# -------- STEP 4: Face detection (optional) --------
echo "[4/8] Running face detection (MTCNN)..."
python preprocessing/video_preprocess_mtcnn.py

# -------- STEP 5: Text preprocessing from CSV --------
echo "[5/8] Preprocessing MELD text annotations..."
python preprocessing/text_preprocess.py

# -------- STEP 6: Feature extraction --------
echo "[6/8] Extracting Audio features (HuBERT)..."
python features/extract_audio_features.py

echo "[6/8] Extracting Visual features (ResNet)..."
python features/extract_visual_features.py

echo "[6/8] Extracting Text features (BERT)..."
python features/extract_text_features.py

# -------- STEP 7: Model training --------
echo "[7/8] Training AVT-Improved MemoCMT..."
python train.py

# -------- STEP 8: Evaluation & Explainability --------
echo "[8/8] Evaluating model and generating plots..."
python evaluate.py

echo "[8/8] Generating attention visualizations..."
python explainability/attention_visualization.py

echo "=========================================="
echo "Pipeline completed successfully!"
echo "Results available in experiments/ folder"
echo "=========================================="
