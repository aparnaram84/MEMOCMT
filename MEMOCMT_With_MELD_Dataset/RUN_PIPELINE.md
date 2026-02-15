# run_pipeline.sh – Full Execution Pipeline Documentation

This document explains the purpose, structure, and execution flow of  
`run_pipeline.sh`, which automates the **entire AVT-Improved MemoCMT pipeline** on the MELD dataset.

This file is suitable for:
- GitHub documentation
- Thesis appendix
- Reviewer and viva clarification

---

## 📌 Purpose of `run_pipeline.sh`

`run_pipeline.sh` provides a **single-command, end-to-end execution pipeline** that:

- Ensures full reproducibility
- Eliminates manual execution errors
- Standardizes experiments across environments

It executes **all stages** from raw MELD videos (`.mp4`) to final evaluation and explainability outputs.

---

## 🧱 High-Level Pipeline Overview

```text
MP4 Videos (MELD)
   │
   ├── Audio Extraction → VAD
   ├── Video Frame Extraction → Face Detection
   ├── Text Processing (CSV)
   │
   ├── Feature Extraction (A / V / T)
   │
   ├── Model Training
   │
   ├── Evaluation
   │
   └── Explainability (Attention Maps)
```

---

## 🛠 Prerequisites

Before running the script, ensure:

- Conda environment is activated
- `ffmpeg` is installed and accessible
- MELD dataset is placed under:
  ```
  data/MELD/
  ```

---

## 📄 Script Breakdown (Step-by-Step)

### STEP 1: Audio Extraction from MP4
```bash
python preprocessing/audio_preprocess.py
```
- Extracts audio streams from MELD video files
- Converts audio to mono, 16 kHz

---

### STEP 2: Voice Activity Detection (VAD)
```bash
python preprocessing/audio_preprocess_vad.py
```
- Removes non-speech segments
- Improves robustness to silence and noise

---

### STEP 3: Video Frame Extraction
```bash
python preprocessing/video_preprocess.py
```
- Samples frames from video clips
- Produces frame-level visual input

---

### STEP 4: Face Detection (Optional)
```bash
python preprocessing/video_preprocess_mtcnn.py
```
- Crops facial regions using MTCNN
- Improves facial emotion focus

*This step is optional and can be disabled if needed.*

---

### STEP 5: Text Preprocessing
```bash
python preprocessing/text_preprocess.py
```
- Cleans utterances from MELD CSV annotations
- Standardizes textual input

---

### STEP 6: Feature Extraction
```bash
python features/extract_audio_features.py
python features/extract_visual_features.py
python features/extract_text_features.py
```
- Converts signals into embeddings
- Prepares input for Cross-Modal Transformer

---

### STEP 7: Model Training
```bash
python train.py
```
- Trains the AVT-Improved MemoCMT model
- Saves checkpoints and logs

---

### STEP 8: Evaluation
```bash
python evaluate.py
```
- Computes Accuracy, F1-score, and UAR
- Generates confusion matrix

---

### STEP 9: Explainability
```bash
python explainability/attention_visualization.py
```
- Generates cross-modal attention maps
- Saves visualizations for analysis

---

## 📂 Outputs Generated

After successful execution:

```text
experiments/
├── results.csv
├── confusion_matrix.png
└── attention_maps/
```

---

## 🧠 Architectural Clarification (Important)

- `run_pipeline.sh` **does not modify the architecture**
- It only orchestrates execution of existing modules
- All architectural novelty remains in:
  - Cross-Modal Transformer
  - Temporal Transformer
  - AVT fusion strategy

---

## 🧪 Reproducibility Notes

- Dataset splits unchanged
- Pre-trained encoders documented
- Deterministic execution supported (seed-controlled)

---

## 🎓 Viva-Ready Explanation (Use Verbatim)

> “We provide a single shell script that automates the complete preprocessing, feature extraction, training, evaluation, and explainability pipeline, ensuring reproducibility and ease of experimentation.”

---

## ✅ Conclusion

`run_pipeline.sh` serves as the **execution backbone** of the project, transforming the proposed architecture into a fully reproducible and deployable system.

This file can be included as:
- `RUN_PIPELINE.md`
- Thesis appendix
- Supplementary material for publication
