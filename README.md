# MEMOCMT

MEMOCMT is a research codebase for multimodal emotion recognition in
conversations, organized into two separate implementations using the
CMU-MOSI and MELD datasets respectively.

------------------------------------------------------------------------

## Repository Layout

    MEMOCMT/
    ├── MEMOCMT_With_CMU_MOSI_Dataset/   # Experiments and code for CMU-MOSI
    ├── MEMOCMT_With_MELD_Dataset/       # Experiments and code for MELD
    └── README.md                        # This file

------------------------------------------------------------------------

# Research Journey Overview

This project represents a progressive experimental journey in Multimodal
Emotion Recognition (MER):

1.  **Phase 1 -- Synthetic Data Validation**
2.  **Phase 2 -- CMU-MOSEI (Primary Benchmark Attempt)**
3.  **Phase 3 -- CMU-MOSI (Three Experimental Pipelines)**
4.  **Phase 4 -- MELD (Dialogue-Centric Emotion Recognition)**

The architecture evolved through these stages into the final
AVT-Improved MemoCMT model.

------------------------------------------------------------------------

# Phase 1 -- Synthetic Dataset Validation

## Objective

Validate cross-modal alignment logic before moving to real-world
datasets.

## Approach

-   Generated 1,000 synthetic samples.
-   Randomized audio and visual features around sentiment-conditioned
    means.
-   Text embeddings simulated using random vectors.

## Result

-   Confirmed 256D projection alignment works.
-   Verified CMT attention mechanism functions correctly.

## Limitation

-   No real-world noise or conversational context.
-   Used only for structural sanity check.

------------------------------------------------------------------------

# Phase 2 -- CMU-MOSEI (Primary Dataset Attempt)

## Why MOSEI?

-   Largest tri-modal dataset (\~23k segments).
-   Ideal for large Transformer-based models.

## Download

Official: https://github.com/A2Zadeh/CMU-MultimodalSDK

## Issues Faced

-   `.csd` data format required heavy SDK processing.
-   Memory overflow during multimodal alignment.
-   Required \>100GB RAM for full sequence alignment.
-   Cloud compute limits exceeded.

## Decision

Pivoted to CMU-MOSI due to manageable dataset size.

------------------------------------------------------------------------

# Phase 3 -- CMU-MOSI Implementation

## Dataset Details

-   2,199 video segments.
-   Sentiment range: -3 to +3.
-   Opinion-level annotations.

## Download

Official: https://github.com/A2Zadeh/CMU-MultimodalSDK

Raw Video Data: http://immortal.multicomp.cs.cmu.edu/raw_datasets/

------------------------------------------------------------------------

## Three Experimental Pipelines

### 1️⃣ Binary Classification

-   Positive vs Negative
-   Training Accuracy: 98.38%
-   Test Accuracy: \~86%
-   ROC-AUC: 0.76

### 2️⃣ Multiclass Classification

-   7-class sentiment intensity
-   Accuracy: 36.67%
-   Observed confusion in adjacent intensity classes

### 3️⃣ Regression

-   Continuous score prediction (-3 to +3)
-   Achieved competitive MAE vs late-fusion baselines

------------------------------------------------------------------------

# Faculty Feedback & Research Pivot

Mid-Semester Feedback:

> "Prioritize MELD dataset because dialogue-centric data introduces
> inter-person variability."

This shifted research toward conversational emotion modeling.

------------------------------------------------------------------------

# Phase 4 -- MELD Implementation

## Dataset Overview

-   13,000+ utterances
-   1,433 dialogues
-   7 emotion classes:
    -   Neutral
    -   Joy
    -   Sadness
    -   Anger
    -   Fear
    -   Disgust
    -   Surprise

## Download

Official GitHub: https://github.com/declare-lab/MELD

Direct Download: https://huggingface.co/datasets/declare-lab/MELD

------------------------------------------------------------------------

# MELD Implementation Details

## Feature Extraction

  Modality   Backbone    Output Dimension
  ---------- ----------- ------------------
  Audio      HuBERT      768
  Visual     ResNet-50   2048
  Text       BERT        768

All projected to 256D shared latent space.

------------------------------------------------------------------------

## Architecture

AVT-Improved MemoCMT includes:

-   Modality Encoders
-   Cross-Modal Transformer (6-way attention)
-   Temporal Transformer
-   Classification Head

------------------------------------------------------------------------

## MELD Results

-   Strong weighted F1-score.
-   High performance on Joy and Anger.
-   Lower recall for Fear due to class imbalance.

------------------------------------------------------------------------

# Explainability

Includes:

-   Cross-modal attention heatmaps
-   Grad-CAM visual saliency
-   Integrated Gradients for text
-   Emotion-wise modality analysis
-   Composite IEEE-ready figures

------------------------------------------------------------------------

# Final Conclusion

The project demonstrates:

-   Progressive dataset-driven architectural refinement.
-   Strong tri-modal fusion capability.
-   Robust dialogue-centric emotion recognition.
-   Explainable AI integration for interpretability.

------------------------------------------------------------------------

# Future Work

-   Larger dialogue datasets (e.g., MOSEI revisit)
-   Multilingual emotion recognition
-   Real-time inference deployment
-   Speaker-aware contextual modeling

------------------------------------------------------------------------

**Author:** Aparna Ram K\
**Supervisor:** Kilaru VamsiKrishna\
**Project Title:** AI-Driven Multimodal Affective Computing using Audio,
Visual, and Text Cues
